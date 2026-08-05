"""Centralised weather access for both training and forecasting.

Everything the model knows about the weather comes through :func:`get_weather`. There is
one variable vocabulary (:data:`CONVERSION_DICT`), one set of unit conversions, one
daily-aggregation rule per variable (:data:`DAILY_AGGREGATION`), and one code path that
turns an Open-Meteo response into an ``xarray.Dataset``.

That single-path property is the point of this module. The previous design had two
independent implementations of "get the weather" — Google Earth Engine ERA5 CSV exports for
training and the Open-Meteo forecast API for serving — and nothing compared them. Three
separate defects (wind speeds inflated 3.6x, a scrambled wind-direction convention, and
daily aggregation meaning three different things depending on the location) were all
instances of the same structural problem. See DEVELOPMENT.md section 5.2.

Three sources are available, all speaking the same vocabulary:

``"cache"``
    A local Parquet store built by ``scripts/build_weather_cache.py``. This is what
    training reads. No network access, so sweeps and repeated runs are fast and
    reproducible.
``"archive"``
    The Open-Meteo ERA5 archive API. Used to build the cache; not called during training.
``"forecast"``
    The Open-Meteo forecast API. Used by the daily prediction job.

A note on what unification does and does not buy us. It removes the unit, naming and
aggregation mismatches, and it removes a real elevation bias: Open-Meteo corrects
temperature and pressure to the requested point's DEM elevation (417 m at Defile), while
the GEE export returned the raw ERA5 cell (roughly 750 m), so the old training data sat
about 1.8 K and 3151 Pa away from what the forecast path serves. Both Open-Meteo endpoints
report the same elevation, so that bias is gone by construction. What unification does
*not* fix is that ERA5 is a reanalysis and a five-day forecast is not — that distribution
shift is a separate problem (DEVELOPMENT.md section 5.2).
"""

import warnings

import numpy as np
import pandas as pd
import requests_cache
import xarray as xr
from omegaconf import ListConfig
from openmeteo_requests import Client
from retry_requests import TSession, retry
from suncalc import get_position
from tqdm import tqdm

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# The archive is pinned to plain ERA5 (0.25 deg, 1940-present), which is the same product
# the retired GEE export used (ECMWF/ERA5/HOURLY). Leaving this unset would let Open-Meteo
# pick per date between ERA5, ERA5-Land (1950+), ECMWF IFS (2017+) and CERRA (1985-2021),
# silently changing dataset and resolution part-way through the training history.
ARCHIVE_MODEL = "era5"

# Read timeouts, in seconds. Archive requests span years and take far longer than a
# forecast request, so the two are budgeted separately.
ARCHIVE_TIMEOUT = 300
FORECAST_TIMEOUT = 60

# Default location of the Parquet cache, relative to the data directory.
CACHE_SUBDIR = "weather/era5_hourly"

LOCATIONS = {
    "Defile": (46.117215, 5.914877),
    "Basel": (47.561227, 7.603047),
    "MontTendre": (46.594542, 6.309533),
    "Chasseral": (47.132963, 7.058849),
    "ColGrandSaintBernard": (45.868846, 7.165453),
    "Schaffhausen": (47.697376, 8.634737),
    "Munich": (48.144714, 11.572036),
    "Dijon": (47.323463, 5.033933),
    "Frankfurt": (50.118894, 8.670885),
    "Stuttgart": (48.775951, 9.181457),
    "Berlin": (52.522825, 13.404231),
    "Prague": (50.072197, 14.436136),
    "Warsaw": (52.21639, 21.014928),
}


def get_lat_lon(locations):
    """Returns lists of latitudes and longitudes for a list of location names.

    Args:
        locations (str or list of str): A location name or a list of location names.

    Returns:
        tuple: Two lists - one for latitudes and one for longitudes.
    """
    locations = _as_list(locations)

    missing = [name for name in locations if name not in LOCATIONS]
    if missing:
        raise KeyError(
            f"Unknown location(s): {', '.join(missing)}. "
            f"Known locations: {', '.join(LOCATIONS)}"
        )

    latitudes = [LOCATIONS[name][0] for name in locations]
    longitudes = [LOCATIONS[name][1] for name in locations]

    return latitudes, longitudes


# Variables are named after the ERA5 bands the project has always used, so configs and
# checkpoints keep working. Each entry says which Open-Meteo variables it needs and how to
# convert them.
#
# Source units (Open-Meteo, with the units pinned in `_request_params`):
#   temperature/dew point  degC        precipitation  mm
#   pressure               hPa         wind speed     m/s (pinned; the API default is km/h)
#   wind direction         deg, meteorological (the direction the wind blows FROM)
#   cloud cover            %           shortwave radiation  W/m2 (hourly mean)
#
# Target units (ERA5):
#   temperature/dew point  K           precipitation  m
#   pressure               Pa          wind components  m/s
#   cloud cover            fraction    radiation      J/m2 accumulated over the hour
#
# `archive` marks whether the variable exists in the ERA5 archive. CAPE does not: the
# archive returns all-null for it, so it is forecast-only and requesting it for training
# raises rather than silently filling the feature with NaN.
CONVERSION_DICT = {
    "temperature_2m": {
        "var": ["temperature_2m"],
        "conv": lambda df: df["temperature_2m"] + 273.15,  # degC -> K
    },
    "dewpoint_temperature_2m": {
        "var": ["dew_point_2m"],
        "conv": lambda df: df["dew_point_2m"] + 273.15,  # degC -> K
    },
    "surface_pressure": {
        "var": ["surface_pressure"],
        "conv": lambda df: df["surface_pressure"] * 100,  # hPa -> Pa
    },
    "total_precipitation": {
        "var": ["precipitation"],
        "conv": lambda df: df["precipitation"] / 1000,  # mm -> m
    },
    # Meteorological wind direction is the direction the wind blows FROM, in degrees
    # clockwise from north, so the vector components are u = -V*sin(theta) (eastward) and
    # v = -V*cos(theta) (northward). Sanity check, covered by tests/test_weather.py: a
    # 10 m/s wind from due north (theta=0) gives u=0, v=-10.
    "u_component_of_wind_10m": {
        "var": ["wind_speed_10m", "wind_direction_10m"],
        "conv": lambda df: -df["wind_speed_10m"]
        * np.sin(np.radians(df["wind_direction_10m"])),
    },
    "v_component_of_wind_10m": {
        "var": ["wind_speed_10m", "wind_direction_10m"],
        "conv": lambda df: -df["wind_speed_10m"]
        * np.cos(np.radians(df["wind_direction_10m"])),
    },
    "u_component_of_wind_100m": {
        "var": ["wind_speed_100m", "wind_direction_100m"],
        "conv": lambda df: -df["wind_speed_100m"]
        * np.sin(np.radians(df["wind_direction_100m"])),
    },
    "v_component_of_wind_100m": {
        "var": ["wind_speed_100m", "wind_direction_100m"],
        "conv": lambda df: -df["wind_speed_100m"]
        * np.cos(np.radians(df["wind_direction_100m"])),
    },
    "instantaneous_10m_wind_gust": {
        "var": ["wind_gusts_10m"],
        "conv": lambda df: df["wind_gusts_10m"],
    },
    "total_cloud_cover": {
        "var": ["cloud_cover"],
        "conv": lambda df: df["cloud_cover"] / 100,  # % -> fraction
    },
    "low_cloud_cover": {
        "var": ["cloud_cover_low"],
        "conv": lambda df: df["cloud_cover_low"] / 100,
    },
    "medium_cloud_cover": {
        "var": ["cloud_cover_mid"],
        "conv": lambda df: df["cloud_cover_mid"] / 100,
    },
    "high_cloud_cover": {
        "var": ["cloud_cover_high"],
        "conv": lambda df: df["cloud_cover_high"] / 100,
    },
    "surface_solar_radiation_downwards": {
        "var": ["shortwave_radiation"],
        # Open-Meteo reports W/m2 averaged over the preceding hour; ERA5 accumulates J/m2
        # over the same hour. W/m2 * 3600 s = J/m2. Verified against the retired GEE export
        # to 0.1% over 8832 hours.
        "conv": lambda df: df["shortwave_radiation"] * 3600,  # W/m2 -> J/m2
    },
    "convective_available_potential_energy": {
        "var": ["cape"],
        "conv": lambda df: df["cape"],
        "archive": False,
    },
}

# How each variable collapses from hourly to daily. Accumulations sum, state variables
# average. This is applied identically to the cache/archive and forecast paths, which is
# what stops the three-way disagreement described in DEVELOPMENT.md section 4.5 from
# recurring.
#
# Gust is treated as a state variable and averaged. A daily *maximum* is arguably the more
# meteorologically meaningful summary; that is a modelling change rather than a bug fix, so
# it is left as a deliberate future choice rather than folded in here.
DAILY_AGGREGATION = {
    "total_precipitation": "sum",
    "surface_solar_radiation_downwards": "sum",
}
DEFAULT_DAILY_AGGREGATION = "mean"

SUN_VARIABLES = ["sun_altitude", "sun_azimuth"]


def _as_list(value):
    """Normalises a single name or a (possibly OmegaConf) sequence into a plain list."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in value]
    raise TypeError(f"Expected a string or a list of strings, got {type(value).__name__}")


def _check_variables(variables, source):
    """Validates requested variables and returns the Open-Meteo variables they need."""
    variables = _as_list(variables)

    unknown = [v for v in variables if v not in CONVERSION_DICT]
    if unknown:
        raise ValueError(
            f"Unknown weather variable(s): {', '.join(unknown)}. "
            f"Known variables: {', '.join(CONVERSION_DICT)}"
        )

    if source in ("cache", "archive"):
        unavailable = [
            v for v in variables if not CONVERSION_DICT[v].get("archive", True)
        ]
        if unavailable:
            raise ValueError(
                f"Variable(s) {', '.join(unavailable)} are not available in the ERA5 "
                f"archive and can therefore only be used on the forecast path. Requesting "
                f"them for training would fill the feature with NaN."
            )

    openmeteo_variables = sorted(
        {om for v in variables for om in CONVERSION_DICT[v]["var"]}
    )
    return variables, openmeteo_variables


def _request_params(openmeteo_variables):
    """Base Open-Meteo request parameters shared by the archive and forecast endpoints.

    Every unit the conversions in :data:`CONVERSION_DICT` assume is stated explicitly, so
    a change to an Open-Meteo default cannot silently rescale a feature. `wind_speed_unit`
    in particular is not optional: the API default is km/h while ERA5 is m/s, which is
    where the historical factor-of-3.6 wind inflation came from.
    """
    return {
        "hourly": openmeteo_variables,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timezone": "GMT",
    }


class _CachedTimeoutSession(requests_cache.CachedSession):
    """A `CachedSession` that applies a default timeout, like `retry_requests.TSession`.

    `retry_requests.retry()` only supplies a timeout when it creates the session itself; a
    session passed in keeps `requests`' default of waiting forever. This makes the timeout
    explicit on the cached path too.
    """

    def __init__(self, *args, timeout, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout

    def request(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return super().request(method, url, *args, **kwargs)


def _client(use_http_cache, timeout):
    """Open-Meteo client with retries, a timeout, and optionally an on-disk HTTP cache.

    The HTTP cache is useful for the forecast path (the daily job and repeated dev runs hit
    the same URL) but counter-productive for bulk archive fetches, where responses are tens
    of megabytes each and are only ever read once.

    `timeout` is worth stating explicitly: `retry_requests` defaults to 5 seconds, which is
    fine for a forecast request and far too short for a multi-year archive request.
    """
    # The session has to be built here rather than left to `retry()`: `retry()` forwards
    # its extra keyword arguments to `urllib3.Retry`, and would otherwise fall back to
    # `TSession()` with its 5 second default.
    session = (
        _CachedTimeoutSession(".cache", expire_after=3600, timeout=timeout)
        if use_http_cache
        else TSession(timeout=timeout)
    )
    return Client(session=retry(session, retries=5, backoff_factor=0.2))


def _responses_to_frame(responses, locations, variables, openmeteo_variables, add_sun):
    """Converts Open-Meteo flatbuffer responses into a tidy long DataFrame."""
    frames = []

    for i, response in enumerate(responses):
        hourly = response.Hourly()

        # The SDK returns variables in the order they were requested.
        raw = pd.DataFrame(
            {
                om_var: hourly.Variables(j).ValuesAsNumpy()
                for j, om_var in enumerate(openmeteo_variables)
            }
        )

        df = pd.DataFrame(
            {
                "datetime": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=f"{hourly.Interval()}s",
                    inclusive="left",
                ),
                "location": locations[i],
            }
        )

        for var in variables:
            df[var] = CONVERSION_DICT[var]["conv"](raw).astype("float32")

        if add_sun:
            df = _add_sun_position(df, locations[i])

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _add_sun_position(df, location):
    """Adds solar altitude and azimuth for `location` at each timestamp in `df`."""
    lat, lon = get_lat_lon(location)
    position = get_position(df["datetime"], lon[0], lat[0])
    df["sun_altitude"] = np.asarray(position["altitude"], dtype="float32")
    df["sun_azimuth"] = np.asarray(position["azimuth"], dtype="float32")
    return df


def _to_hourly_dataset(df):
    """Reshapes a tidy long DataFrame into an (date, time, location) `xarray.Dataset`."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"].dt.date)
    df["time"] = pd.to_timedelta(df["datetime"].dt.time.astype(str))
    df = df.drop(columns="datetime")

    return df.set_index(["date", "time", "location"]).to_xarray()


def fetch_archive(locations, variables, start_date, end_date, add_sun=False):
    """Downloads hourly ERA5 reanalysis from the Open-Meteo archive API.

    Parameters
    ----------
    locations : str or list of str
        Location names, resolved through :data:`LOCATIONS`.
    variables : list of str
        ERA5 variable names, keys of :data:`CONVERSION_DICT`.
    start_date, end_date : str
        Inclusive bounds, ``"YYYY-MM-DD"``.
    add_sun : bool
        Whether to add solar altitude and azimuth.

    Returns
    -------
    xarray.Dataset
        Indexed by ``date``, ``time`` and ``location``.
    """
    locations = _as_list(locations)
    variables, openmeteo_variables = _check_variables(variables, "archive")
    lat, lon = get_lat_lon(locations)

    params = _request_params(openmeteo_variables)
    params.update(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "models": ARCHIVE_MODEL,
        }
    )

    print(
        f"Downloading ERA5 archive ({start_date} to {end_date}) for "
        f"{len(locations)} location(s)..."
    )
    responses = _client(use_http_cache=False, timeout=ARCHIVE_TIMEOUT).weather_api(
        url=ARCHIVE_URL, params=params
    )

    df = _responses_to_frame(
        responses, locations, variables, openmeteo_variables, add_sun
    )
    return _to_hourly_dataset(df)


def fetch_forecast(locations, variables, lag_day=0, forecast_day=0, add_sun=False):
    """Downloads hourly weather from the Open-Meteo forecast API.

    Parameters
    ----------
    locations : str or list of str
        Location names, resolved through :data:`LOCATIONS`.
    variables : list of str
        ERA5 variable names, keys of :data:`CONVERSION_DICT`.
    lag_day : int
        Number of past days to include.
    forecast_day : int
        Number of future days to include beyond today.
    add_sun : bool
        Whether to add solar altitude and azimuth.

    Returns
    -------
    xarray.Dataset
        Indexed by ``date``, ``time`` and ``location``.
    """
    locations = _as_list(locations)
    variables, openmeteo_variables = _check_variables(variables, "forecast")
    lat, lon = get_lat_lon(locations)

    params = _request_params(openmeteo_variables)
    params.update(
        {
            "latitude": lat,
            "longitude": lon,
            "past_days": lag_day,
            "forecast_days": forecast_day + 1,
        }
    )

    print(f"Downloading forecast for {len(locations)} location(s)...")
    responses = _client(use_http_cache=True, timeout=FORECAST_TIMEOUT).weather_api(
        url=FORECAST_URL, params=params
    )

    df = _responses_to_frame(
        responses, locations, variables, openmeteo_variables, add_sun
    )
    return _to_hourly_dataset(df)


def load_cache(cache_dir, locations, variables, years=None, doy=None, add_sun=False):
    """Reads hourly weather from the local Parquet cache.

    Only the requested locations, years, days-of-year and columns are read from disk, and
    the dense ``(date, time, location)`` array is built after that filtering rather than
    before it. The retired CSV reader did the opposite, densifying the full 1966-2025 range
    for every location before narrowing to the migration season.

    Parameters
    ----------
    cache_dir : str
        Cache directory, as written by ``scripts/build_weather_cache.py``.
    locations : str or list of str
        Location names to read.
    variables : list of str
        ERA5 variable names to read.
    years : iterable of int, optional
        Years to keep. All years if omitted.
    doy : tuple of (int, int), optional
        Inclusive day-of-year bounds. All days if omitted.
    add_sun : bool
        Whether to add solar altitude and azimuth.

    Returns
    -------
    xarray.Dataset
        Indexed by ``date``, ``time`` and ``location``.
    """
    locations = _as_list(locations)
    variables, _ = _check_variables(variables, "cache")

    # Hive partitioning on `location` means unrequested locations are never opened.
    filters = [("location", "in", list(locations))]
    if years is not None:
        filters.append(("year", "in", [int(y) for y in years]))
    if doy is not None:
        filters.append(("doy", ">=", int(doy[0])))
        filters.append(("doy", "<=", int(doy[1])))

    print(f"Reading hourly weather cache for {locations}...")
    df = pd.read_parquet(
        cache_dir,
        columns=["datetime", "location"] + list(variables),
        filters=filters,
    )

    # `location` comes back from the partition path as a dictionary/categorical type.
    df["location"] = df["location"].astype(str)

    if df.empty:
        raise ValueError(
            f"The weather cache at {cache_dir} has no rows for locations={locations}, "
            f"years={years}, doy={doy}. Has scripts/build_weather_cache.py been run for "
            f"these locations and years?"
        )

    missing = set(locations) - set(df["location"].unique())
    if missing:
        raise ValueError(
            f"The weather cache at {cache_dir} is missing location(s): "
            f"{', '.join(sorted(missing))}. Re-run scripts/build_weather_cache.py."
        )

    # A partially built cache would otherwise train on fewer years than the config asks for
    # without saying so. The backfill takes several days, so this is a warning rather than an
    # error, but it must be loud: it changes what the model was trained on.
    if years is not None:
        absent = sorted(set(int(y) for y in years) - set(df["datetime"].dt.year.unique()))
        if absent:
            warnings.warn(
                f"The weather cache at {cache_dir} has no data for {len(absent)} of the "
                f"{len(list(years))} requested years "
                f"({absent[0]}-{absent[-1]}); those years are silently excluded from "
                f"training. Run scripts/build_weather_cache.py to extend the cache, or set "
                f"data.years to match what it holds.",
                stacklevel=2,
            )

    if add_sun:
        df = pd.concat(
            [_add_sun_position(g, loc) for loc, g in df.groupby("location", observed=True)],
            ignore_index=True,
        )

    return _to_hourly_dataset(df)


def to_daily(dataset, lag_day):
    """Collapses an hourly dataset to daily values and stacks `lag_day` lags.

    Aggregation is per variable, following :data:`DAILY_AGGREGATION`: accumulations are
    summed over the 24 hours and state variables averaged. Both the training and the
    forecast path call this function, so the two cannot disagree about what "daily" means.

    Parameters
    ----------
    dataset : xarray.Dataset
        Hourly data indexed by ``date``, ``time`` and ``location``.
    lag_day : int
        Number of lags to build. ``lag=0`` is the day itself.

    Returns
    -------
    xarray.Dataset
        Indexed by ``date``, ``location`` and ``lag``, with dates lacking a full set of
        lags dropped so every sample has the same shape.
    """
    daily = xr.Dataset(
        {
            var: (
                dataset[var].sum(dim="time")
                if DAILY_AGGREGATION.get(var, DEFAULT_DAILY_AGGREGATION) == "sum"
                else dataset[var].mean(dim="time")
            )
            for var in dataset.data_vars
        }
    )

    # `shift` moves by position, not by calendar day, so a date axis with gaps in it — the
    # off-season gap left by any day-of-year filtering, or a missing day — would make lag 1
    # reach across the gap instead of to yesterday. Reindexing onto a gapless daily
    # calendar first makes position and calendar day equivalent, and turns the gaps into
    # NaN that the `dropna` below removes. Daily data is small, so this is cheap.
    daily = daily.reindex(
        date=pd.date_range(
            daily.date.min().values, daily.date.max().values, freq="D"
        )
    )

    daily = daily.assign_coords(lag=[0])
    for var in daily.data_vars:
        daily[var] = daily[var].expand_dims({"lag": daily.lag})

    lagged = daily.copy()
    for lag in range(1, lag_day):
        shifted = daily.shift(date=lag).assign_coords(lag=[lag])
        lagged = lagged.merge(shifted.copy())

    # Dropping dates with NaN removes the leading dates that have no full lag window, so
    # every sample has the same shape.
    return lagged.dropna(dim="date")


def get_weather(
    locations,
    variables,
    source="cache",
    resolution="hourly",
    cache_dir=None,
    years=None,
    doy=None,
    start_date=None,
    end_date=None,
    lag_day=0,
    forecast_day=0,
    add_sun=False,
):
    """Single entry point for weather data, for training and for forecasting alike.

    Parameters
    ----------
    locations : str or list of str
        Location names, resolved through :data:`LOCATIONS`.
    variables : list of str
        ERA5 variable names, keys of :data:`CONVERSION_DICT`.
    source : {"cache", "archive", "forecast"}
        Where the data comes from. ``"cache"`` reads the local Parquet store and is what
        training uses; ``"archive"`` hits the Open-Meteo ERA5 archive API and is used to
        build that store; ``"forecast"`` hits the Open-Meteo forecast API and is what the
        daily prediction job uses.
    resolution : {"hourly", "daily"}
        ``"daily"`` aggregates per :data:`DAILY_AGGREGATION` and stacks ``lag_day`` lags.
    cache_dir : str, optional
        Cache directory. Required when ``source="cache"``.
    years : iterable of int, optional
        Years to keep. ``source="cache"`` only.
    doy : tuple of (int, int), optional
        Inclusive day-of-year bounds. ``source="cache"`` only.
    start_date, end_date : str, optional
        Inclusive ``"YYYY-MM-DD"`` bounds. Required when ``source="archive"``.
    lag_day : int
        Past days to fetch (``source="forecast"``) and lags to build
        (``resolution="daily"``).
    forecast_day : int
        Days ahead to fetch. ``source="forecast"`` only.
    add_sun : bool
        Whether to add solar altitude and azimuth. Hourly resolution only.

    Returns
    -------
    xarray.Dataset
        Indexed by ``date``, ``time`` and ``location`` when hourly, or by ``date``,
        ``location`` and ``lag`` when daily.
    """
    if resolution not in ("hourly", "daily"):
        raise ValueError(f"resolution must be 'hourly' or 'daily', got {resolution!r}")

    if resolution == "daily":
        if add_sun:
            raise ValueError("add_sun is only meaningful at hourly resolution.")
        if lag_day < 1:
            raise ValueError(f"resolution='daily' needs lag_day >= 1, got {lag_day}")

    # Building `lag_day` lags needs the `lag_day - 1` days preceding the requested window,
    # so the read is widened and the result narrowed back down afterwards.
    read_doy = doy
    if resolution == "daily" and doy is not None:
        read_doy = (max(1, int(doy[0]) - lag_day), int(doy[1]))

    if source == "cache":
        if cache_dir is None:
            raise ValueError("cache_dir is required when source='cache'.")
        hourly = load_cache(cache_dir, locations, variables, years, read_doy, add_sun)
    elif source == "archive":
        if start_date is None or end_date is None:
            raise ValueError(
                "start_date and end_date are required when source='archive'."
            )
        hourly = fetch_archive(locations, variables, start_date, end_date, add_sun)
    elif source == "forecast":
        hourly = fetch_forecast(locations, variables, lag_day, forecast_day, add_sun)
    else:
        raise ValueError(
            f"source must be 'cache', 'archive' or 'forecast', got {source!r}"
        )

    if resolution == "daily":
        daily = to_daily(hourly, lag_day)
        if doy is not None:
            daily = daily.sel(
                date=(daily.date.dt.dayofyear >= int(doy[0]))
                & (daily.date.dt.dayofyear <= int(doy[1]))
            )
        return daily
    return hourly
