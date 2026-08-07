"""Tests for the centralised weather module.

The offline tests cover the conventions and aggregation rules that were previously wrong in
ways no metric would have surfaced: the wind-direction convention, the unit conversions, and
what "daily" means. The networked tests, marked `network`, check the train/serve contract
against the live API and are the durable answer to "how do we know training and forecast
features stay comparable" (DEVELOPMENT.md 5.2).

    pytest tests/                    # offline only
    pytest tests/ -m network         # hit the API too
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.weather import (
    CONVERSION_DICT,
    DAILY_AGGREGATION,
    DEFAULT_DAILY_AGGREGATION,
    LOCATIONS,
    _check_variables,
    _request_params,
    get_lat_lon,
    get_weather,
    to_daily,
)


def hourly_dataset(dates, values_by_var, location="Defile"):
    """Builds a synthetic hourly dataset shaped like the real thing."""
    times = pd.to_timedelta(np.arange(24), unit="h")
    return xr.Dataset(
        {
            var: (
                ("date", "time", "location"),
                np.full((len(dates), 24, 1), value, dtype="float64"),
            )
            for var, value in values_by_var.items()
        },
        coords={"date": list(dates), "time": times, "location": [location]},
    )


# --------------------------------------------------------------------------------------
# Wind convention
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "direction, expected_u, expected_v",
    [
        # Meteorological direction is where the wind comes FROM, so a northerly blows
        # southward: v negative, u zero.
        (0, 0.0, -10.0),  # from the north
        (90, -10.0, 0.0),  # from the east  -> blows west
        (180, 0.0, 10.0),  # from the south -> blows north
        (270, 10.0, 0.0),  # from the west  -> blows east
    ],
)
def test_wind_components_follow_meteorological_convention(direction, expected_u, expected_v):
    """u = -V*sin(theta), v = -V*cos(theta).

    The retired code used u = V*cos(theta), v = V*sin(theta), which returned the wind vector
    reflected about the north-east axis: components swapped and sign-flipped. Wind direction
    is one of the strongest drivers of raptor passage, so this mattered a great deal and was
    invisible in aggregate metrics.
    """
    df = pd.DataFrame(
        {"wind_speed_10m": [10.0], "wind_direction_10m": [float(direction)]}
    )

    u = CONVERSION_DICT["u_component_of_wind_10m"]["conv"](df).iloc[0]
    v = CONVERSION_DICT["v_component_of_wind_10m"]["conv"](df).iloc[0]

    assert u == pytest.approx(expected_u, abs=1e-9)
    assert v == pytest.approx(expected_v, abs=1e-9)


def test_wind_components_preserve_speed():
    """Whatever the direction, the vector magnitude must equal the reported speed."""
    df = pd.DataFrame(
        {
            "wind_speed_100m": np.full(36, 7.5),
            "wind_direction_100m": np.arange(0, 360, 10, dtype="float64"),
        }
    )

    u = CONVERSION_DICT["u_component_of_wind_100m"]["conv"](df)
    v = CONVERSION_DICT["v_component_of_wind_100m"]["conv"](df)

    np.testing.assert_allclose(np.hypot(u, v), 7.5, rtol=1e-9)


# --------------------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------------------


def test_unit_conversions_reach_era5_units():
    df = pd.DataFrame(
        {
            "temperature_2m": [0.0, 20.0],
            "dew_point_2m": [0.0, 10.0],
            "surface_pressure": [1013.25, 900.0],
            "precipitation": [0.0, 5.0],
            "cloud_cover": [0.0, 100.0],
            "shortwave_radiation": [0.0, 100.0],
        }
    )

    # degC -> K, hPa -> Pa, mm -> m, % -> fraction, W/m2 -> J/m2 over the hour.
    assert list(CONVERSION_DICT["temperature_2m"]["conv"](df)) == [273.15, 293.15]
    assert list(CONVERSION_DICT["dewpoint_temperature_2m"]["conv"](df)) == [273.15, 283.15]
    assert list(CONVERSION_DICT["surface_pressure"]["conv"](df)) == [101325.0, 90000.0]
    assert list(CONVERSION_DICT["total_precipitation"]["conv"](df)) == [0.0, 0.005]
    assert list(CONVERSION_DICT["total_cloud_cover"]["conv"](df)) == [0.0, 1.0]
    assert list(CONVERSION_DICT["surface_solar_radiation_downwards"]["conv"](df)) == [
        0.0,
        360000.0,
    ]


def test_request_pins_every_unit_the_conversions_assume():
    """Open-Meteo's wind default is km/h while ERA5 is m/s.

    Leaving `wind_speed_unit` unset inflated all five wind features by 3.6 at forecast time,
    which after normalisation put them far outside the trained range. Every unit the
    conversions depend on is therefore pinned explicitly rather than left to a default.
    """
    params = _request_params(["temperature_2m", "wind_speed_10m"])

    assert params["wind_speed_unit"] == "ms"
    assert params["temperature_unit"] == "celsius"
    assert params["precipitation_unit"] == "mm"
    assert params["timezone"] == "GMT"


# --------------------------------------------------------------------------------------
# Daily aggregation
# --------------------------------------------------------------------------------------


def test_accumulations_sum_and_state_variables_average():
    """Precipitation and radiation are hourly accumulations; the rest are state variables.

    Averaging an accumulation understates a daily total by a factor of 24. The retired code
    averaged everything, and separately fed daily *sums* for four of the seven daily
    locations in training while computing 24-hour means at forecast time.
    """
    dates = pd.date_range("2024-09-01", "2024-09-10")
    ds = hourly_dataset(
        dates,
        {
            "total_precipitation": 0.001,  # per hour
            "surface_solar_radiation_downwards": 1000.0,  # per hour
            "temperature_2m": 290.0,  # state
        },
    )

    daily = to_daily(ds, lag_day=1)

    assert DAILY_AGGREGATION["total_precipitation"] == "sum"
    assert DAILY_AGGREGATION["surface_solar_radiation_downwards"] == "sum"
    assert DEFAULT_DAILY_AGGREGATION == "mean"

    assert float(daily.total_precipitation.isel(date=0, lag=0)) == pytest.approx(0.024)
    assert float(
        daily.surface_solar_radiation_downwards.isel(date=0, lag=0)
    ) == pytest.approx(24000.0)
    assert float(daily.temperature_2m.isel(date=0, lag=0)) == pytest.approx(290.0)


def test_lags_never_reach_across_a_gap_in_the_date_axis():
    """`xarray.shift` moves by position, not by calendar day.

    Filtering to the migration season leaves a gap between one season's end and the next
    season's start. Shifting positionally across that gap would give the first days of a
    season a "yesterday" from the previous November. Dates without a full lag window must be
    dropped instead.
    """
    dates = list(pd.date_range("2024-09-01", "2024-09-05")) + list(
        pd.date_range("2024-09-20", "2024-09-25")
    )
    # Encode the day of month in the value so lag wiring is directly readable.
    times = pd.to_timedelta(np.arange(24), unit="h")
    values = np.stack([np.full((24, 1), d.day, dtype="float64") for d in dates])
    ds = xr.Dataset(
        {"temperature_2m": (("date", "time", "location"), values)},
        coords={"date": dates, "time": times, "location": ["Defile"]},
    )

    daily = to_daily(ds, lag_day=3)
    kept = [pd.Timestamp(d) for d in daily.date.values]

    # The first two days of each block cannot have two full lags, so they are dropped.
    assert kept == [
        pd.Timestamp("2024-09-03"),
        pd.Timestamp("2024-09-04"),
        pd.Timestamp("2024-09-05"),
        pd.Timestamp("2024-09-22"),
        pd.Timestamp("2024-09-23"),
        pd.Timestamp("2024-09-24"),
        pd.Timestamp("2024-09-25"),
    ]

    for lag in (1, 2):
        lagged = daily.temperature_2m.sel(lag=lag).squeeze("location").values
        same_day = daily.temperature_2m.sel(lag=0).squeeze("location").values
        np.testing.assert_allclose(lagged, same_day - lag)


def test_daily_requires_at_least_one_lag():
    ds = hourly_dataset(pd.date_range("2024-09-01", "2024-09-05"), {"temperature_2m": 1.0})

    with pytest.raises(ValueError, match="lag_day >= 1"):
        get_weather(
            "Defile", ["temperature_2m"], source="cache", resolution="daily", lag_day=0
        )

    # to_daily itself is happy with lag_day=1, which is the no-lag case.
    assert to_daily(ds, lag_day=1).sizes["lag"] == 1


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def test_forecast_only_variables_are_refused_for_training():
    """CAPE exists on the forecast endpoint but the ERA5 archive returns all-null for it.

    Silently accepting it would fill a training feature with NaN.
    """
    with pytest.raises(ValueError, match="not available in the ERA5 archive"):
        _check_variables(["convective_available_potential_energy"], "archive")

    with pytest.raises(ValueError, match="not available in the ERA5 archive"):
        _check_variables(["convective_available_potential_energy"], "cache")

    names, om_names = _check_variables(
        ["convective_available_potential_energy"], "forecast"
    )
    assert om_names == ["cape"]


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown weather variable"):
        _check_variables(["temperature_at_altitude"], "forecast")

    with pytest.raises(KeyError, match="Unknown location"):
        get_lat_lon(["Atlantis"])


def test_bad_source_and_resolution_are_rejected():
    with pytest.raises(ValueError, match="resolution must be"):
        get_weather("Defile", ["temperature_2m"], resolution="weekly")

    with pytest.raises(ValueError, match="source must be"):
        get_weather("Defile", ["temperature_2m"], source="magic")

    with pytest.raises(ValueError, match="cache_dir is required"):
        get_weather("Defile", ["temperature_2m"], source="cache")

    with pytest.raises(ValueError, match="start_date and end_date are required"):
        get_weather("Defile", ["temperature_2m"], source="archive")


def test_every_location_has_coordinates_in_europe():
    lats, lons = get_lat_lon(list(LOCATIONS))

    assert len(lats) == len(LOCATIONS)
    assert all(40 < lat < 60 for lat in lats)
    assert all(-5 < lon < 25 for lon in lons)


# --------------------------------------------------------------------------------------
# Train/serve parity, against the live API
# --------------------------------------------------------------------------------------

PARITY_VARIABLES = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "u_component_of_wind_100m",
    "v_component_of_wind_100m",
    "total_cloud_cover",
    "total_precipitation",
    "instantaneous_10m_wind_gust",
    "surface_solar_radiation_downwards",
]

# What the two paths must agree on, and what they need not.
#
# The archive is ERA5 reanalysis on a 0.25 deg (~25 km) grid; the forecast endpoint serves
# high-resolution NWP. They are different products, so they do not agree hour by hour, and
# how much they disagree depends on the field and on the terrain (see
# `test_wind_over_complex_terrain_is_documented_as_divergent` below). Asserting a high
# correlation everywhere would therefore be asserting something false.
#
# What must hold regardless is *scale*: the mean offset and the ratio of standard deviations.
# Every unit defect this project has actually had would violate those - km/h vs m/s is 3.6x,
# a daily sum read as an hourly rate is 24x, percent vs fraction is 100x, and celsius vs
# kelvin is a 273 K offset - while none of them would be caught by a loose correlation bound.
#
# `max_bias` is in each variable's own ERA5 units.
PARITY_SCALE_TOLERANCE = {
    "temperature_2m": dict(max_bias=3.0),
    "dewpoint_temperature_2m": dict(max_bias=3.0),
    "surface_pressure": dict(max_bias=500.0),
    "u_component_of_wind_10m": dict(max_bias=2.5),
    "v_component_of_wind_10m": dict(max_bias=2.5),
    "u_component_of_wind_100m": dict(max_bias=2.5),
    "v_component_of_wind_100m": dict(max_bias=2.5),
    "total_cloud_cover": dict(max_bias=0.25),
    "total_precipitation": dict(max_bias=5e-4),
    "instantaneous_10m_wind_gust": dict(max_bias=2.5),
    "surface_solar_radiation_downwards": dict(max_bias=1e5),
}

# Bounds on std(forecast)/std(archive). Wide enough for genuine model and resolution
# differences (measured 0.69-1.73 at Defile, 0.71-2.12 at Frankfurt over 61 days), far
# tighter than any unit error.
MIN_STD_RATIO, MAX_STD_RATIO = 0.3, 3.0

# Synoptic-scale fields that both products must track closely anywhere, because they are set
# by the large-scale flow rather than by local terrain. Measured 0.97-0.99.
SYNOPTIC_MIN_CORR = {
    "temperature_2m": 0.90,
    "surface_pressure": 0.90,
    "surface_solar_radiation_downwards": 0.90,
}

# Wind correlation is asserted over flat terrain only, where the two products do track each
# other (measured 0.88-0.94 with the forecast path pinned to ecmwf_ifs025). This is what
# would catch a flipped or swapped wind convention: it would drive these correlations to
# roughly zero or negative.
FLAT_REFERENCE_LOCATION = "Frankfurt"
FLAT_WIND_MIN_CORR = 0.75

PARITY_PAST_DAYS = 60


def compare_paths(location, variables):
    """Builds the same variables for the same days through both paths.

    Returns a dict of per-variable (bias, std_ratio, corr).
    """
    forecast = get_weather(
        location, variables, source="forecast", lag_day=PARITY_PAST_DAYS, forecast_day=0
    )

    dates = pd.to_datetime(forecast.date.values)
    archive = get_weather(
        location,
        variables,
        source="archive",
        start_date=dates.min().strftime("%Y-%m-%d"),
        end_date=dates.max().strftime("%Y-%m-%d"),
    )

    shared = np.intersect1d(forecast.date.values, archive.date.values)
    assert len(shared) >= 20, (
        f"only {len(shared)} overlapping day(s) between the forecast and archive paths at "
        f"{location}; the archive lags real time by about five days, so past_days must "
        f"comfortably exceed that for the comparison to be stable"
    )
    forecast = forecast.sel(date=shared)
    archive = archive.sel(date=shared)

    stats = {}
    for var in variables:
        a = archive[var].values.ravel()
        f = forecast[var].values.ravel()
        keep = ~(np.isnan(a) | np.isnan(f))
        a, f = a[keep], f[keep]
        stats[var] = dict(
            bias=float(np.mean(f - a)),
            std_ratio=float(f.std() / a.std()) if a.std() > 0 else float("nan"),
            corr=(
                float(np.corrcoef(a, f)[0, 1]) if a.std() > 0 and f.std() > 0 else 1.0
            ),
        )
    return stats


@pytest.mark.network
def test_archive_and_forecast_paths_agree_on_scale():
    """The train/serve parity check.

    Builds every feature for the same recent days through both paths and compares mean offset
    and spread. A renamed variable, a changed Open-Meteo default or a unit slip shows up here
    immediately; this is what would have caught the historical wind-unit bug (km/h read as
    m/s, a factor of 3.6) and the historical daily-aggregation bug (daily sums compared
    against 24-hour means, a factor of 24).
    """
    stats = compare_paths("Defile", PARITY_VARIABLES)

    failures = []
    for var, s in stats.items():
        limit = PARITY_SCALE_TOLERANCE[var]["max_bias"]
        if abs(s["bias"]) > limit:
            failures.append(f"{var}: bias={s['bias']:+.4g} exceeds {limit:g}")
        if not MIN_STD_RATIO <= s["std_ratio"] <= MAX_STD_RATIO:
            failures.append(
                f"{var}: std ratio={s['std_ratio']:.3f} outside "
                f"[{MIN_STD_RATIO}, {MAX_STD_RATIO}]"
            )

    assert not failures, "train/serve scale parity drifted:\n  " + "\n  ".join(failures)


@pytest.mark.network
def test_synoptic_fields_track_each_other():
    """Temperature, pressure and radiation are set by the large-scale flow.

    Both products must follow them closely everywhere. If one of these correlations collapses,
    a variable has been mismapped somewhere in `CONVERSION_DICT`.
    """
    stats = compare_paths("Defile", list(SYNOPTIC_MIN_CORR))

    failures = [
        f"{var}: corr={stats[var]['corr']:.4f} below {limit}"
        for var, limit in SYNOPTIC_MIN_CORR.items()
        if stats[var]["corr"] < limit
    ]
    assert not failures, "synoptic agreement lost:\n  " + "\n  ".join(failures)


@pytest.mark.network
def test_wind_convention_holds_against_the_live_api_over_flat_terrain():
    """Guards the wind convention end to end, at a site where terrain does not confound it.

    The offline test pins the formulas; this one checks that the components the two live
    endpoints produce actually agree, which they only can if both are being decoded with the
    same convention. A swapped or sign-flipped conversion on one path would send these
    correlations to about zero.
    """
    wind = [
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "u_component_of_wind_100m",
        "v_component_of_wind_100m",
    ]
    stats = compare_paths(FLAT_REFERENCE_LOCATION, wind)

    failures = [
        f"{var}: corr={stats[var]['corr']:.4f} below {FLAT_WIND_MIN_CORR}"
        for var in wind
        if stats[var]["corr"] < FLAT_WIND_MIN_CORR
    ]
    assert not failures, (
        f"wind components disagree between paths at {FLAT_REFERENCE_LOCATION}, where "
        f"terrain is not a plausible explanation:\n  " + "\n  ".join(failures)
    )


@pytest.mark.network
def test_wind_over_complex_terrain_is_documented_as_divergent():
    """Records the resolution gap at Defile that persists even after resolution-matching.

    The forecast path is pinned to ecmwf_ifs025 (0.25 deg, matching the ERA5 archive's grid),
    which closed most of the original gap: before the pin, 10 m wind correlated only about
    0.27 at Defile against 0.90 at flat Frankfurt (DWD ICON-D2's 2 km grid resolving the gorge
    that ERA5's 25 km cell cannot). After the pin, both paths use the same 0.25 deg product,
    and correlation at Defile rises to about 0.55 -- but Frankfurt rises further, to about
    0.93. So terrain still drives a real, if smaller, disagreement between the two paths even
    at matched resolution: ERA5 is a reanalysis that assimilates observations and IFS-025 is a
    raw forecast, and that distinction still bites harder in complex terrain than over flat
    ground. Wind direction at a bottleneck is one of the strongest drivers of raptor passage,
    so this residual gap is still worth tracking. See DECISIONS.md -> Weather.

    This test asserts only the direction of the effect, so it fails if the gap ever closes -
    which would mean the products, or this understanding of them, have changed.
    """
    var = "u_component_of_wind_10m"
    defile = compare_paths("Defile", [var])[var]
    flat = compare_paths(FLAT_REFERENCE_LOCATION, [var])[var]

    assert flat["corr"] > defile["corr"], (
        f"expected 10 m wind to agree better over flat terrain than in the gorge, but got "
        f"{FLAT_REFERENCE_LOCATION}={flat['corr']:.3f} vs Defile={defile['corr']:.3f}. "
        f"If this now passes trivially, re-measure and update DECISIONS.md -> Weather."
    )


@pytest.mark.network
def test_archive_returns_the_pinned_era5_model_back_to_1966():
    """The archive must reach the start of the count data on a single, unswitched dataset."""
    dataset = get_weather(
        "Defile",
        ["temperature_2m", "total_precipitation"],
        source="archive",
        start_date="1966-07-15",
        end_date="1966-07-20",
    )

    assert dataset.sizes["date"] == 6
    assert dataset.sizes["time"] == 24
    assert int(dataset.temperature_2m.isnull().sum()) == 0
    # Plausible July surface temperatures, in kelvin.
    assert 270 < float(dataset.temperature_2m.mean()) < 310
