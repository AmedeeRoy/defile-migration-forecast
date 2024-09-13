import numpy as np
import pandas as pd
from openmeteo_requests import Client
from openmeteo_sdk.Variable import Variable
from suncalc import get_position

from src.data.get_era5 import get_lat_lon


def convert_era5_variable(variables):
    """Convert era5 hourly variable names to forecast variable names based on a predefined mapping.

    Parameters:
    variables (list of str): A list of variable names to be converted.

    Returns:
    list of str: A list of converted variable names, with duplicates removed.
    """

    # Dictionary defining the conversion mapping
    # https://www.ecmwf.int/en/forecasts/datasets/open-data

    conversion_dict = {
        "temperature_2m": {
            "var_forecast": ["temperature_2m"],
            "conv": lambda df: df["temperature_2m"],
        },
        "surface_pressure": {
            "var_forecast": ["surface_pressure"],
            "conv": lambda df: df["surface_pressure"],
        },
        "total_precipitation": {
            "var_forecast": ["precipitation"],
            "conv": lambda df: df["precipitation"],
        },
        "dewpoint_temperature_2m": {
            "var_forecast": ["dew_point_2m"],
            "conv": lambda df: df["dew_point_2m"],
        },
        "u_component_of_wind_10m": {
            "var_forecast": ["wind_speed_10m", "wind_direction_10m"],
            "conv": lambda df: df["wind_speed_10m"] * np.cos(np.radians(df["wind_direction_10m"])),
        },
        "v_component_of_wind_10m": {
            "var_forecast": ["wind_speed_10m", "wind_direction_10m"],
            "conv": lambda df: df["wind_speed_10m"] * np.sin(np.radians(df["wind_direction_10m"])),
        },
        "u_component_of_wind_100m": {
            "var_forecast": ["wind_speed_100m", "wind_direction_100m"],
            "conv": lambda df: df["wind_speed_100m"]
            * np.cos(np.radians(df["wind_direction_100m"])),
        },
        "v_component_of_wind_100m": {
            "var_forecast": ["wind_speed_100m", "wind_direction_100m"],
            "conv": lambda df: df["wind_speed_100m"]
            * np.sin(np.radians(df["wind_direction_100m"])),
        },
        "high_cloud_cover": {
            "var_forecast": ["cloud_cover"],
            "conv": lambda df: df["cloud_cover"],
        },
        "low_cloud_cover": {
            "var_forecast": ["cloud_cover_low"],
            "conv": lambda df: df["cloud_cover_low"],
        },
        "medium_cloud_cover": {
            "var_forecast": ["cloud_cover_mid"],
            "conv": lambda df: df["cloud_cover_mid"],
        },
        "total_cloud_cover": {
            "var_forecast": ["cloud_cover_high"],
            "conv": lambda df: df["cloud_cover_high"],
        },
        "instantaneous_10m_wind_gust": {
            "var_forecast": ["wind_gusts_10m"],
            "conv": lambda df: df["wind_gusts_10m"],
        },
        "surface_solar_radiation_downwards": {
            "var_forecast": ["shortwave_radiation"],
            "conv": lambda df: df["shortwave_radiation"],
        },
        "convective_available_potential_energy": {
            "var_forecast": ["cape"],
            "conv": lambda df: df["cape"],
        },
    }

    if isinstance(variables, str):
        variables = [variables]

    # Find all unmatched variables
    unmatched_variables = [var for var in variables if var not in conversion_dict]

    # If there are unmatched variables, raise an error with the list of unmatched variables
    if unmatched_variables:
        raise ValueError(
            f"The following variables are not matched in the conversion_dict: {unmatched_variables}"
        )

    # Collect the unique var_forecast lists and conv functions
    var_forecast_list = []
    conv_functions = []

    for var in variables:
        var_forecast_list.extend(conversion_dict[var]["var_forecast"])
        conv_functions.append(conversion_dict[var]["conv"])

    # Return a tuple: (unique var_forecast list, list of conv functions)
    return list(set(var_forecast_list)), conv_functions


def download_forecast_hourly(locations, variables, lag_day, forecast_day, add_sun=False):
    """Retrieves weather forecast data for given locations and variables using the Open-Meteo API.

    Parameters:
    -----------
    locations : list of str
        List of location names for which to retrieve the forecast.
    variables : list of str
        List of weather variables to retrieve (e.g., temperature, humidity).
    lag_day : int
        Number of past days to include in the forecast.
    forecast_day : int, optional
        Number of future days to include in the forecast.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the weather data for all locations and variables with timestamps.
    """
    # Check if locations is a string, and if so, convert it to a list
    if isinstance(locations, str):
        locations = [locations]

    # Get latitude and longitude coordinates for the provided locations
    lat, lon = get_lat_lon(locations)

    # Convert the requested variables to the format required by the API
    conv_var = convert_era5_variable(variables)

    # Initialize the Open-Meteo API client
    om = Client()

    print("Download ERA5...")
    # Make a request to the Open-Meteo API to get weather data
    responses = om.weather_api(
        url="https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": conv_var[0],
            # "models": "ecmwf_ifs025",
            "past_days": lag_day,
            "forecast_days": forecast_day + 1,
        },
    )

    # List to hold DataFrames for each location
    df_list = []

    # Iterate over the API responses
    for i, r in enumerate(responses):
        hourly = r.Hourly()

        df_forcast = pd.DataFrame(
            {var: hourly.Variables(j).ValuesAsNumpy() for j, var in enumerate(conv_var[0])}
        )

        # Create a DataFrame with timestamps and the corresponding location name
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

        # Add variable based on the function defined in the conversion variable table
        for j, fx in enumerate(conv_var[1]):
            df[variables[j]] = fx(df_forcast)

        # Get sun position (altitude, azimuth)
        if add_sun:
            lat, lon = get_lat_lon(locations[i])
            sun_position = get_position(df["datetime"], lon[0], lat[0])
            df["sun_altitude"] = sun_position["altitude"]
            df["sun_azimuth"] = sun_position["azimuth"]

        # Append the DataFrame for this location to the list
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame and return it
    era5 = pd.concat(df_list, ignore_index=True)

    # Rename date to datetime to make it easier to then create `date` and `time`
    era5["date"] = pd.to_datetime(era5["datetime"].dt.date)
    era5["time"] = pd.to_timedelta(era5.datetime.dt.time.astype(str))
    era5 = era5.drop("datetime", axis=1)

    # Create data xarray (better to handle multi-indexing)
    era5 = era5.set_index(["date", "time", "location"]).to_xarray()

    return era5


def download_forecast_daily(locations, variables, lag_day, forecast_day):
    # Read the hourly data
    era5_hourly = download_forecast_hourly(
        locations=locations,
        variables=variables,
        lag_day=lag_day,
        forecast_day=forecast_day,
        add_sun=False,
    )

    # Create daily data (with lags)
    era5_daily = era5_hourly.mean(dim="time")  # get daily mean
    era5_daily = era5_daily.assign_coords(lag=[0])  # add lag as new coordinate
    # Make all existing variables depend on the new coordinate
    for var in era5_daily.data_vars:
        era5_daily[var] = era5_daily[var].expand_dims({"lag": era5_daily.lag})

    # Shift and merge daily data
    era5_daily_lagged = era5_daily.copy()
    for lag in range(1, lag_day):
        df = era5_daily.shift(date=lag)
        df = df.assign_coords(lag=[lag])
        era5_daily_lagged = era5_daily_lagged.merge(df.copy())

    #  Remove all dates with NaN
    # (-> to guarantee that each item has the same size)
    # (= equivalent to removing date when no lags are available)
    era5_daily_lagged = era5_daily_lagged.dropna(dim="date")

    # check that no NaN values are remaining -> ok !
    # print('Remaining NaN in ERA5 daily :', era5_daily_lagged.isnull().sum())

    return era5_daily_lagged
