from suncalc import get_position
import pandas as pd


def get_lat_lon(locations):
    """Returns lists of latitudes and longitudes for a list of location names.

    Args:
        locations (list of str): A list of location names.

    Returns:
        tuple: Two lists - one for latitudes and one for longitudes.
    """
    locations_dict = {
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

    # Check if locations is a string, and if so, convert it to a list
    if isinstance(locations, str):
        locations = [locations]

    # Check if all location names are present in the locations dictionary
    missing_locations = [name for name in locations if name not in locations_dict]

    if missing_locations:
        return f"Error: The following locations are not found: {', '.join(missing_locations)}"

    # Extract latitudes and longitudes using list comprehensions
    latitudes = [locations_dict[name][0] for name in locations]
    longitudes = [locations_dict[name][1] for name in locations]

    return latitudes, longitudes


def get_era5_hourly(data_dir, locations, variables, add_sun=False):
    """
    Retrieves hourly ERA5 weather data for specified locations and variables, with the optional addition of sun position data.

    Parameters:
    ----------
    data_dir : str


    locations : str or list of str
        A single location (string) or a list of location names. Each location should have a corresponding CSV file in the `data_dir/era5/` directory, containing hourly ERA5 weather data.

    variables : list of str
        A list of variables to be extracted from the CSV files. These variables must correspond to columns in the ERA5 data files.

    add_sun : bool, optional
        If True, the function will calculate and add sun position data (altitude and azimuth) for each location and timestamp. Default is False.

    Returns:
    -------
    xarray.Dataset
        An xarray Dataset indexed by date, time, and location, containing the specified ERA5 variables and optionally the sun position data.
    """

    # Check if locations is a string or list of strings
    assert isinstance(
        locations, (str, list)
    ), "Error: 'locations' must be a string or a list of strings."
    if isinstance(locations, list):
        assert all(
            isinstance(loc, str) for loc in locations
        ), "Error: All elements in 'locations' list must be strings."
    # Check if locations is a string, and if so, convert it to a list
    if isinstance(locations, str):
        locations = [locations]

    # Check if variables is a list of strings
    assert isinstance(variables, list), "Error: 'variables' must be a list of strings."
    assert all(
        isinstance(var, str) for var in variables
    ), "Error: All elements in 'variables' list must be strings."

    # Check if add_sun is a boolean
    assert isinstance(add_sun, bool), "Error: 'add_sun' must be a boolean value."

    era5 = None
    for loc in locations:
        # Read the CSV file for the location
        era5_loc = pd.read_csv(f"{data_dir}/era5/{loc}.csv", parse_dates=["datetime"])

        # Check if all specified variables are in the CSV file
        missing_vars = [var for var in variables if var not in era5_loc.columns]
        assert (
            not missing_vars
        ), f"Error: The following variables are not present in the data for '{loc}': {', '.join(missing_vars)}"

        # Subset for the specified variables
        era5_loc = era5_loc[["datetime"] + variables]

        # Get sun position (altitude, azimuth)
        if add_sun:
            lat, lon = get_lat_lon(loc)
            sun_position = get_position(era5_loc["datetime"], lon[0], lat[0])
            era5_loc["sun_altitude"] = sun_position["altitude"]
            era5_loc["sun_azimuth"] = sun_position["azimuth"]

        # Add a location column and concatenate dataframes
        # Option 1: Rename column name to unique name with location name
        # era5_loc.rename(columns=lambda x: f"{loc}_{x}" if x != 'date' else x, inplace=True)
        # Merge/join table based on the date column
        # era5 = era5_loc if era5 is None else pd.merge(era5, era5_loc, on='date')

        # Option 2
        era5_loc["location"] = loc
        era5 = (
            era5_loc if era5 is None else pd.concat([era5, era5_loc], ignore_index=True)
        )

    # Rename date to datetime to make it easier to then create `date` and `time`
    era5["date"] = pd.to_datetime(era5["datetime"].dt.date)
    era5["time"] = pd.to_timedelta(era5.datetime.dt.time.astype(str))
    era5 = era5.drop("datetime", axis=1)

    # Create data xarray (better to handle multi-indexing)
    era5 = era5.set_index(["date", "time", "location"]).to_xarray()

    return era5


def get_era5_daily(data_dir, locations, variables, lag_day):
    # Read the hourly data
    era5_hourly = get_era5_hourly(
        data_dir=data_dir, locations=locations, variables=variables, add_sun=False
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
