import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.data_transformer import DataTransformer
from src.data.get_era5 import *  # All function to load the csv era5 data
from src.data.open_meteo import *  # All function related to the forecast
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def sample2tensor(sample):
    transformed_sample = []
    for s in sample:
        if isinstance(s, np.ndarray):
            transformed_sample.append(s)  # Keep np.array unchanged
        elif hasattr(s, "to_array"):
            transformed_sample.append(s.to_array().values)  # xarray to numpy
        else:
            transformed_sample.append(np.array([s]))

    # to tensor
    sample = tuple([torch.FloatTensor(s) for s in transformed_sample])

    return sample


class DefileDataset(Dataset):
    def __init__(
        self,
        data_dir,
        species="Common Buzzard",
        era5_main_location="Defile",
        era5_main_variables: list = [
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        era5_hourly_locations: list = [
            "MontTendre",
            "Chasseral",
            "Basel",
            "Dijon",
            "ColGrandSaintBernard",
        ],
        era5_hourly_variables: list = [
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        era5_daily_locations: list = [
            "Defile",
            "Schaffhausen",
            "Basel",
            "Munich",
            "Stuttgart",
            "Frankfurt",
            "Berlin",
        ],
        era5_daily_variables: list = [
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        years=range(1966, 2024),
        doy=[196, 335],
        lag_day=7,
        transform=False,
        transform_data=None,
    ):
        # MAIN ERA-5 DATA ----------------------------
        era5_main = get_era5_hourly(
            data_dir,
            locations=era5_main_location,
            variables=era5_main_variables,
            add_sun=True,
        )

        # HOURLY ERA-5 DATA ----------------------------
        era5_hourly = get_era5_hourly(
            data_dir,
            locations=era5_hourly_locations,
            variables=era5_hourly_variables,
            add_sun=False,
        )

        # DAILY ERA-5 DATA ----------------------------
        era5_daily = get_era5_daily(
            data_dir,
            locations=era5_daily_locations,
            variables=era5_daily_variables,
            lag_day=lag_day,
        )

        # --------
        ## Why not filtering era5 by year too?
        # -------

        # COUNT DATA ----------------------------
        # Read data
        df = pd.read_csv(
            data_dir + "/count/all_count_processed.csv",
            parse_dates=["date", "start", "end"],
        )

        # Filter data by years
        df["doy"] = df["date"].dt.day_of_year
        df["year"] = df["date"].dt.year
        dfy = df[
            df["date"].dt.year.isin(years)
            & (df["doy"] >= doy[0])
            & (df["doy"] <= doy[1])
        ]

        # Filter data by species and sum count of all species happening during the same period
        data_count = (
            dfy[dfy.species == species][["date", "count", "start", "end"]]
            .groupby(["date", "start", "end"], as_index=False)["count"]
            .sum()
        )
        if len(data_count) == 0:
            raise ValueError(f"No data for species {species} in the selected years.")

        # Build data.frame with the zero count
        # Extract the dataframe with all period (regardless of the species)
        df_all_period = dfy[
            [x for x in list(dfy) if x not in ["species", "count"]]
        ].drop_duplicates()

        dfys = pd.merge(df_all_period, data_count, how="left")
        # Replace NA (no match in data_count) with 0
        dfys["count"] = dfys["count"].fillna(0)

        # Add pre-cumputed variable
        dfys["duration"] = dfys["end"] - dfys["start"]

        # Create mask
        # Corresponding to the fraction of each hour of the day during which the count in question has been happening
        hours_mat = np.repeat(np.arange(24), len(dfys)).reshape(24, len(dfys))
        startHour = dfys["start"].dt.hour.values + dfys["start"].dt.minute.values / 60
        endHour = dfys["end"].dt.hour.values + dfys["end"].dt.minute.values / 60
        tmp1 = np.maximum(np.minimum(hours_mat - startHour + 1, 1), 0)
        tmp2 = np.maximum(np.minimum(endHour - hours_mat, 1), 0)
        mask = np.minimum(tmp1, tmp2)

        # Check mask is never 0
        # mask.sum(axis=0)

        # normalizing
        # Create a DataTransformers for each era5 data. this class does not store the data, only the transformation and the parameters of the transformation
        if transform_data is None:
            transform_data = {}
            transform_data["main"] = DataTransformer(dataset=era5_main)
            transform_data["hourly"] = DataTransformer(dataset=era5_hourly)
            transform_data["daily"] = DataTransformer(dataset=era5_daily)

        if transform:
            # Apply the transformer to each variable
            era5_main = transform_data["main"].apply_transformers(era5_main)
            era5_hourly = transform_data["hourly"].apply_transformers(era5_hourly)
            era5_daily = transform_data["daily"].apply_transformers(era5_daily)
            # dfys["count"] = np.log10(dfys["count"]+1)
            dfys["count"] = np.sqrt(dfys["count"]) / 10
            dfys["doy"] = (dfys["doy"] - 183) / 366
            dfys["year"] = (dfys["year"] - 2000) / 100

        # Assign to self
        self.data = dfys.reset_index(drop=True)
        self.era5_main = era5_main
        self.era5_daily = era5_daily
        self.era5_hourly = era5_hourly
        self.mask = mask
        self.lag_day = lag_day
        self.transform = transform
        self.transform_data = transform_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # index by count/observation
        sample = (
            self.data["count"][idx],
            self.data["year"][idx],
            self.data["doy"][idx],
            self.era5_main.sel(date=self.data["date"][idx]),
            self.era5_hourly.sel(date=self.data["date"][idx]),
            self.era5_daily.sel(date=self.data["date"][idx]),
            self.mask[:, idx],
        )

        # Transform to tensor
        sample = sample2tensor(sample)

        return sample

    def set_transform(self, value):
        self.transform = value


class ForecastDataset(Dataset):
    def __init__(
        self,
        era5_main_location="Defile",
        era5_main_variables=[
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        era5_hourly_locations=[
            "MontTendre",
            "Chasseral",
            "Basel",
            "Dijon",
            "ColGrandSaintBernard",
        ],
        era5_hourly_variables=[
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        era5_daily_locations=[
            "Defile",
            "Schaffhausen",
            "Basel",
            "Munich",
            "Stuttgart",
            "Frankfurt",
            "Berlin",
        ],
        era5_daily_variables=[
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        lag_day=7,
        forecast_day=5,
        transform=False,
        transform_data=None,
    ):
        # Assert that if transform=True
        if transform == True:
            if transform_data is None:
                raise ValueError(
                    f"transform_data need to be provided if transform is True"
                )

        # MAIN ERA-5 DATA ----------------------------
        era5_main = download_forecast_hourly(
            locations=era5_main_location,
            variables=era5_main_variables,
            lag_day=0,
            forecast_day=forecast_day,
            add_sun=True,
        )

        # HOURLY ERA-5 DATA ----------------------------
        era5_hourly = download_forecast_hourly(
            locations=era5_hourly_locations,
            variables=era5_hourly_variables,
            lag_day=0,
            forecast_day=forecast_day,
            add_sun=False,
        )

        # DAILY ERA-5 DATA ----------------------------
        era5_daily = download_forecast_daily(
            locations=era5_daily_locations,
            variables=era5_daily_variables,
            lag_day=lag_day,
            forecast_day=forecast_day,
        )

        # Create a data.frame with the initial values
        df = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=pd.Timestamp.now().normalize(),
                    # start = pd.to_datetime("2017-09-01").normalize(),
                    periods=forecast_day + 1,
                    freq="D",
                )
            }
        )
        df["doy"] = df["date"].dt.day_of_year
        df["year"] = df["date"].dt.year

        if transform:
            # Apply the transformer to each variable
            era5_main = transform_data["main"].apply_transformers(era5_main)
            era5_hourly = transform_data["hourly"].apply_transformers(era5_hourly)
            era5_daily = transform_data["daily"].apply_transformers(era5_daily)
            df["doy"] = (df["doy"] - 183) / 366
            df["year"] = (df["year"] - 2000) / 100

        # Assign to self
        self.data = df.reset_index(drop=True)
        self.era5_main = era5_main
        self.era5_daily = era5_daily
        self.era5_hourly = era5_hourly
        self.lag_day = lag_day
        self.transform = transform
        self.transform_data = transform_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # index by count/observation
        sample = (
            self.data["year"][idx],
            self.data["doy"][idx],
            self.era5_main.sel(date=self.data["date"][idx]),
            self.era5_hourly.sel(date=self.data["date"][idx]),
            self.era5_daily.sel(date=self.data["date"][idx]),
        )

        # Transoform to tensor
        sample = sample2tensor([np.zeros(1)] + list(sample) + [np.zeros(1)])

        return sample

    def set_transform(self, value):
        self.transform = value


class DefileDataModule(LightningDataModule):
    """`LightningDataModule` for the Defile dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        species: str = "Common Buzzard",
        era5_main_location: str = "Defile",
        era5_main_variables: list = [
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        era5_hourly_locations: list = [
            "MontTendre",
            "Chasseral",
            "Basel",
            "Dijon",
            "ColGrandSaintBernard",
        ],
        era5_hourly_variables: list = [
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        era5_daily_locations: list = [
            "Defile",
            "Schaffhausen",
            "Basel",
            "Munich",
            "Stuttgart",
            "Frankfurt",
            "Berlin",
        ],
        era5_daily_variables: list = [
            "temperature_2m",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ],
        years: list = range(1966, 2024),
        doy: Tuple[float, float] = (196, 335),
        lag_day: int = 7,
        forecast_day: int = 5,
        train_val_test_cum_ratio: Tuple[float, float] = (0.7, 0.9),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DefileDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param species: The species for which to model the count. Defaults to `"Common Buzzard"`.
        :param lag_day: The number of lag day to consider in the model. Defaults to `7`.
        :param seed: The seed. Defaults to `0`.
        :param train_val_test_cum_ratio: The train, validation and test split defined as the cumulative ratio of the total dataset. Defaults to `(0.7, 0.9)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.data_dir = data_dir
        self.species = species
        self.era5_main_location = era5_main_location
        self.era5_main_variables = era5_main_variables
        self.era5_hourly_locations = era5_hourly_locations
        self.era5_hourly_variables = era5_hourly_variables
        self.era5_daily_locations = era5_daily_locations
        self.era5_daily_variables = era5_daily_variables
        self.years = years
        self.doy = doy
        self.lag_day = lag_day
        self.forecast_day = forecast_day
        self.train_val_test_cum_ratio = np.array(train_val_test_cum_ratio)
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # split dataset years based on type of data collected
        yr_grp = [
            np.array(
                [y for y in self.years if y < 1993]
            ),  # size 26. Incidental monitoring
            np.array([y for y in self.years if 1993 <= y <= 2013]),  # size 20.
            np.array([y for y in years if y > 2013]),
        ]

        # Shuffle order of the year in each group
        # np.random.seed(self.seed)
        [np.random.shuffle(y) for y in yr_grp]

        # Assign years to each group according to the cumulative ratio defined
        self.ytraining, self.yval, self.ytest = [], [], []
        for y in yr_grp:
            sz = (len(y) * self.train_val_test_cum_ratio).astype(int)
            y_data = np.split(y, sz)
            self.ytraining.extend(y_data[0])
            self.yval.extend(y_data[1])
            self.ytest.extend(y_data[2])

        log.info(f"Train dataset : selected years - {self.ytraining}")
        log.info(f"Validation dataset : selected years - {self.yval}")
        log.info(f"Test dataset : selected years -{self.ytest}")

        self.data_train = DefileDataset(
            data_dir=self.data_dir,
            species=self.species,
            era5_main_location=self.era5_main_location,
            era5_main_variables=self.era5_main_variables,
            era5_hourly_locations=self.era5_hourly_locations,
            era5_hourly_variables=self.era5_hourly_variables,
            era5_daily_locations=self.era5_daily_locations,
            era5_daily_variables=self.era5_daily_variables,
            years=self.ytraining,
            doy=self.doy,
            lag_day=self.lag_day,
            transform=True,
            transform_data=None,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        self.data_val = DefileDataset(
            data_dir=self.data_dir,
            species=self.species,
            era5_main_location=self.era5_main_location,
            era5_main_variables=self.era5_main_variables,
            era5_hourly_locations=self.era5_hourly_locations,
            era5_hourly_variables=self.era5_hourly_variables,
            era5_daily_locations=self.era5_daily_locations,
            era5_daily_variables=self.era5_daily_variables,
            years=self.yval,
            lag_day=self.lag_day,
            transform=self.data_train.transform,
            transform_data=self.data_train.transform_data,
        )

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        self.data_test = DefileDataset(
            data_dir=self.data_dir,
            species=self.species,
            era5_main_location=self.era5_main_location,
            era5_main_variables=self.era5_main_variables,
            era5_hourly_locations=self.era5_hourly_locations,
            era5_hourly_variables=self.era5_hourly_variables,
            era5_daily_locations=self.era5_daily_locations,
            era5_daily_variables=self.era5_daily_variables,
            years=self.ytest,
            lag_day=self.lag_day,
            transform=self.data_train.transform,
            transform_data=self.data_train.transform_data,
        )

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The test dataloader.
        """
        self.data_predict = ForecastDataset(
            era5_main_location=self.era5_main_location,
            era5_main_variables=self.era5_main_variables,
            era5_hourly_locations=self.era5_hourly_locations,
            era5_hourly_variables=self.era5_hourly_variables,
            era5_daily_locations=self.era5_daily_locations,
            era5_daily_variables=self.era5_daily_variables,
            lag_day=self.lag_day,
            forecast_day=self.forecast_day,
            transform=self.data_train.transform,
            transform_data=self.data_train.transform_data,
        )
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = DefileDataModule()
