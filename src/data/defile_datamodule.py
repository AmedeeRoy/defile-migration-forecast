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
    return tuple([torch.FloatTensor(s) for s in transformed_sample])


class DefileDataset(Dataset):
    def __init__(
        self,
        count,
        era5_main,
        era5_hourly,
        era5_daily,
        era5_main_trans,
        era5_hourly_trans,
        era5_daily_trans,
        mask,
        return_original=False,
    ):

        # Assign to self
        self.count = count
        self.era5_main = era5_main
        self.era5_daily = era5_daily
        self.era5_hourly = era5_hourly
        self.era5_main_trans = era5_main_trans
        self.era5_hourly_trans = era5_hourly_trans
        self.era5_daily_trans = era5_daily_trans
        self.mask = mask
        self.return_original = return_original

    def __len__(self):
        return len(self.count)

    def __getitem__(self, idx):
        count = self.count.iloc[idx]

        # index by count/observation
        if self.return_original:
            sample = (
                count["count"],
                count["year_used"],
                count["doy"],
                self.era5_main.sel(date=count["date"]),
                self.era5_hourly.sel(date=count["date"]),
                self.era5_daily.sel(date=count["date"]),
                self.mask[:, idx],
            )
        else:
            sample = (
                count["count"],
                count["year_used_trans"],
                count["doy_trans"],
                self.era5_main_trans.sel(date=count["date"]),
                self.era5_hourly_trans.sel(date=count["date"]),
                self.era5_daily_trans.sel(date=count["date"]),
                self.mask[:, idx],
            )
            sample = sample2tensor(sample)
        return sample

    def set_return_original(self, value):
        self.return_original = value


class ForecastDataset(Dataset):
    def __init__(
        self,
        era5_main_location,
        era5_main_variables,
        era5_hourly_locations,
        era5_hourly_variables,
        era5_daily_locations,
        era5_daily_variables,
        lag_day,
        forecast_day,
        transform_data,
        return_original=False,
        year_used="none",
    ):
        # Assert that if transform=True
        if transform_data is None:
            raise ValueError(f"transform_data need to be provided if transform is True")

        self.year_used = year_used
        self.lag_day = lag_day
        self.forecast_day = forecast_day
        self.transform_data = transform_data
        self.return_original = return_original
        self.year_used = year_used

        # MAIN ERA-5 DATA ----------------------------
        self.era5_main = download_forecast_hourly(
            locations=era5_main_location,
            variables=era5_main_variables,
            lag_day=0,
            forecast_day=forecast_day,
            add_sun=True,
        )

        # HOURLY ERA-5 DATA ----------------------------
        self.era5_hourly = download_forecast_hourly(
            locations=era5_hourly_locations,
            variables=era5_hourly_variables,
            lag_day=0,
            forecast_day=forecast_day,
            add_sun=False,
        )

        # DAILY ERA-5 DATA ----------------------------
        self.era5_daily = download_forecast_daily(
            locations=era5_daily_locations,
            variables=era5_daily_variables,
            lag_day=lag_day,
            forecast_day=forecast_day,
        )

        # Create a data.frame with the initial values
        count = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=pd.Timestamp.now().normalize(),
                    periods=forecast_day + 1,
                    freq="D",
                )
            }
        )

        # fake date
        if False:
            dt = (
                pd.Timestamp.now().normalize()
                - pd.to_datetime("2024-09-01").normalize()
            )
            count["date"] = count["date"] + dt
            self.era5_main["date"] = self.era5_main["date"] + dt
            self.era5_hourly["date"] = self.era5_hourly["date"] + dt
            self.era5_daily["date"] = self.era5_daily["date"] + dt

        count["doy"] = count["date"].dt.day_of_year
        count["year"] = count["date"].dt.year
        # This value are determine to become 0,1,2 when transformed.
        count["year_period"] = np.where(
            count["year"] < 1993, 2000, np.where(count["year"] <= 2013, 2100, 2200)
        )
        count["year_used"] = np.where(
            self.year_used == "constant",
            2000,
            np.where(self.year_used == "period", count["year_period"], count["year"]),
        )
        count["year_used_trans"] = (count["year_used"] - 2000) / 100
        count["doy_trans"] = (count["doy"] - 183) / 366

        self.count = count

        # Apply transformation
        self.era5_main_trans = self.transform_data["main"].apply_transformers(
            self.era5_main
        )
        self.era5_hourly_trans = self.transform_data["hourly"].apply_transformers(
            self.era5_hourly
        )
        self.era5_daily_trans = self.transform_data["daily"].apply_transformers(
            self.era5_daily
        )

    def __len__(self):
        return len(self.count)

    def __getitem__(self, idx):
        date = self.count["date"][idx]

        # index by count/observation
        if self.return_original:
            return (
                self.count["year_used"][idx],
                self.count["doy"][idx],
                self.era5_main.sel(date=date),
                self.era5_hourly.sel(date=date),
                self.era5_daily.sel(date=date),
            )
        else:
            sample = (
                self.count["year_used_trans"][idx],
                self.count["doy_trans"][idx],
                self.era5_main_trans.sel(date=date),
                self.era5_hourly_trans.sel(date=date),
                self.era5_daily_trans.sel(date=date),
            )
            return sample2tensor([np.zeros(1)] + list(sample) + [np.zeros(1)])

    def set_return_original(self, value):
        self.return_original = value


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
        train_val_test: str = "period",
        year_used: str = "none",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DefileDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param species: The species for which to model the count. Defaults to `"Common Buzzard"`.
        :param years: List of the year considered in the model
        :param doy: Range (min and max) of the day of year considered in the model
        :param lag_day: The number of lag day to consider in the model. Defaults to `7`.
        :param forecast_day: Number of day ahead used for prediction
        :param train_val_test_cum_ratio: The train, validation and test split defined as the cumulative ratio of the total dataset. Defaults to `(0.7, 0.9)`.
        :param train_val_test: The type of train, validation and test split. Defaults to `"period"`.
        :param year_used: The type of year variable used in the model. Defaults to `"none"`.  "constant" for no information of year included in the model, "none" for the exact year or "period" where only a broad category of year period is included
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
        self.train_val_test = train_val_test
        self.train_val_test_cum_ratio = np.array(train_val_test_cum_ratio)
        self.year_used = year_used
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

        # Load the data
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

        # Filter data by years and day of year
        def filter_by_date(xr):
            return xr.sel(
                date=(
                    (xr.date.dt.year.isin(years))
                    & (xr.date.dt.dayofyear >= doy[0])
                    & (xr.date.dt.dayofyear <= doy[1])
                )
            )

        self.era5_main = filter_by_date(era5_main)
        self.era5_hourly = filter_by_date(era5_hourly)
        self.era5_daily = filter_by_date(era5_daily)

        # COUNT DATA ----------------------------
        # Read data
        all_count = pd.read_csv(
            data_dir + "/count/all_count_processed.csv",
            parse_dates=["date", "start", "end"],
        )

        # Filter data by years and day of year
        all_count = all_count[
            all_count["date"].dt.year.isin(years)
            & all_count["date"].dt.day_of_year.between(*doy)
        ]

        # Filter data by species and sum count of all species happening during the same period
        count_sp = (
            all_count[all_count.species == species][["date", "count", "start", "end"]]
            .groupby(["date", "start", "end"], as_index=False)["count"]
            .sum()
        )
        if len(count_sp) == 0:
            raise ValueError(f"No data for species {species} in the selected years.")

        # Build dataframe with the zero count
        # Extract the dataframe with all period (regardless of the species)
        all_count_sp = all_count[
            [x for x in list(all_count) if x not in ["species", "count"]]
        ].drop_duplicates()

        count = pd.merge(all_count_sp, count_sp, how="left")
        # Replace NA (no match in data_count) with 0
        count["count"] = count["count"].fillna(0)

        # counvert to hourly count
        count["duration"] = (count["end"] - count["start"]).dt.total_seconds() / 3600
        count["count_raw"] = count["count"]
        count["count"] = count["count"] / count["duration"]

        # Add pre-cumputed variable
        count["doy"] = count["date"].dt.day_of_year
        count["year"] = count["date"].dt.year
        # This value are determine to become 0,1,2 when transformed.
        count["year_period"] = np.where(
            count["year"] < 1993, 2000, np.where(count["year"] <= 2013, 2100, 2200)
        )
        count["year_used"] = np.where(
            self.year_used == "constant",
            2000,
            np.where(self.year_used == "period", count["year_period"], count["year"]),
        )

        # Create mask
        # Corresponding to the fraction of each hour of the day during which the count in question has been happening
        hours_mat = np.repeat(np.arange(24), len(count)).reshape(24, len(count))
        startHour = count["start"].dt.hour.values + count["start"].dt.minute.values / 60
        endHour = count["end"].dt.hour.values + count["end"].dt.minute.values / 60
        tmp1 = np.maximum(np.minimum(hours_mat - startHour + 1, 1), 0)
        tmp2 = np.maximum(np.minimum(endHour - hours_mat, 1), 0)
        self.mask = np.minimum(tmp1, tmp2)
        # mask.sum(axis=0) # Check mask is never 0

        # Splitting Training, Validation and Test
        # OPTION 1: split dataset years based on type of data collected
        if train_val_test == "period":
            # Get unique years for each period
            yr_grp = {
                period: np.unique(count.loc[count["year_period"] == period, "year"])
                for period in [2000, 2100, 2200]
            }

            # Shuffle order of the year in each group
            # np.random.seed(self.seed)
            [np.random.shuffle(y) for y in yr_grp.values()]

            # Assign years to each group according to the cumulative ratio defined
            for period, y in yr_grp.items():
                sz = (len(y) * self.train_val_test_cum_ratio).astype(int)
                y_train, y_val, y_test = np.split(y, sz)
                count.loc[count["year"].isin(y_train), "tvt"] = "train"
                count.loc[count["year"].isin(y_val), "tvt"] = "val"
                count.loc[count["year"].isin(y_test), "tvt"] = "test"

            log.info(
                f"Train dataset: selected years - {sorted(count.loc[count['tvt'] == 'train', 'year'].unique())}"
            )
            log.info(
                f"Validation dataset: selected years - {sorted(count.loc[count['tvt'] == 'val', 'year'].unique())}"
            )
            log.info(
                f"Test dataset: selected years - {sorted(count.loc[count['tvt'] == 'test', 'year'].unique())}"
            )
        else:
            # Assign each row in count["tvt"] randomly based on proportions
            count["tvt"] = np.random.choice(
                ["train", "val", "test"],
                size=len(count),
                p=np.diff(np.concatenate(([0], train_val_test_cum_ratio, [1]))),
            )

        # Compute transformation

        # Create a DataTransformers for each era5 data. This class does not store the data, only the transformation and the parameters of the transformation
        self.transform_data = {
            # "count": lambda x: x,  # np.log1p(x),  # np.sqrt(x) / 10,
            # "count_rev": lambda x: x,  # np.expm1(x),  # (10 * x) ** 2,
            "year_used": lambda x: (x - 2000) / 100,
            "doy": lambda x: (x - 183) / 366,
            "main": DataTransformer(dataset=self.era5_main),
            "hourly": DataTransformer(dataset=self.era5_hourly),
            "daily": DataTransformer(dataset=self.era5_daily),
        }

        # Transformation of the count
        # count["count_trans"] = self.transform_data["count"](count["count"])
        count["year_used_trans"] = self.transform_data["year_used"](count["year_used"])
        count["doy_trans"] = self.transform_data["doy"](count["doy"])
        self.count = count

        # Apply transformation
        self.era5_main_trans = self.transform_data["main"].apply_transformers(
            self.era5_main
        )
        self.era5_hourly_trans = self.transform_data["hourly"].apply_transformers(
            self.era5_hourly
        )
        self.era5_daily_trans = self.transform_data["daily"].apply_transformers(
            self.era5_daily
        )

        idx = self.count["tvt"] == "train"
        count = self.count[idx]
        mask = self.mask[:, idx]

        self.data_train = DefileDataset(
            count=count,
            era5_main=self.era5_main.sel(
                date=np.isin(self.era5_main["date"], count["date"])
            ),
            era5_hourly=self.era5_hourly.sel(
                date=np.isin(self.era5_hourly["date"], count["date"])
            ),
            era5_daily=self.era5_daily.sel(
                date=np.isin(self.era5_daily["date"], count["date"])
            ),
            era5_main_trans=self.era5_main_trans.sel(
                date=np.isin(self.era5_main_trans["date"], count["date"])
            ),
            era5_hourly_trans=self.era5_hourly_trans.sel(
                date=np.isin(self.era5_hourly_trans["date"], count["date"])
            ),
            era5_daily_trans=self.era5_daily_trans.sel(
                date=np.isin(self.era5_daily_trans["date"], count["date"])
            ),
            mask=mask,
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
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        idx = self.count["tvt"] == "val"
        count = self.count[idx]
        mask = self.mask[:, idx]

        self.data_val = DefileDataset(
            count=count,
            era5_main=self.era5_main.sel(
                date=np.isin(self.era5_main["date"], count["date"])
            ),
            era5_hourly=self.era5_hourly.sel(
                date=np.isin(self.era5_hourly["date"], count["date"])
            ),
            era5_daily=self.era5_daily.sel(
                date=np.isin(self.era5_daily["date"], count["date"])
            ),
            era5_main_trans=self.era5_main_trans.sel(
                date=np.isin(self.era5_main_trans["date"], count["date"])
            ),
            era5_hourly_trans=self.era5_hourly_trans.sel(
                date=np.isin(self.era5_hourly_trans["date"], count["date"])
            ),
            era5_daily_trans=self.era5_daily_trans.sel(
                date=np.isin(self.era5_daily_trans["date"], count["date"])
            ),
            mask=mask,
        )

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        idx = self.count["tvt"] == "test"
        count = self.count[idx]
        mask = self.mask[:, idx]

        self.data_test = DefileDataset(
            count=count,
            era5_main=self.era5_main.sel(
                date=np.isin(self.era5_main["date"], count["date"])
            ),
            era5_hourly=self.era5_hourly.sel(
                date=np.isin(self.era5_hourly["date"], count["date"])
            ),
            era5_daily=self.era5_daily.sel(
                date=np.isin(self.era5_daily["date"], count["date"])
            ),
            era5_main_trans=self.era5_main_trans.sel(
                date=np.isin(self.era5_main_trans["date"], count["date"])
            ),
            era5_hourly_trans=self.era5_hourly_trans.sel(
                date=np.isin(self.era5_hourly_trans["date"], count["date"])
            ),
            era5_daily_trans=self.era5_daily_trans.sel(
                date=np.isin(self.era5_daily_trans["date"], count["date"])
            ),
            mask=mask,
        )

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
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
            transform_data=self.transform_data,
            return_original=self.data_train.return_original,
            year_used=self.year_used,
        )
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = DefileDataModule()
