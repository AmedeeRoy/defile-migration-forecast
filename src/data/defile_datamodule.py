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


class DefileDataset(Dataset):
    def __init__(
        self,
        data_dir,
        species="Buse variable",
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
        years=range(1966, 2023),
        lag_day=7,
        transform=False,
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
            add_sun=True,
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
        df["duration"] = df["end"] - df["start"]
        df["doy"] = df["date"].dt.day_of_year
        df["year"] = df["date"].dt.year

        # Check that ERA5 values are available for all observations -> ok !
        # Otherwise would need to subset the count dataset
        # print('Number of dates not in ERA5 daily :', len([d for d in df.date.unique() if d not in era5_daily_lagged.date]))

        # Filter data by years
        dfy = df[df["date"].dt.year.isin(years)]

        # Filter data by species
        data_count = dfy[dfy.species == species][["date", "count", "start", "end"]]
        dfys = (
            dfy[[x for x in list(dfy) if x not in ["species", "count"]]]
            .drop_duplicates()
            .merge(data_count, how="left")
        )
        dfys["count"] = dfys["count"].fillna(0)

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
        trans_main = DataTransformer(dataset=era5_main)
        trans_hourly = DataTransformer(dataset=era5_hourly)
        trans_daily = DataTransformer(dataset=era5_daily)

        if transform:
            # Apply the transformer to each variable
            era5_main = trans_main.apply_transformers(era5_main)
            era5_hourly = trans_hourly.apply_transformers(era5_hourly)
            era5_daily = trans_daily.apply_transformers(era5_daily)
            dfys["count"] = np.log10(1 + dfys["count"])
            dfys["doy"] = (dfys["doy"] - 183) / 366
            dfys["year"] = (dfys["year"] - 2000) / 100

        # Assign to self
        self.data = dfys.reset_index(drop=True)
        self.era5_main = era5_main
        self.era5_daily = era5_daily
        self.era5_hourly = era5_hourly
        # Export the transformer as a simple dictionary which can me more easily written to the config.yaml file
        # Note that we can re-create the transoformer super easily with DataTransformer(transformers=trans_main)
        self.trans_main = trans_main
        self.trans_daily = trans_daily
        self.trans_hourly = trans_hourly
        self.mask = mask
        self.lag_day = lag_day
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # index by count/observation
        count = self.data["count"][idx]
        yr = self.data["year"][idx]
        doy = self.data["doy"][idx]
        mask = self.mask[:, idx]

        date = self.data["date"][idx]
        era5_main = self.era5_main.sel(date=date)
        era5_hourly = self.era5_hourly.sel(date=date)
        era5_daily = self.era5_daily.sel(date=date)

        # convert to numpy before transformations
        sample = count, yr, doy, era5_main, era5_hourly, era5_daily, mask

        # apply transformations
        if self.transform:
            # to array
            sample = (
                np.array([count]),
                np.array([yr]),
                np.array([doy]),
                era5_main.to_array().values,
                era5_hourly.to_array().values,
                era5_daily.to_array().values,
                mask,
            )
            # to tensor
            sample = tuple([torch.FloatTensor(s) for s in sample])

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
        trans_main=None,
        trans_daily=None,
        trans_hourly=None,
    ):
        # Assert that if transform=True
        if transform == True:
            if trans_main is None or trans_daily is None or trans_hourly is None:
                raise ValueError(
                    f"trans_main, trans_hourly and trans_daily need to be provided if transform is True"
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
            add_sun=True,
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
                    periods=forecast_day + 1,
                    freq="D",
                )
            }
        )
        df["doy"] = df["date"].dt.day_of_year
        df["year"] = df["date"].dt.year

        if transform:
            # Apply the transformer to each variable
            era5_main = trans_main.apply_transformers(era5_main)
            era5_hourly = trans_hourly.apply_transformers(era5_hourly)
            era5_daily = trans_daily.apply_transformers(era5_daily)
            df["doy"] = (df["doy"] - 183) / 366
            df["year"] = (df["year"] - 2000) / 100

        # Assign to self
        self.data = df.reset_index(drop=True)
        self.era5_main = era5_main
        self.era5_daily = era5_daily
        self.era5_hourly = era5_hourly
        # Export the transformer as a simple dictionary which can me more easily written to the config.yaml file
        # Note that we can re-create the transoformer super easily with DataTransformer(transformers=trans_main)
        self.trans_main = trans_main
        self.trans_daily = trans_daily
        self.trans_hourly = trans_hourly
        self.lag_day = lag_day
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # index by count/observation
        yr = self.data["year"][idx]
        doy = self.data["doy"][idx]
        date = self.data["date"][idx]

        era5_main = self.era5_main.sel(date=date)
        era5_hourly = self.era5_hourly.sel(date=date)
        era5_daily = self.era5_daily.sel(date=date)

        # convert to numpy before transformations
        sample = yr, doy, era5_main, era5_hourly, era5_daily

        # apply transformations
        if self.transform:
            # to array
            sample = (
                np.zeros(1),
                np.array([yr]),
                np.array([doy]),
                era5_main.to_array().values,
                era5_hourly.to_array().values,
                era5_daily.to_array().values,
                np.zeros(1),
            )
            # to tensor
            sample = tuple([torch.FloatTensor(s) for s in sample])

        return sample

    def set_transform(self, value):
        self.transform = value


class DefileDataModule(LightningDataModule):
    """`LightningDataModule` for the Defile dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        species: str = "Buse variable",
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
        lag_day: int = 7,
        forecast_day: int = 5,
        seed: int = 0,
        train_val_test_cum_ratio: Tuple[float, float] = (0.7, 0.9),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DefileDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param species: The species for which to model the count. Defaults to `"Buse variable"`.
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
        self.lag_day = lag_day
        self.forecast_day = forecast_day
        self.seed = seed
        self.train_val_test_cum_ratio = np.array(train_val_test_cum_ratio)
        self.batch_size_per_device = batch_size

    # def prepare_data(self) -> None:
    #     """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
    #     within a single process on CPU, so you can safely add your downloading logic within. In
    #     case of multi-node training, the execution of this hook depends upon
    #     `self.prepare_data_per_node()`.

    #     Do not use it to assign state (self.x = y).
    #     """
    #     MNIST(self.hparams.data_dir, train=True, download=True)
    #     MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # split dataset years based on type of data collected
        yr_grp = [
            np.arange(1966, 1992),  # size 26. Incidental monitoring
            np.arange(1993, 2013),  # size 20.
            np.arange(2014, 2021),  # size 7.
        ]

        # Shuffle order of the year in each group
        np.random.seed(self.seed)
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
            lag_day=self.lag_day,
            transform=True,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # self.data_train = DefileDataset(
        #     data_dir=self.data_dir,
        #     species=self.species,
        #     era5_main_location=self.era5_main_location,
        #     era5_main_variables=self.era5_main_variables,
        #     era5_hourly_locations=self.era5_hourly_locations,
        #     era5_hourly_variables=self.era5_hourly_variables,
        #     era5_daily_locations=self.era5_daily_locations,
        #     era5_daily_variables=self.era5_daily_variables,
        #     years=self.ytraining,
        #     lag_day=self.lag_day,
        #     transform=True,
        # )

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
            transform=True,
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
            transform=True,
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
            trans_main=self.data_train.trans_main,
            trans_daily=self.data_train.trans_daily,
            trans_hourly=self.data_train.trans_hourly,
        )
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    # def teardown(self, stage: Optional[str] = None) -> None:
    #     """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
    #     `trainer.test()`, and `trainer.predict()`.

    #     :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
    #         Defaults to ``None``.
    #     """
    #     pass

    # def state_dict(self) -> Dict[Any, Any]:
    #     """Called when saving a checkpoint. Implement to generate and save the datamodule state.

    #     :return: A dictionary containing the datamodule state that you want to save.
    #     """
    #     return {}

    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
    #     `state_dict()`.

    #     :param state_dict: The datamodule state returned by `self.state_dict()`.
    #     """
    #     pass


if __name__ == "__main__":
    _ = DefileDataModule()
