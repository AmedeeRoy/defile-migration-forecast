from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from suncalc import get_position
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms

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
        years=range(1966, 2023),
        lag_day=7,
        transform=False,
    ):
        # WEATHER DATA ----------------------------
        # Create data xarray (better to handle multi-indexing)
        era5_hourly = pd.read_csv(data_dir + "/era5_hourly.csv", parse_dates=["datetime"])
        era5_hourly["date"] = pd.to_datetime(era5_hourly["datetime"].dt.date)
        era5_hourly["time"] = pd.to_timedelta(era5_hourly.datetime.dt.time.astype(str))
        # Add sun position information
        lon = 5.8919
        lat = 46.1178
        sun_position = get_position(era5_hourly["datetime"], lon, lat)
        era5_hourly["sun_altitude"] = sun_position["altitude"]
        era5_hourly["sun_azimuth"] = sun_position["azimuth"]
        era5_hourly = era5_hourly.drop("datetime", axis=1)
        era5_hourly = era5_hourly.set_index(
            ["date", "time"]
        ).to_xarray()  # date and time as distinct indexes

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

        # COUNT DATA ----------------------------
        # Read data
        df = pd.read_csv(
            data_dir + "/all_count_processed.csv", parse_dates=["date", "start", "end"]
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
        if transform:
            # era5_daily_lagged = (era5_daily_lagged - era5_daily_lagged.mean(dim = "date")) / era5_daily_lagged.std(dim = "date")
            # era5_hourly = (era5_hourly - era5_hourly.mean(dim = "date")) / era5_hourly.std(dim = "date")
            era5_daily_lagged = (
                era5_daily_lagged - era5_daily_lagged.mean()
            ) / era5_daily_lagged.std()
            era5_hourly = (era5_hourly - era5_hourly.mean()) / era5_hourly.std()
            dfys["count"] = np.log10(1 + dfys["count"])
            dfys["doy"] = (dfys["doy"] - 183) / 366
            dfys["year"] = (dfys["year"] - 2000) / 100

        # Assign to self
        self.data = dfys.reset_index(drop=True)
        self.era5_daily = era5_daily_lagged
        self.era5_hourly = era5_hourly
        self.mask = mask
        self.lag_day = lag_day
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # index by count/observation
        count = self.data["count"][idx]
        doy = self.data["doy"][idx]
        yr = self.data["year"][idx]
        m = self.mask[:, idx]

        date = self.data["date"][idx]
        era5_h = self.era5_hourly.sel(date=date)
        era5_d = self.era5_daily.sel(date=date)

        # convert to numpy before transformations
        sample = count, yr, doy, era5_h, era5_d, m

        # apply transformations
        if self.transform:
            # to array
            sample = (
                np.array([count]),
                np.array([yr]),
                np.array([doy]),
                era5_h.to_array().values,
                era5_d.to_array().values,
                m,
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
        lag_day: int = 7,
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
        self.lag_day = lag_day
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
        ytraining, yval, ytest = [], [], []
        for y in yr_grp:
            sz = (len(y) * self.train_val_test_cum_ratio).astype(int)
            y_data = np.split(y, sz)
            ytraining.extend(y_data[0])
            yval.extend(y_data[1])
            ytest.extend(y_data[2])

        log.info(f"Train dataset : selected years - {ytraining}")
        log.info(f"Validation dataset : selected years - {yval}")
        log.info(f"Test dataset : selected years -{ytest}")

        # Create the three dataset (train, validation and test)
        self.data_train = DefileDataset(
            self.data_dir,
            species=self.species,
            years=ytraining,
            lag_day=self.lag_day,
            transform=True,
        )
        self.data_val = DefileDataset(
            self.data_dir, species=self.species, years=yval, lag_day=self.lag_day, transform=True
        )
        self.data_test = DefileDataset(
            self.data_dir, species=self.species, years=ytest, lag_day=self.lag_day, transform=True
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
        return DataLoader(
            dataset=self.data_test,
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
