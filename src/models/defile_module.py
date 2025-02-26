import os
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
import datetime
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import ExplainedVariance, SpearmanCorrCoef

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class DefileLitModule(LightningModule):
    """A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Any,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for saving predictions
        self.val_pred = {"obs": [], "mask": [], "pred": []}
        self.test_pred = {"obs": [], "mask": [], "pred": []}
        self.predict_pred = {"pred": []}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, yr, doy, era5_main, era5_hourly, era5_daily) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(yr, doy, era5_main, era5_hourly, era5_daily)

    def loss(self, count_pred, count, mask):
        return torch.stack([c.forward(count_pred, count, mask) for c in self.criterion]).sum()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        count, yr, doy, era5_main, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)
        loss = self.loss(count_pred, count, mask)
        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictions
        count, yr, doy, era5_main, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)

        self.val_pred["obs"].append(count)
        self.val_pred["mask"].append(mask)
        self.val_pred["pred"].append(count_pred)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        # Concatenate batches
        for k in self.val_pred.keys():
            self.val_pred[k] = torch.cat(self.val_pred[k], 0)

        # Get masked predictions
        obs = self.val_pred["obs"].squeeze()
        pred_masked = torch.sum(
            self.val_pred["pred"][:, 0, :].squeeze() * self.val_pred["mask"], dim=1
        )

        # Compute R2 score
        explained_variance = ExplainedVariance()
        self.val_r2_score = explained_variance(pred_masked, obs)
        self.log("val/r2_score", self.val_r2_score, on_step=False, on_epoch=True, prog_bar=True)

        # Compute spearman correlation coeff
        spearman_coeff = SpearmanCorrCoef()
        self.val_spearman_coeff = spearman_coeff(pred_masked, obs)
        self.log(
            "val/spearman_coeff",
            self.val_spearman_coeff,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # reinitialize validation step
        self.val_pred = {"obs": [], "mask": [], "pred": []}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictions
        count, yr, doy, era5_main, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)

        self.test_pred["obs"].append(count)
        self.test_pred["mask"].append(mask)
        self.test_pred["pred"].append(count_pred)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Concatenate batches
        for k in self.test_pred.keys():
            self.test_pred[k] = torch.cat(self.test_pred[k], 0)

        # Get masked predictions
        obs = self.test_pred["obs"].squeeze()
        pred_masked = torch.sum(
            self.test_pred["pred"][:, 0, :].squeeze() * self.test_pred["mask"], dim=1
        )

        # Compute R2 score
        explained_variance = ExplainedVariance()
        self.test_r2_score = explained_variance(pred_masked, obs)
        self.log("test/r2_score", self.test_r2_score, on_step=False, on_epoch=True, prog_bar=True)

        # Compute spearman correlation coeff
        spearman_coeff = SpearmanCorrCoef()
        self.test_spearman_coeff = spearman_coeff(pred_masked, obs)
        self.log(
            "test/spearman_coeff",
            self.test_spearman_coeff,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Compute spearman correlation coeff
        self.save_test(self.trainer.datamodule.data_test, self.test_pred)

    def save_test(self, test_dataset, test_pred):
        test_dataset.set_transform(False)
        predictions = []

        for i in range(len(test_dataset)):
            count, yr, doy, era5_main, era5_hourly, era5_daily, mask = test_dataset[i]
            pred = era5_main.copy()
            pred = pred.assign(estimated_hourly_counts=("time", test_pred["pred"][i, 0, :]))
            if test_pred["pred"].shape[1] > 1:
                pred = pred.assign(
                    estimated_hourly_counts_var=("time", test_pred["pred"][i, 1, :])
                )

            pred = pred.assign(masked_total_counts_=("", test_pred["obs"][i]))
            pred = pred.assign(mask=("time", test_pred["mask"][i]))
            predictions.append(pred)

        predictions = xr.concat(predictions, dim="date")
        # ugly way to deal with 'obs' dimensions
        predictions = predictions.assign(
            masked_total_counts=("date", predictions["masked_total_counts_"].values.squeeze())
        )
        predictions = predictions.drop_dims("")
        predictions = predictions.assign(
            estimated_masked_total_counts=(
                predictions["estimated_hourly_counts"] * predictions["mask"]
            ).sum(dim="time")
        )

        predictions["time"] = predictions.time.astype(str)
        
        filename = "_".join(self.trainer.datamodule.species.split(" ")) + ".nc"
        predictions.to_netcdf(os.path.join(self.trainer.logger.log_dir, filename))
        print(predictions)

        if not os.path.exists(self.trainer.logger.log_dir):
            os.makedirs(self.trainer.logger.log_dir)

        self.plt_counts_distribution(predictions)
        self.plt_true_vs_prediction(predictions)
        self.plt_timeseries(predictions)
        self.plt_doy_sum(predictions)

    def plt_counts_distribution(self, data):
        true_count = data.masked_total_counts.values
        pred_count = data.estimated_masked_total_counts.values

        plt.hist(
            true_count, label="True count", alpha=0.5, edgecolor="k", bins=np.linspace(0, 10, 50)
        )
        plt.hist(
            pred_count,
            label="Predicted count",
            alpha=0.5,
            edgecolor="k",
            bins=np.linspace(0, 10, 50),
        )
        plt.xlabel("Counts (transform-scale)")
        plt.ylabel("Histogram")
        plt.legend()

        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "counts_distribution.jpg"
        plt.savefig(os.path.join(self.trainer.logger.log_dir, filename))
        plt.close()

    def plt_true_vs_prediction(self, data):
        true_count = data.masked_total_counts.values
        pred_count = data.estimated_masked_total_counts.values

        x = np.linspace(min(true_count), max(true_count), 100)
        coef = np.polyfit(true_count, pred_count, 5)
        y = np.polyval(coef, x)

        plt.scatter(true_count, pred_count, c="black", s=5, alpha=0.4)
        plt.plot(x, y, c="red")
        plt.plot(x, x, "--", c="black")
        plt.xlabel("True count (transform-scale)")
        plt.ylabel("Predicted count (transform-scale)")
        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "true_vs_prediction.jpg"
        plt.savefig(os.path.join(self.trainer.logger.log_dir, filename))
        plt.close()

    def plt_timeseries(self, data):
        fig, ax = plt.subplots(4, 4, figsize=(12, 8), tight_layout=True)
        ax = ax.flatten()
        for i, d in enumerate(np.random.randint(0, len(data.date), size=16)):
            subs = data.isel(date=d)

            ax[i].plot(np.arange(0, 24), subs.estimated_hourly_counts)
            for k, m in enumerate(subs.mask.values):
                if m == 1:
                    ax[i].add_patch(Rectangle((k, 0), m, 10, color="yellow"))
                    obs = subs.masked_total_counts.item() / subs.mask.sum().item()
                    ax[i].plot([k, k + 1], [obs, obs], c="tab:red")

            ax[i].set_ylim(0, max(subs.estimated_hourly_counts.max(), obs) + 0.1)
            ax[i].set_xlabel("hours")
            ax[i].set_ylabel("Bird counts (transform)")
            ax[i].set_title(data.isel(date=d).date.dt.strftime("%Y-%m-%d").item())
        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "timeseries.jpg"
        fig.savefig(os.path.join(self.trainer.logger.log_dir, filename))
        plt.close()

    def plt_doy_sum(self, data):
        data = data.assign_coords(doy=data.date.dt.dayofyear)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        data.groupby("doy").sum().masked_total_counts.plot(ax=ax, label="True")
        data.groupby("doy").sum().estimated_masked_total_counts.plot(ax=ax, label="Prediction")
        plt.legend()
        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "doy_sum.jpg"
        fig.savefig(os.path.join(self.trainer.logger.log_dir, filename))
        plt.close()

   ### EXPORT PREDICTIONS -------------------
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single forward step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # save all predictions
        _, yr, doy, era5_main, era5_hourly, era5_daily, _ = batch
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)

        self.predict_pred["pred"].append(count_pred)

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        # Concatenate batches
        for k in self.predict_pred.keys():
            self.predict_pred[k] = torch.cat(self.predict_pred[k], 0)

        self.save_predict(self.trainer.datamodule.data_predict, self.predict_pred)

    def save_predict(self, predict_dataset, predict_pred):
        predict_dataset.set_transform(False)
        predictions = []

        for i in range(len(predict_dataset)):
            yr, doy, era5_main, era5_hourly, era5_daily = predict_dataset[i]
            pred = era5_main.copy()
            pred = pred.assign(estimated_hourly_counts=("time", predict_pred["pred"][i, 0, :]))
            if predict_pred["pred"].shape[1] > 1:
                pred = pred.assign(
                    estimated_hourly_counts_var=("time", predict_pred["pred"][i, 1, :])
                )
            predictions.append(pred)

        predictions = xr.concat(predictions, dim="date")
        # predictions["time"] = predictions.time.astype(str)
        today = datetime.date.today().strftime("%Y%m%d")
        filename = "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".nc"

        predictions.to_netcdf(os.path.join(self.trainer.logger.log_dir, filename))
        self.plt_predict(predictions)

    def plt_predict(self, data):
        daily_counts_transform = data.sum(dim="time").estimated_hourly_counts
        daily_counts = (10 * daily_counts_transform) ** 2

        data_smooth = data.rolling({"time": 3}, center=True).mean()

        fig, ax = plt.subplots(2, 3, figsize=(10, 5), tight_layout=True, sharex=True, sharey=True)
        ax = ax.flatten()
        for k in range(len(data.date)):
            subset = data_smooth.isel(date=k)

            weights = subset.estimated_hourly_counts / subset.estimated_hourly_counts.sum()

            ax[k].bar(np.arange(24), weights * daily_counts[k])

            ax[k].set_title(subset.date.dt.strftime("%Y-%m-%d").item())
            ax[k].set_xticks(np.arange(0, 24, 3), [str(h) + "h" for h in np.arange(0, 24, 3)])

            ax[k].text(
                0.05,
                0.93,
                f"Total = {daily_counts[k]:.0f}",
                transform=ax[k].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="gray", alpha=0.25),
            )

        ax[0].set_ylabel("Forecasted individual \ncounts (#)")
        ax[3].set_ylabel("Forecasted individual \ncounts (#)")
        plt.suptitle(f"Defile Bird Forecasts - {self.trainer.datamodule.species}")
        today = datetime.date.today().strftime("%Y%m%d")
        filename = "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".png"
        plt.savefig(os.path.join(self.trainer.logger.log_dir, filename))
        plt.close()


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DefileLitModule(None, None, None, None)
