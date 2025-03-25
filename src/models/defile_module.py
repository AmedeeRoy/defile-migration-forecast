import os
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
import datetime
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import ExplainedVariance, SpearmanCorrCoef
from scipy.stats import spearmanr

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import pickle
from src.models.criterion import applyMask


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
        output_dir: str
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net
        self.criterion = criterion
        self.output_dir = output_dir

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
        return torch.stack(
            [c.forward(count_pred, count, mask) for c in self.criterion]
        ).sum()

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

    ### TRAIN -------------------

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
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    ### VALIDATION -------------------

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
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
        pred_masked = applyMask(self.val_pred["pred"][:, 0, :], self.val_pred["mask"])

        # Compute R2 score
        explained_variance = ExplainedVariance()
        self.val_r2_score = explained_variance(pred_masked, obs)
        self.log(
            "val/r2_score",
            self.val_r2_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Compute spearman correlation coeff
        # spearman_coeff = SpearmanCorrCoef()
        # self.val_spearman_coeff = spearman_coeff(pred_masked, obs)
        obs_np = obs.cpu().numpy()
        pred_np = pred_masked.cpu().numpy()
        self.val_spearman_coeff, _ = spearmanr(pred_np, obs_np)

        self.log(
            "val/spearman_coeff",
            self.val_spearman_coeff,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # reinitialize validation step
        self.val_pred = {"obs": [], "mask": [], "pred": []}

    ### TEST -------------------

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # save all predictions
        count, yr, doy, era5_main, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)

        self.test_pred["obs"].append(count)  # single value
        self.test_pred["mask"].append(mask)  # hourly mask
        self.test_pred["pred"].append(count_pred)  # hourly count

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Concatenate batches
        for k in self.test_pred.keys():
            self.test_pred[k] = torch.cat(self.test_pred[k], 0).cpu()

        # Get masked predictions
        obs = self.test_pred["obs"].squeeze()

        pred_masked = applyMask(self.test_pred["pred"][:, 0, :], self.test_pred["mask"])

        # Compute R2 score
        explained_variance = ExplainedVariance()
        self.test_r2_score = explained_variance(pred_masked, obs)
        self.log(
            "test/r2_score",
            self.test_r2_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Compute spearman correlation coeff
        # spearman_coeff = SpearmanCorrCoef()
        # self.test_spearman_coeff = spearman_coeff(pred_masked, obs)
        obs_np = obs.cpu().numpy()
        pred_np = pred_masked.cpu().numpy()
        self.test_spearman_coeff, _ = spearmanr(pred_np, obs_np)
        self.log(
            "test/spearman_coeff",
            self.test_spearman_coeff,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Save test result to logger
        if self.trainer.logger:  # Only save if logger present (e.g., not during debug)
            self.save_test(self.trainer.datamodule.data_test, self.test_pred)

    def save_test(self, test_dataset, test_pred):

        # Retrieve ERA5 at the main location in xarray (not tensor) and WITHOUT transformation
        test_dataset.set_return_original(True)
        test = []

        for i in range(len(test_dataset)):
            _, _, _, era5_main, _, _, _ = test_dataset[i]
            t = era5_main.copy()
            # Add the predicted hourly count
            t = t.assign(pred_log_hourly_count=("time", test_pred["pred"][i, 0, :]))

            # Add the observed count
            t = t.assign(obs_count_=("", test_pred["obs"][i]))

            # Add the hourly mask
            t = t.assign(mask=("time", test_pred["mask"][i]))
            test.append(t)

        # Concatenate each data along date
        test = xr.concat(test, dim="date")

        # ugly way to deal with 'obs' dimensions
        test = test.assign(
            obs_count=(
                "date",
                test["obs_count_"].values.squeeze(),
            )
        )
        test = test.drop_dims("")

        # Add the total predicted count (sum according to mask)
        test = test.assign(
            pred_count=(
                "date",
                applyMask(test["pred_log_hourly_count"].values, test["mask"].values),
            )
        )

        test["time"] = test.time.astype(str)

        filename = "_".join(self.trainer.datamodule.species.split(" ")) + ".nc"
        print(test)
        test.to_netcdf(os.path.join(self.output_dir, filename))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.plt_counts_distribution(test)
        self.plt_true_vs_prediction(test)
        self.plt_timeseries(test)
        self.plt_timeseries(test, log_transformed=False)
        self.plt_doy_sum(test)

    def plt_counts_distribution(self, data):
        true_count = data.obs_count.values
        pred_count = data.pred_count.values

        bins = np.linspace(0, true_count.max(), 100)
        plt.hist(
            true_count,
            label="True count",
            alpha=0.5,
            edgecolor="k",
            bins=bins,
        )
        plt.hist(
            pred_count,
            label="Predicted count",
            alpha=0.5,
            edgecolor="k",
            bins=bins,
        )
        # plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Average hourly counts per periods")
        plt.ylabel("Histogram")
        plt.legend()

        filename = (
            "_".join(self.trainer.datamodule.species.split(" "))
            + "_counts_distribution.jpg"
        )
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plt_true_vs_prediction(self, data, log_transformed=True):
        true_count = data.obs_count.values
        pred_count = data.pred_count.values
        if log_transformed:
            true_count = np.log1p(true_count)
            pred_count = np.log1p(pred_count)

        x = np.linspace(min(true_count), max(true_count), 100)
        coef = np.polyfit(true_count, pred_count, 1)
        y = np.polyval(coef, x)

        plt.scatter(true_count, pred_count, c="black", s=5, alpha=0.4)
        plt.plot(x, y, c="red")
        plt.plot(x, x, "--", c="black")
        plt.xlabel(f"True count {'(log transformed)' if log_transformed else ''}")
        plt.ylabel(f"Predicted count {'(log transformed)' if log_transformed else ''}")

        filename = (
            "_".join(self.trainer.datamodule.species.split(" "))
            + "_true_vs_prediction.jpg"
        )
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plt_timeseries(self, data, log_transformed=True, global_y_lim=False):

        daily_average = data.mean(dim="time")["obs_count"]
        valid_indices = np.where(daily_average > 0)[0]
        weights = data.pred_log_hourly_count[valid_indices].sum(dim="time").values
        sampled_indices = np.random.choice(
            valid_indices, size=16, p=weights / weights.sum()
        )

        all_obs = []
        all_pred = []
        all_mask = []
        all_pred_first = []

        for d in daily_average[sampled_indices].date:
            subs = data.sel(date=d)
            if log_transformed:
                obs = np.log1p(subs["obs_count"])
                pred = subs["pred_log_hourly_count"]
            else:
                obs = subs["obs_count"]
                pred = np.expm1(subs["pred_log_hourly_count"])

            # If there is a single observation on that day, the structure of pred is different (no date dimension) and the plot need to be done differently.
            if "date" in pred.dims:
                pred_first = pred.isel(date=0)  # If date is a dimension
                mask = subs.mask.values
                # mask = subs.mask.sum(dim="date").values  # summing over all observations.
                obs = obs.values
            else:
                pred_first = pred
                obs = [obs.values]
                mask = [subs.mask.values]

            all_obs.append(obs)
            all_pred.append(pred)
            all_pred_first.append(pred_first)
            all_mask.append(mask)

        # Compute maximum y value across all subplots
        ymax = (
            max(
                np.max([np.max(p.values) for p in all_pred]),
                np.max([np.max(o) for o in all_obs]),
            )
            + 0.1
        )

        fig, ax = plt.subplots(4, 4, figsize=(12, 8), tight_layout=True)
        ax = ax.flatten()

        for i, d in enumerate(daily_average[sampled_indices].date):

            # plot the prediction (only the first prediction is show - all should be the same for the day)
            ax[i].plot(np.arange(0, 24), all_pred_first[i])

            # find the max y value for drawing the rectangle
            if not global_y_lim:
                ymax = max(all_pred[i].max(), max(all_obs[i])) + 0.1

            # Plot the mask as yellow transparant background
            for k, m in enumerate(np.sum(all_mask[i], axis=0)):
                ax[i].add_patch(
                    Rectangle((k, 0), 1, ymax, color=(1, 1, 0, min(1, m)))
                )  # RGBA: (1, 1, 0) is yellow, 'm' controls the alpha

            for u, o in enumerate(all_obs[i]):
                first_nonzero = np.argmax(all_mask[i][u] > 0)
                last_nonzero = (
                    len(all_mask[i][u]) - 1 - np.argmax(np.flip(all_mask[i][u]) > 0)
                )

                ax[i].plot([first_nonzero, last_nonzero + 1], [o, o], c="tab:red")

        # ax[i].set_ylim(0, ymax)
        ax[i].set_xlabel("hours")
        ax[i].set_ylabel(f"Bird counts {'(transform)' if log_transformed else ''}")
        ax[i].set_title(d.date.dt.strftime("%Y-%m-%d").item())

        filename = (
            "_".join(self.trainer.datamodule.species.split(" "))
            + "_timeseries"
            + ("_log_transformed" if log_transformed else "")
            + ".jpg"
        )
        fig.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plt_doy_sum(self, data):
        # Convert data into a DataFrame with only the necessary columns
        data_df = data[["obs_count", "pred_count", "date"]].to_dataframe().reset_index()

        # Extract doy and year from date
        data_df["doy"] = data_df["date"].dt.dayofyear
        data_df["year"] = data_df["date"].dt.year

        # Get unique years and determine layout for subplots
        unique_years = np.unique(data_df["year"].values)
        n_years = len(unique_years)

        # Define the number of columns (2 columns if more than 4 years)
        n_cols = 2 if n_years > 4 else 1
        n_rows = (n_years + n_cols - 1) // n_cols  # Calculate number of rows

        # Set up subplots with dynamic row/column layout
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True
        )

        # Flatten the axes array if necessary
        if n_years == 1:
            axes = [axes]  # In case there's only one year, make sure axes is iterable
        else:
            axes = axes.flatten()

        # Plot for each year
        for ax, y in zip(axes, unique_years):
            yearly_data = data_df[data_df["year"] == y]
            yearly_data.groupby("doy").mean().obs_count.plot(ax=ax, label="True")
            yearly_data.groupby("doy").mean().pred_count.plot(ax=ax, label="Prediction")
            ax.set_ylabel(f"Average hourly count ({y})")
            ax.legend()

        # Set x-label for the last row's axes
        for ax in axes[-n_cols:]:  # Only the last row should have x-label
            ax.set_xlabel("Day of Year")

        plt.tight_layout()

        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "_doy_sum.jpg"
        fig.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    ### EXPORT PREDICTIONS -------------------
    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
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
            self.predict_pred[k] = torch.cat(self.predict_pred[k], 0).cpu()

        self.save_predict(self.trainer.datamodule.data_predict, self.predict_pred)

    def save_predict(self, predict_dataset, predict_pred):
        predict_dataset.set_return_original(True)
        predictions = []

        for i in range(len(predict_dataset)):
            _, _, era5_main, _, _ = predict_dataset[i]
            pred = era5_main.copy()
            pred = pred.assign(
                pred_log_hourly_count=("time", predict_pred["pred"][i, 0, :])
            )
            predictions.append(pred)

        predictions = xr.concat(predictions, dim="date")
        # predictions["time"] = predictions.time.astype(str)
        today = datetime.date.today().strftime("%Y%m%d")
        filename = (
            "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".nc"
        )

        if self.trainer.logger:
            predictions.to_netcdf(os.path.join(self.output_dir, filename))
            self.plt_predict(predictions)

    def plt_predict(self, data):

        pred_count = np.expm1(data.pred_log_hourly_count)

        fig, ax = plt.subplots(
            2, 3, figsize=(10, 5), tight_layout=True, sharex=True, sharey=True
        )
        ax = ax.flatten()
        for k in range(len(pred_count)):
            subset = pred_count.isel(date=k)

            ax[k].bar(np.arange(24), subset.values)

            ax[k].set_title(subset.date.dt.strftime("%Y-%m-%d").item())
            ax[k].set_xticks(
                np.arange(0, 24, 3), [str(h) + "h" for h in np.arange(0, 24, 3)]
            )

            ax[k].text(
                0.05,
                0.93,
                f"Total = {np.sum(subset.values):.0f}",
                transform=ax[k].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="gray", alpha=0.25),
            )
            ax[k].set_xlim(6, 21)

        ax[0].set_ylabel("Forecasted individual \ncounts (#)")
        ax[3].set_ylabel("Forecasted individual \ncounts (#)")
        plt.suptitle(f"Defile Bird Forecasts - {self.trainer.datamodule.species}")
        today = datetime.date.today().strftime("%Y%m%d")
        filename = (
            "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".png"
        )
        plt.savefig(os.path.join(self.output_dir, filename))
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
