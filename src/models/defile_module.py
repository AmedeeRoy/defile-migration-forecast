import datetime
import os
import pickle
from typing import Any, Dict, Tuple

import torch
import xarray as xr
from captum.attr import IntegratedGradients, Saliency
from lightning import LightningModule
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import ExplainedVariance, SpearmanCorrCoef

from src.models.criterion import applyMask
from src.plots.explanations import *
from src.plots.save_predict import *
from src.plots.save_test import *


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
        output_dir: str,
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
        self.test_explanation = []

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
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    ### VALIDATION -------------------
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
    def on_test_epoch_start(self) -> None:
        # Defining Saliency interpreter
        self.explainer = Saliency(self.explainable_model_step)

    def explainable_model_step(self, yr, doy, era5_main, era5_hourly, era5_daily):
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)
        count_pred = torch.mean(count_pred[:, 0, :], dim=1)
        return count_pred

    def explain(self, batch) -> None:
        batch = (b.requires_grad_() for b in batch)
        count, yr, doy, era5_main, era5_hourly, era5_daily, mask = batch
        saliency = self.explainer.attribute((yr, doy, era5_main, era5_hourly, era5_daily))
        return saliency

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        # /!\ work only when Trainer(inference_mode = False)
        with torch.enable_grad():
            saliency = self.explain(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictionsnstall
        count, yr, doy, era5_main, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_main, era5_hourly, era5_daily)

        self.test_pred["obs"].append(count)  # single value
        self.test_pred["mask"].append(mask)  # hourly mask
        self.test_pred["pred"].append(count_pred)  # hourly count
        self.test_explanation.append(saliency)  # saliency

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Concatenate batches
        for k in self.test_pred.keys():
            self.test_pred[k] = torch.cat(self.test_pred[k], 0).cpu()

        self.test_explanation = [
            torch.cat([exp[k] for exp in self.test_explanation], dim=0).cpu()
            for k in range(len(self.test_explanation[0]))
        ]
        # filepath = os.path.join(self.output_dir, 'test_explanation.pickle')
        # with open(filepath, 'wb') as f:
        #     pickle.dump(self.test_explanation, f)

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
            self.save_test()

    def save_test(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Retrieve ERA5 at the main location in xarray (not tensor) and WITHOUT transformation
        test_dataset = self.trainer.datamodule.data_test
        test_dataset.set_return_original(True)
        test = []

        for i in range(len(test_dataset)):
            _, _, _, era5_main, _, _, _ = test_dataset[i]
            t = era5_main.copy()
            # Add the predicted hourly count
            t = t.assign(pred_log_hourly_count=("time", self.test_pred["pred"][i, 0, :]))

            if self.test_pred["pred"].shape[1] == 2:
                t = t.assign(pred_log_hourly_count_std=("time", self.test_pred["pred"][i, 1, :]))

            # Add the observed count
            t = t.assign(obs_count_=("", self.test_pred["obs"][i]))

            # Add the hourly mask
            t = t.assign(mask=("time", self.test_pred["mask"][i]))
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
        if self.test_pred["pred"].shape[1] == 2:
            test = test.assign(
                pred_count_down=(
                    "date",
                    applyMask(
                        np.clip(
                            test["pred_log_hourly_count"].values
                            - test["pred_log_hourly_count_std"].values,
                            a_min=0,
                            a_max=None,
                        ),
                        test["mask"].values,
                    ),
                ),
                pred_count_up=(
                    "date",
                    applyMask(
                        test["pred_log_hourly_count"].values
                        + test["pred_log_hourly_count_std"].values,
                        test["mask"].values,
                    ),
                ),
            )

        test["time"] = test.time.astype(str)

        ## Save predictions ------------------
        filename = "_".join(self.trainer.datamodule.species.split(" ")) + ".nc"
        test.to_netcdf(os.path.join(self.output_dir, filename))
        print(test)

        ## Save plots ------------------
        filename = (
            "_".join(self.trainer.datamodule.species.split(" ")) + "_counts_distribution.jpg"
        )
        plt_counts_distribution(
            test.obs_count.values,
            test.pred_count.values,
            filepath=os.path.join(self.output_dir, filename),
        )

        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "_true_vs_prediction.jpg"
        plt_true_vs_prediction(
            test.obs_count.values,
            test.pred_count.values,
            filepath=os.path.join(self.output_dir, filename),
        )

        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "_timeseries.jpg"
        plt_timeseries(
            test, log_transformed=False, filepath=os.path.join(self.output_dir, filename)
        )

        filename = (
            "_".join(self.trainer.datamodule.species.split(" "))
            + "_timeseries_log_transformed.jpg"
        )
        plt_timeseries(
            test, log_transformed=True, filepath=os.path.join(self.output_dir, filename)
        )
        filename = "_".join(self.trainer.datamodule.species.split(" ")) + "_doy_sum.jpg"
        plt_doy_sum(test, filepath=os.path.join(self.output_dir, filename))

        filename = (
            "_".join(self.trainer.datamodule.species.split(" ")) + "_contributions_metrics.jpg"
        )
        plt_explanations_metrics(
            self.trainer.datamodule,
            self.test_explanation,
            filepath=os.path.join(self.output_dir, filename),
        )

        filename = (
            "_".join(self.trainer.datamodule.species.split(" ")) + "_contributions_locations.jpg"
        )
        plt_explanations_locations(
            self.trainer.datamodule,
            self.test_explanation,
            filepath=os.path.join(self.output_dir, filename),
        )

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
            self.predict_pred[k] = torch.cat(self.predict_pred[k], 0).cpu()

        self.save_predict()

    def save_predict(self):
        predict_dataset = self.trainer.datamodule.data_predict
        predict_dataset.set_return_original(True)
        predictions = []

        for i in range(len(predict_dataset)):
            _, _, era5_main, _, _ = predict_dataset[i]
            pred = era5_main.copy()
            pred = pred.assign(pred_log_hourly_count=("time", self.predict_pred["pred"][i, 0, :]))
            predictions.append(pred)

        predictions = xr.concat(predictions, dim="date")
        # predictions["time"] = predictions.time.astype(str)
        today = datetime.date.today().strftime("%Y%m%d")
        filename = "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".nc"

        filename = "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".nc"
        predictions.to_netcdf(os.path.join(self.output_dir, filename))

        filename = "_".join([today] + self.trainer.datamodule.species.split(" ")) + ".jpg"
        filepath = os.path.join(self.output_dir, filename)
        plt_predict(predictions, species=self.trainer.datamodule.species, filepath=filepath)

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
