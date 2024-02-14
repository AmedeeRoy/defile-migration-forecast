from typing import Any, Dict, Tuple

import os
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import ExplainedVariance

class DefileLitModule(LightningModule):
    """
    A `LightningModule` implements 8 key methods:

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
        criterion : Any,
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
        self.val_pred = {'obs': [], 'mask': [], 'pred': []}
        self.test_pred = {'obs': [], 'mask': [], 'pred': []}
    
    def loss(self, count_pred, count, mask):
        return torch.stack([c.forward(count_pred, count, mask) for c in self.criterion]).sum()

    def forward(self, yr, doy, era5_hourly) -> torch.Tensor:
        """
        Perform a forward pass through the model `self.net`.
        """
        return self.net(yr, doy, era5_hourly)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

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
        count, yr, doy, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_hourly)
        loss = self.loss(count_pred, count, mask)
        return loss

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
        
        loss  = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictions
        count, yr, doy, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_hourly)
        self.val_pred['obs'].append(count)
        self.val_pred['mask'].append(mask)
        self.val_pred['pred'].append(count_pred)  

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        for k in self.val_pred.keys():
            self.val_pred[k] = torch.cat(self.val_pred[k], 0)
        self.val_pred['pred_masked'] = torch.sum(self.val_pred['pred'].squeeze()*self.val_pred['mask'], dim=1).unsqueeze(1)

        explained_variance = ExplainedVariance()
        self.val_r2_score = explained_variance(self.val_pred['pred_masked'], self.val_pred['obs'])
        self.log("val/r2_score", self.val_r2_score, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss  = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictions
        count, yr, doy, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, era5_hourly)
        self.test_pred['obs'].append(count)
        self.test_pred['mask'].append(mask)
        self.test_pred['pred'].append(count_pred)    
        

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        for k in self.test_pred.keys():
            self.test_pred[k] = torch.cat(self.test_pred[k], 0)
        self.test_pred['pred_masked'] = torch.sum(self.test_pred['pred'].squeeze()*self.test_pred['mask'], dim=1).unsqueeze(1)

        explained_variance = ExplainedVariance()
        self.test_r2_score = explained_variance(self.test_pred['pred_masked'], self.test_pred['obs'])
        self.log("test/r2_score", self.test_r2_score, on_step=False, on_epoch=True, prog_bar=True)


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

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
