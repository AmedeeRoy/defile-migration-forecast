import datetime
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from captum.attr import Saliency
from lightning import LightningModule
from scipy.stats import spearmanr
from torchmetrics import MeanMetric
from torchmetrics.regression import ExplainedVariance

from src import metrics as M
from src.models.criterion import applyMask
from src.plots.report import build_report
from src.plots.save_predict import plt_predict
from src.utils import RankedLogger
from src.utils.rich_utils import print_metrics_table

log = RankedLogger(__name__, rank_zero_only=True)


def _years_str(years) -> str:
    """Collapse a sorted year list to `1966-1992 (12)` -- 40 individual years do not fit
    on a report line, and the range plus the count is what actually gets compared."""
    if not len(years):
        return "none"
    return f"{years[0]}-{years[-1]} ({len(years)})" if len(years) > 1 else str(years[0])


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
        criterion: Dict[str, Any],
        compile: bool,
        output_dir: str,
        compute_saliency: bool = True,
        saliency_max_batches: int = 20,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compute_saliency: Whether to compute Captum Saliency attributions during
            testing for the contribution plots. Requires `trainer.inference_mode: False`.
            Set to False to skip it entirely. Defaults to `True`.
        :param saliency_max_batches: Saliency is computed on at most this many leading test
            batches rather than the whole test set -- attributions are only ever consumed as
            an average over samples (see `plt_explanations_*`), so a subsample is enough, and
            it avoids paying the backward-pass cost and memory for every batch.
            Ignored when `compute_saliency` is False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net
        self.criterion = criterion
        self.output_dir = output_dir
        self.compute_saliency = compute_saliency
        self.saliency_max_batches = saliency_max_batches

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for saving predictions
        self.val_pred = {"obs": [], "mask": [], "pred": []}
        self.test_pred = {"obs": [], "mask": [], "pred": []}
        self.test_explanation = []

        self.predict_pred = {"pred": []}

        # Phenology baseline, loaded lazily on first use and reused for every epoch:
        # it is a small JSON, but re-reading and re-parsing it once per validation epoch
        # for a number that is logged every epoch is pure overhead.
        self._phenology: Optional[M.Phenology] = None

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(yr, doy, prior_shape, era5_main, era5_hourly, era5_daily)

    def loss(self, count_pred, count, mask):
        return torch.stack([c.forward(count_pred, count, mask) for c in self.criterion.values()]).sum()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
        """
        count, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily, mask = batch
        count_pred = self.forward(yr, doy, prior_shape, era5_main, era5_hourly, era5_daily)
        loss = self.loss(count_pred, count, mask)
        return loss, count_pred

    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Collect a per-rank-local tensor across DDP processes and restore dataset row order.

        `self.val_pred`/`self.test_pred`/`self.predict_pred` are built one rank-local batch
        at a time. Under `trainer=gpu devices: 1` (what every documented command in this
        repo actually uses) there is only one rank and this is a no-op. But
        `configs/trainer/ddp.yaml` exists and is selectable, and without this call every
        metric in `on_validation_epoch_end`/`on_test_epoch_end` -- and every file
        `save_test`/`save_predict` write -- would silently reflect only rank 0's shard of
        the data.

        `self.all_gather` documents that for `world_size > 1` it returns shape
        `(world_size, batch, ...)`, with no extra dim added when `world_size == 1`. Lightning
        hands rank `r` the interleaved indices `r, r+world_size, r+2*world_size, ...` from
        `DistributedSampler` (since every val/test/predict dataloader here sets
        `shuffle=False`, see `DefileDataModule._dataloader`), so transposing the rank axis
        in front of the local-batch axis and flattening interleaves the ranks back into
        that same order -- row `i` of the result is row `i` of `dataset.count`, which
        `src.metrics.build_frame` and every write in this module assume positionally.

        Caveat inherited from `DistributedSampler`, not introduced here: when the dataset
        size doesn't divide evenly by `world_size`, it pads by repeating a few leading
        samples on the last rank so every rank's local batch has equal size (`all_gather`
        would otherwise hang) -- those few rows are duplicated in the gathered result. This
        project's val/test/predict sets are never sharded in practice (`devices: 1`), so
        that padding never triggers; it would need a custom sampler to eliminate outright
        if DDP were ever used for real.
        """
        gathered = self.all_gather(tensor)
        if gathered.dim() == tensor.dim():
            return gathered  # world_size == 1: all_gather added no dimension
        return gathered.transpose(0, 1).reshape(-1, *tensor.shape[1:])

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
        loss, _ = self.model_step(batch)

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

        loss, count_pred = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictions (reuse the forward pass from model_step)
        count, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily, mask = batch

        self.val_pred["obs"].append(count)
        self.val_pred["mask"].append(mask)
        self.val_pred["pred"].append(count_pred)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        # Concatenate rank-local batches, then gather across DDP processes (a no-op at
        # world_size == 1) so every metric below scores the whole validation set rather
        # than whichever shard this rank happened to see -- see `_gather`.
        preds = {k: self._gather(torch.cat(v, 0)) for k, v in self.val_pred.items()}

        # Get masked predictions
        obs = preds["obs"].squeeze()
        pred_masked = applyMask(preds["pred"][:, 0, :], preds["mask"])

        # Compute R2 score
        self.val_r2_score = ExplainedVariance()(pred_masked, obs)
        self.log("val/r2_score", self.val_r2_score, on_step=False, on_epoch=True, prog_bar=True)

        # Compute spearman correlation coeff
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

        # Skill against day-of-year phenology, logged every epoch rather than only at
        # test time: val/loss says the model is improving at its own objective, this says
        # whether the weather features are buying anything over knowing the date alone.
        # A run whose val/loss falls while this stays at or below zero is learning the
        # season, not the weather (DEVELOPMENT.md Phase 1).
        skill = self._skill_vs_phenology("val", preds["pred"][:, 0, :].cpu().numpy())
        if skill is not None:
            self.log(
                "val/skill_vs_phenology",
                skill,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # reinitialize validation step
        self.val_pred = {"obs": [], "mask": [], "pred": []}

    def _skill_vs_phenology(self, split: str, pred_hourly: np.ndarray) -> Optional[float]:
        """Row-level MAE skill against phenology for one split, or None if unavailable.

        Returns None rather than raising whenever the predictions do not line up with the
        split's rows -- Lightning's sanity check and `debug=limit` both run a truncated
        loop, and a diagnostic metric must never be the thing that breaks a debug run.
        """
        dataset = getattr(self.trainer.datamodule, f"data_{split}", None)
        if dataset is None or len(pred_hourly) != len(dataset.count):
            return None

        try:
            phenology = self.phenology
        except (FileNotFoundError, KeyError) as err:  # no phenology for this species
            log.warning(f"Skipping skill-vs-phenology: {err}")
            return None

        return M.validation_skill(dataset.count, dataset.mask, pred_hourly, phenology)

    @property
    def phenology(self) -> M.Phenology:
        """The species' day-of-year phenology, loaded once per run."""
        if self._phenology is None:
            datamodule = self.trainer.datamodule
            self._phenology = M.Phenology.load(datamodule.data_dir, datamodule.species)
        return self._phenology

    ### TEST -------------------
    def on_test_epoch_start(self) -> None:
        # Defining Saliency interpreter
        if self.compute_saliency:
            self.explainer = Saliency(self.explainable_model_step)

    def explainable_model_step(self, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily):
        count_pred = self.forward(yr, doy, prior_shape, era5_main, era5_hourly, era5_daily)
        count_pred = torch.mean(count_pred[:, 0, :], dim=1)
        return count_pred

    def explain(self, batch) -> None:
        batch = (b.requires_grad_() for b in batch)
        count, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily, mask = batch
        saliency = self.explainer.attribute(
            (yr, doy, prior_shape, era5_main, era5_hourly, era5_daily)
        )
        return saliency

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, count_pred = self.model_step(batch)

        # Saliency needs grad-tracking on the batch tensors (only possible with
        # Trainer(inference_mode=False)) and a full backward-style pass per attributed
        # sample, so it is capped to the first `saliency_max_batches` test batches rather
        # than run on every one. The contribution plots only ever
        # consume the mean attribution across samples, so a subsample is representative.
        if self.compute_saliency and batch_idx < self.saliency_max_batches:
            with torch.enable_grad():
                saliency = self.explain(batch)
            self.test_explanation.append(saliency)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save all predictions (reuse the forward pass from model_step)
        count, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily, mask = batch

        self.test_pred["obs"].append(count)  # single value
        self.test_pred["mask"].append(mask)  # hourly mask
        self.test_pred["pred"].append(count_pred)  # hourly count

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Concatenate rank-local batches, then gather across DDP processes (a no-op at
        # world_size == 1) so every metric, plot and file below reflects the whole test
        # set rather than whichever shard this rank happened to see -- see `_gather`.
        for k in self.test_pred.keys():
            self.test_pred[k] = self._gather(torch.cat(self.test_pred[k], 0)).cpu()

        if self.test_explanation:
            # Order doesn't matter here -- `plt_explanations_*` only ever consume a mean
            # over samples -- so no reordering beyond what `_gather` already does.
            self.test_explanation = [
                self._gather(torch.cat([exp[k] for exp in self.test_explanation], dim=0)).cpu()
                for k in range(len(self.test_explanation[0]))
            ]
        else:
            self.test_explanation = None

        # Get masked predictions
        obs = self.test_pred["obs"].squeeze()

        pred_masked = applyMask(self.test_pred["pred"][:, 0, :], self.test_pred["mask"])

        # Compute R2 score
        self.test_r2_score = ExplainedVariance()(pred_masked, obs)
        self.log("test/r2_score", self.test_r2_score, on_step=False, on_epoch=True, prog_bar=True)

        # Compute spearman correlation coeff
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

    # ---------------------------------------------------------------- test artefacts

    def save_test(self) -> None:
        """Score the test predictions and write the run's artefacts.

        Three files per species per run, and no more: the scored predictions
        (`<species>.nc`), the metrics in machine-readable form (`<species>_metrics.json`,
        for comparing runs without reopening a PDF), and the consolidated report
        (`<species>_report.pdf`). This replaced seven separate JPEGs that carried no
        metrics and no ordering.

        `self.log_dict`/`print_metrics_table` run on every rank -- Lightning expects
        `self.log` called the same number of times on every process, and it's cheap. The
        file writes below run once, on the coordinating rank only: every rank has already
        gathered the identical global `report`/`pred_hourly` by this point
        (`on_test_epoch_end`/`_gather`), so writing from every rank would only race them
        all against the same paths, never add anything.
        """
        datamodule = self.trainer.datamodule
        dataset = datamodule.data_test

        pred_hourly = self.test_pred["pred"][:, 0, :].numpy()
        report = M.evaluate(
            count=dataset.count,
            mask=dataset.mask,
            pred_hourly=pred_hourly,
            species=datamodule.species,
            data_dir=datamodule.data_dir,
        )

        self.log_dict(report.logged("test"))
        print_metrics_table(report, title=f"{datamodule.species} — test metrics")

        if not self.trainer.is_global_zero:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.join(self.output_dir, "_".join(datamodule.species.split(" ")))

        self._write_netcdf(f"{stem}.nc", dataset, report, pred_hourly)

        with open(f"{stem}_metrics.json", "w") as f:
            json.dump(
                {
                    "species": datamodule.species,
                    "scalars": report.scalars,
                    "by_era": report.by_era.to_dict(orient="records"),
                    "per_year": report.per_year.to_dict(orient="records"),
                },
                f,
                indent=2,
                default=float,
            )

        build_report(
            path=f"{stem}_report.pdf",
            report=report,
            phenology=self.phenology,
            mask=dataset.mask,
            pred_hourly=pred_hourly,
            run_info=self._run_info(),
            datamodule=datamodule,
            explanations=self.test_explanation,
        )
        log.info(f"Wrote test report to {stem}_report.pdf")

    def _write_netcdf(
        self, path: str, dataset, report: M.MetricReport, pred_hourly: np.ndarray
    ) -> None:
        """Write the test predictions alongside the untransformed weather they came from.

        One vectorised `.sel` over the test dates rather than one `.sel` per row followed
        by an `xr.concat` of thousands of single-date Datasets: the old form was quadratic
        in the number of test rows (a full test split runs to several thousand) and took
        longer than the test epoch itself.
        """
        dates = xr.DataArray(pd.DatetimeIndex(dataset.count["date"]), dims="date")
        test = self.trainer.datamodule.era5_main.sel(date=dates)

        test = test.assign(
            pred_log_hourly_count=(("date", "time"), pred_hourly),
            mask=(("date", "time"), np.asarray(dataset.mask, dtype=float).T),
            obs_count=("date", report.frame["obs"].to_numpy()),
            pred_count=("date", report.frame["pred"].to_numpy()),
            phen_count=("date", report.frame["phen"].to_numpy()),
        )
        test["time"] = test.time.astype(str)
        test.to_netcdf(path)

    def _run_info(self) -> Dict[str, str]:
        """The provenance block printed on the report's first page.

        Which years were held out, and how many rows each split holds, is the first thing
        anyone comparing two reports needs -- and the first thing lost if it lives only in
        a log file that the PDF does not travel with.
        """
        datamodule = self.trainer.datamodule
        count = datamodule.count
        years = {
            split: sorted(count.loc[count["tvt"] == split, "year"].unique())
            for split in ("train", "val", "test")
        }
        sizes = {split: int((count["tvt"] == split).sum()) for split in years}
        test_rows = count[count["tvt"] == "test"]

        checkpoint = getattr(self.trainer, "ckpt_path", None) or "current weights"
        return {
            "output_dir": str(self.output_dir),
            "checkpoint": str(checkpoint),
            "train": f"{sizes['train']:>6} rows  years {_years_str(years['train'])}",
            "val": f"{sizes['val']:>6} rows  years {_years_str(years['val'])}",
            "test": f"{sizes['test']:>6} rows  years {_years_str(years['test'])}",
            "test zero rows": f"{float((test_rows['count'] == 0).mean()):.1%}",
            "doy range": str(list(datamodule.doy)),
        }

    ### EXPORT PREDICTIONS -------------------
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single forward step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # No loss is computed here: ForecastDataset supplies a dummy zero `count` and a
        # length-1 zero `mask`, so evaluating the criterion would divide by
        # mask.sum() == 0 and yield NaN, on top of costing an extra forward pass.
        _, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily, _ = batch
        count_pred = self.forward(yr, doy, prior_shape, era5_main, era5_hourly, era5_daily)

        self.predict_pred["pred"].append(count_pred)

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        # Concatenate rank-local batches, then gather across DDP processes (a no-op at
        # world_size == 1) so the forecast reflects every date rather than whichever
        # shard this rank happened to see -- see `_gather`.
        for k in self.predict_pred.keys():
            self.predict_pred[k] = self._gather(torch.cat(self.predict_pred[k], 0)).cpu()

        # Runs on every rank up to here (cheap, and every rank now holds the identical
        # gathered forecast); only the coordinating rank writes it, so multiple ranks
        # under `trainer=ddp` don't race each other against the same output paths.
        if self.trainer.is_global_zero:
            self.save_predict()

    def save_predict(self):
        """Write the daily forecast: one NetCDF (consumed by defileViz) and one preview JPEG.

        The filename pattern `<YYYYMMDD>_<Species_Name>.nc` and the variable name
        `pred_log_hourly_count` are a contract with the frontend, which fetches them by
        URL -- see AGENTS.md "Related repo". Do not rename either without coordinating.
        """
        datamodule = self.trainer.datamodule
        predict_dataset = datamodule.data_predict

        # One gather over the forecast dates rather than a per-date `.sel` + `xr.concat`,
        # matching `_write_netcdf`. Only a handful of dates here, but the two paths
        # disagreeing about how the output is built is how they drift apart.
        dates = xr.DataArray(pd.DatetimeIndex(predict_dataset.count["date"]), dims="date")
        predictions = predict_dataset.era5_main.sel(date=dates).assign(
            pred_log_hourly_count=(("date", "time"), self.predict_pred["pred"][:, 0, :].numpy())
        )

        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.join(
            self.output_dir,
            "_".join([datetime.date.today().strftime("%Y%m%d")] + datamodule.species.split(" ")),
        )
        predictions.to_netcdf(f"{stem}.nc")
        plt_predict(predictions, species=datamodule.species, filepath=f"{stem}.jpg")

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
