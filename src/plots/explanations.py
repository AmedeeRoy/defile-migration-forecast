"""Saliency (Captum) attribution panels.

Axes-level, like `src/plots/panels.py`: the caller owns the figure. The attributions
themselves are computed in `DefileLitModule.test_step` over a capped number of test
batches and only ever consumed as a mean over samples, which is what makes that cap safe.

These panels are the starting point for the location/variable ablation in DEVELOPMENT.md
Phase 2 -- a variable whose mean attribution is indistinguishable from zero across the
test set is a candidate for removal, though attribution is a hint, not the ablation.
"""

import numpy as np
import torch

from src.plots.panels import C_PRED as C_ATTR


def _mean_attribution(tensor: torch.Tensor, dims) -> np.ndarray:
    """Mean absolute-gradient attribution, reduced over everything but the axis of interest."""
    return torch.mean(tensor, dim=dims).numpy()


def draw_explanations_metrics(axes, datamodule, explanations) -> None:
    """Mean attribution per weather variable, one axes per input stack.

    :param axes: Three axes (local hourly, remote hourly, remote daily).
    """
    _, _, _, era5_main, era5_hourly, era5_daily = explanations

    panels = (
        (era5_main, (0, 2, 3), list(datamodule.era5_main.data_vars), "Local hourly (Défilé)"),
        (era5_hourly, (0, 2, 3), list(datamodule.era5_hourly.data_vars), "Remote hourly"),
        (era5_daily, (0, 2, 3), list(datamodule.era5_daily.data_vars), "Remote daily"),
    )

    for ax, (tensor, dims, labels, title) in zip(axes, panels):
        ax.barh(labels, _mean_attribution(tensor, dims), color=C_ATTR)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)


def draw_explanations_locations(axes, datamodule, explanations) -> None:
    """Mean attribution per location, one axes for the hourly stack and one for the daily."""
    _, _, _, _, era5_hourly, era5_daily = explanations

    panels = (
        (era5_hourly, (0, 1, 2), list(datamodule.era5_hourly.location.values), "Hourly locations"),
        (era5_daily, (0, 1, 2), list(datamodule.era5_daily.location.values), "Daily locations"),
    )

    for ax, (tensor, dims, labels, title) in zip(axes, panels):
        ax.barh(labels, _mean_attribution(tensor, dims), color=C_ATTR)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
