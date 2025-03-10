from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

# from pytorch_forecasting.metrics import TweedieLoss


def applyMask(y_pred, mask, return_hourly=True):
    # y_pred is in log(bird)/hr: The model is build to predict value in the range of log(bird)/hr for each hour (0-8) -> bird/hr -> 0-1200

    # Convert to bird per hours
    y_pred_count = torch.expm1(y_pred)

    # Apply the mask by summing for all hours when observation was perform: y_masked is total number of bird observder for the period
    y_masked = torch.sum(y_pred_count.squeeze() * mask, dim=1)

    # Normalized y_mask to get the average hourly rate of bird
    if return_hourly:
        y_masked = y_masked / torch.sum(mask, dim=1)

    return y_masked


@dataclass
class RMSE:
    """Home-made Loss function criterion."""

    alpha: 1

    def forward(self, y_pred, y, mask):
        y_masked = applyMask(y_pred, mask)
        loss = torch.mean((y_masked - y.squeeze()) ** 2)
        return self.alpha * loss


@dataclass
class TweedieLoss:
    """Home-made Loss function criterion."""

    alpha: 1
    p: 1.5

    def forward(self, y_pred, y, mask):

        # Compute expected average hourly count during the survey period
        y_masked = applyMask(y_pred, mask, return_hourly=True)  # Average hourly count

        epsilon = 1e-8

        # a = y.squeeze() * torch.exp(y_masked * (1 - self.p)) / (1 - self.p)
        # b = torch.exp(y_masked * (2 - self.p)) / (2 - self.p)

        epsilon = 1e-8  # Avoid division errors
        a = y.squeeze() * torch.pow(y_masked + epsilon, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_masked + epsilon, 2 - self.p) / (2 - self.p)

        loss = torch.mean(b - a)

        return self.alpha * loss


@dataclass
class L2:
    alpha: 1

    def forward(self, y_pred, y, mask):
        loss = torch.mean((torch.sum(y_pred.squeeze(), dim=1)) ** 2)
        return self.alpha * loss


@dataclass
class DiffL2:
    alpha: 1

    def forward(self, y_pred, y, mask):
        loss = torch.mean(torch.diff(y_pred, 1) ** 2)
        return self.alpha * loss


@dataclass
class Poisson:
    alpha: float = 1.0  # setting alpha as a float to be flexible

    def forward(self, y_pred, y, mask):

        # Compute expected average hourly count during the survey period based on the predicted hourly log rate (y_pred)
        y_masked = applyMask(y_pred, mask, return_hourly=True)  # Average hourly count
        # y_masked_trans = torch.logm1(y_masked)  # log(bird/hr)

        epsilon = 1e-8  # Small value to avoid log(0)
        loss = torch.mean(y_masked - y.squeeze() * torch.log(y_masked + epsilon))

        return self.alpha * loss
