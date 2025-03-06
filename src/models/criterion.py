from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

# from pytorch_forecasting.metrics import TweedieLoss


def applyMask(y_pred, mask, return_hourly=False):
    y_masked = torch.sum(y_pred.squeeze() * mask, dim=1)
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
        y_masked = applyMask(y_pred, mask)
        a = y.squeeze() * torch.exp(y_masked * (1 - self.p)) / (1 - self.p)
        b = torch.exp(y_masked * (2 - self.p)) / (2 - self.p)
        loss = torch.mean(-a + b)
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

        # Compute expected total count using predicted hourly log rate (y_pred)
        y_pred_count = torch.expm1(y_pred)
        y_masked = applyMask(y_pred_count, mask)

        epsilon = 1e-8  # Small value to avoid log(0)
        loss = torch.mean(y_masked - y.squeeze() * torch.log(y_masked + epsilon))

        return self.alpha * loss
