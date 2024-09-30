from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

# from pytorch_forecasting.metrics import TweedieLoss


@dataclass
class RMSE:
    """Home-made Loss function criterion."""

    alpha: 1
    weights: False

    def forward(self, y_pred, y, mask):
        y_pred_start_to_end = torch.sum(y_pred.squeeze() * mask, dim=1)

        if self.weights:
            # Compute a weight for each hour based on which hour of day it is
            # w_hour = ( (torch.arange(24) - 12 )**2+1 )
            # np.sum(mask, axis = 1)
            sum_mask = torch.Tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.58,
                    48,
                    740,
                    1777,
                    3334,
                    4086,
                    3878,
                    3719,
                    3543,
                    3363,
                    2995,
                    2378,
                    1440,
                    585,
                    160,
                    17,
                    0,
                    0,
                    0,
                ]
            )
            w_hour = 1 / (1 + sum_mask)
            w_hour[6:21] = w_hour[6:21] / sum(w_hour[6:21])
            w_count = torch.sum(w_hour * mask, dim=1)
        else:
            w_count = 1

        loss = torch.mean((y_pred_start_to_end - y.squeeze()) ** 2 * w_count)
        return self.alpha * loss


@dataclass
class TweedieLoss:
    """Home-made Loss function criterion."""

    alpha: 1
    p: 1.5

    def forward(self, y_pred, y, mask):
        y_hat = torch.sum(y_pred.squeeze() * mask, dim=1)
        a = y.squeeze() * torch.exp(y_hat * (1 - self.p)) / (1 - self.p)
        b = torch.exp(y_hat * (2 - self.p)) / (2 - self.p)
        loss = torch.mean(-a + b)
        return self.alpha * loss


@dataclass
class L2:
    alpha: 1

    def forward(self, y_pred, y, mask):
        loss = torch.mean(y_pred)
        return self.alpha * loss


@dataclass
class DiffL2:
    alpha: 1

    def forward(self, y_pred, y, mask):
        loss = torch.mean(torch.diff(y_pred, 1) ** 2)
        return self.alpha * loss


@dataclass
class Gaussian:
    alpha: 1

    def forward(self, y_pred, y, mask):
        y_pred_start_to_end = torch.sum(y_pred * mask.unsqueeze(1), dim=2)
        loss = torch.mean(
            ((y_pred_start_to_end[:, 0] - y.squeeze()) ** 2) / (1e-5 + y_pred_start_to_end[:, 1])
        )
        return self.alpha * loss
