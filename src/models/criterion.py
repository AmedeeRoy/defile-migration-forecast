import torch
from dataclasses import dataclass
import numpy as np

# from pytorch_forecasting.metrics import TweedieLoss

def applyMask(y_pred, mask, return_hourly=True):
    # Check if y_pred and mask are xarray DataArrays
    if isinstance(y_pred, np.ndarray):
        # Convert log(bird)/hr to bird/hr for DataArray (using xarray methods)
        y_pred_count = np.expm1(y_pred)

        # Apply the mask (sum over hours where the mask is 1)
        y_masked = np.sum(y_pred_count * mask, axis=1)

        # Normalize by summing the mask and dividing if requested
        if return_hourly:
            y_masked = y_masked / np.sum(mask, axis=1)

        return y_masked

    # Check if y_pred and mask are torch tensors
    elif isinstance(y_pred, torch.Tensor):
        # Convert log(bird)/hr to bird/hr for torch tensor
        y_pred_count = torch.expm1(y_pred)

        # Apply the mask (sum over hours where the mask is 1)
        y_masked = torch.sum(y_pred_count.squeeze() * mask, dim=1)

        # Normalize by summing the mask and dividing if requested
        if return_hourly:
            y_masked = y_masked / torch.sum(mask, dim=1)

        return y_masked

    else:
        raise TypeError(
            "Unsupported type for y_pred or mask. Must be torch.Tensor or xarray.DataArray."
        )



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
