from dataclasses import dataclass

import numpy as np
import torch

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
        y_masked = applyMask(y_pred[:, 0, :], mask, return_hourly=True)  # Average hourly count

        epsilon = 1e-8  # Avoid division errors
        a = y.squeeze() * torch.pow(y_masked + epsilon, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_masked + epsilon, 2 - self.p) / (2 - self.p)

        loss = torch.mean(b - a)
        return self.alpha * loss


@dataclass
class ProbaRMSE:
    """Home-made Loss function criterion."""

    alpha: 1

    def forward(self, y_pred, y, mask):
        epsilon = 1e-8
        pi = torch.acos(torch.zeros(1)).item() * 2

        y_masked = torch.mean(y_pred[:, 0, :] * mask, dim=1)
        y_std_masked = epsilon + torch.mean(y_pred[:, 1, :] * mask, dim=1) / 4

        loss = self.alpha * torch.mean((torch.log1p(y.squeeze()) - y_masked) ** 2) + (
            1 - self.alpha
        ) * torch.mean(
            (
                (torch.log1p(y.squeeze()) - y_masked) ** 2 / (2 * y_std_masked)
                + torch.log(2 * pi * y_std_masked) / 2
            )
        )
        return self.alpha * loss


# @dataclass
# class ProbaTweedieLoss:
#     """
#     Home-made Loss function criterion.
#     https://dl.acm.org/doi/pdf/10.1145/3583780.3615215
#     """
#     alpha: 1
#     def forward(self, y_pred, y, mask):

#         epsilon = 1e-8
#         mu = applyMask(y_pred[:,0,:], mask, return_hourly=True)  # Average hourly count
#         phi = torch.mean(y_pred[:,1,:].squeeze() * mask, dim=1) + epsilon
#         rho = torch.mean(y_pred[:,2,:].squeeze() * mask, dim=1)/8 + 1 + epsilon

#         # Loss TD y > 0
#         j_max = torch.pow(y.squeeze() + epsilon, 2 - rho) / (2 - rho) / phi
#         alpha = (2 - rho) / (1 - rho)
#         a = y.squeeze() * torch.pow(mu + epsilon, 1 - rho) / (1 - rho)
#         b = torch.pow(mu + epsilon, 2 - rho) / (2 - rho)
#         loss_1 = (a - b) / phi - torch.log(epsilon + j_max * torch.sqrt(-alpha * y.squeeze())) + j_max * (alpha - 1)

#         # Loss TD X = 0
#         loss_2 = torch.pow(mu, 2- rho)/ (2 - rho) / phi

#         loss = - loss_1 * (y.squeeze() !=0) - loss_2 * (y.squeeze() == 0)

#         print(loss)
#         # print((y !=0) * loss_1)
#         # print(loss_1.shape, loss_2.shape, loss.shape)
#         return self.alpha * loss


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
