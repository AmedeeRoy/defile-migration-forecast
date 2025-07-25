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
class TweedieLoss:
    """
    Tweedie Loss Function for Count Data Regression.

    The Tweedie distribution is a member of the exponential dispersion family that
    generalizes several well-known distributions (Poisson, Gamma, etc.). It's particularly
    useful for modeling count data with overdispersion, making it suitable for ecological
    data such as bird migration counts.

    The Tweedie loss is defined as:
    L = mean(μ^(2-p)/(2-p) - y*μ^(1-p)/(1-p))

    where μ is the predicted mean and p is the power parameter that determines the
    distribution characteristics:
    - p = 1: Poisson distribution (discrete, constant variance)
    - 1 < p < 2: Compound Poisson distribution (discrete with zeros, increasing variance)
    - p = 2: Gamma distribution (continuous, quadratic variance)

    Attributes:
        alpha (float): Scaling factor for the loss. Default is 1.
        p (float): Power parameter of the Tweedie distribution. Default is 1.5.
                  Controls the variance-mean relationship: Var(Y) = φ * μ^p
    """

    alpha: 1
    p: 1.5

    def forward(self, y_pred, y, mask):
        """
        Compute the Tweedie loss.

        Args:
            y_pred (torch.Tensor): Predicted values with shape [batch_size, channels, time_steps].
                                  Only the first channel (index 0) is used for mean predictions.
                                  Values are expected to be in log-space and will be transformed
                                  to count space using the applyMask function.
            y (torch.Tensor): Ground truth target values with shape [batch_size, 1] or [batch_size].
                             Expected to be average hourly counts during the survey period.
            mask (torch.Tensor): Binary mask with shape [batch_size, time_steps].
                                Indicates which time steps to include in the calculation.

        Returns:
            torch.Tensor: Scalar Tweedie loss value scaled by alpha.

        Notes:
            - The function uses applyMask to convert log-space predictions to average hourly counts
            - Epsilon is added to prevent numerical issues with power operations
            - The loss implements the deviance form of the Tweedie loss function
            - Higher values of p increase the penalty for larger prediction errors
        """
        # Compute expected average hourly count during the survey period
        y_masked = applyMask(
            y_pred[:, 0, :], mask, return_hourly=True
        )  # Average hourly count

        epsilon = 1e-8  # Avoid division errors and numerical instability

        # Tweedie loss components:
        # a: -y * μ^(1-p) / (1-p) term (likelihood component)
        a = y.squeeze() * torch.pow(y_masked + epsilon, 1 - self.p) / (1 - self.p)

        # b: μ^(2-p) / (2-p) term (normalization component)
        b = torch.pow(y_masked + epsilon, 2 - self.p) / (2 - self.p)

        # Tweedie deviance: E[b - a]
        loss = torch.mean(b - a)

        return self.alpha * loss


@dataclass
class ProbaRMSE:
    """
    Probabilistic Root Mean Square Error Loss Function.

    This loss function combines a standard RMSE term with a negative log-likelihood term
    for uncertainty quantification. It's designed for models that predict both mean and
    standard deviation (uncertainty) values.

    The loss consists of two components:
    1. A weighted RMSE term between log-transformed predictions and targets
    2. A negative log-likelihood term that penalizes poor uncertainty estimates

    Attributes:
        alpha (float): Weighting factor for the combined loss. Default is 1.
                      When alpha=1, only RMSE is used.
                      When alpha=0, only negative log-likelihood is used.
                      Values between 0 and 1 combine both terms.
    """

    alpha: 1

    def forward(self, y_pred, y, mask):
        """
        Compute the probabilistic RMSE loss.

        Args:
            y_pred (torch.Tensor): Predicted values with shape [batch_size, 2, time_steps].
                                  Channel 0 contains mean predictions (log-transformed).
                                  Channel 1 contains standard deviation predictions.
            y (torch.Tensor): Ground truth target values with shape [batch_size, 1] or [batch_size].
                             Expected to be in original scale (not log-transformed).
            mask (torch.Tensor): Binary mask with shape [batch_size, time_steps].
                                Indicates which time steps to include in the calculation.

        Returns:
            torch.Tensor: Scalar loss value combining RMSE and negative log-likelihood terms.

        Notes:
            - The function applies log1p transformation to targets for numerical stability
            - Standard deviation is scaled by dividing by 4 (empirical scaling factor)
            - Epsilon is added to prevent division by zero and log(0) issues
            - The final loss is scaled by alpha, making it effectively alpha² * (weighted combination)
        """
        epsilon = 1e-8
        pi = torch.acos(torch.zeros(1)).item() * 2

        # Compute masked mean predictions (log-transformed)
        y_masked = torch.mean(y_pred[:, 0, :] * mask, dim=1)

        # Compute masked standard deviation with scaling and epsilon for stability
        y_std_masked = epsilon + torch.mean(y_pred[:, 1, :] * mask, dim=1) / 4

        # RMSE term: squared difference between log-transformed target and prediction
        rmse_term = torch.mean((torch.log1p(y.squeeze()) - y_masked) ** 2)

        # Negative log-likelihood term: penalizes poor uncertainty estimates
        nll_term = torch.mean(
            (
                (torch.log1p(y.squeeze()) - y_masked) ** 2 / (2 * y_std_masked)
                + torch.log(2 * pi * y_std_masked) / 2
            )
        )

        # Combine terms with alpha weighting
        loss = self.alpha * rmse_term + (1 - self.alpha) * nll_term

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


@dataclass
class RMSE:
    """Home-made Loss function criterion."""

    alpha: 1

    def forward(self, y_pred, y, mask):
        y_masked = applyMask(y_pred, mask)
        loss = torch.mean((y_masked - y.squeeze()) ** 2)
        return self.alpha * loss
