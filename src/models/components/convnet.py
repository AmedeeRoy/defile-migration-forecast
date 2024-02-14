import torch
import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, nb_features, nb_layer):
        super(ConvNet, self).__init__()

        self.nb_features = nb_features
        self.nb_layer = nb_layer

        nb_features_in_layer = nb_features

        # Add layers as per your requirement
        layers = []
        for n in range(nb_layer):
            layers.append(nn.Conv1d(in_channels = nb_features_in_layer, out_channels=nb_features_in_layer*2, kernel_size = 5, stride = 1, padding = 2, dilation = 1))
            layers.append(nn.BatchNorm1d(num_features = nb_features_in_layer*2))
            layers.append(nn.ReLU())
            nb_features_in_layer *= 2

        self.layers = nn.Sequential(*layers)

        # Add layers as per your requirement
        self.last_layer = nn.Sequential(
                          nn.Conv1d(in_channels = nb_features_in_layer, out_channels=1, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
                          nn.Sigmoid()
                          )

    def forward(self, yr, doy, era5_hourly):
        # Define forward pass

        # ---------------------------
        # Example of a model that only takes doy and era5_hourly
        # and applies distinct layers of 1D convolutions

        # doy repeated to go from (batch, 1) to (batch, 1, 24)
        doy = doy.repeat(1, 24).unsqueeze(1)
        yr = yr.repeat(1, 24).unsqueeze(1)
        # concatenate doy with era_hourly as feature (batch, nfeatures + 1, 24)
        X = torch.cat([era5_hourly, doy, yr], 1)

        out = self.layers(X)
        out = self.last_layer(out)
        out = 5*out # car la dernière couche de sigmoid force 0 < out < 1, making possible to generate count data between 10^0 and 10^5

        ## !!!!!!!! Usually not good practice to instantiate value in Tensor !!!!!!!!!
        # -> it makes the computation of automatic differentiation impossible
        # -> usually better to multiply by some mask
        # -> Here in practice it does not change anything yet
        # # Force count to be zero between 0-? and ?-24 hr
        # out[:,:,:6] = 0
        # out[:,:,21:] = 0
        # Force count to be zero between 0-? and ?-24 hr
        pred_mask = np.array([1 for i in range(24)])
        pred_mask[:6] = 0
        pred_mask[21:] = 0
        pred_mask = torch.FloatTensor(pred_mask).repeat(out.shape[0], 1).unsqueeze(1)
        out = out * pred_mask

        return out # (batch, 1, 24)