import numpy as np
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, nb_input_features, nb_hidden_features, nb_layer):
        super(ConvNet, self).__init__()

        self.nb_input_features = nb_input_features
        self.nb_layer = nb_layer
        self.nb_hidden_features = nb_hidden_features

        # Add layers as per your requirement
        layers = []
        for n in range(nb_layer):
            if n == 0:
                layers.append(
                    nn.Conv1d(
                        in_channels=nb_input_features,
                        out_channels=nb_hidden_features,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                    )
                )
            else:
                layers.append(
                    nn.Conv1d(
                        in_channels=nb_hidden_features,
                        out_channels=nb_hidden_features,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                    )
                )
            layers.append(nn.BatchNorm1d(num_features=nb_hidden_features))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        # Add layers as per your requirement
        self.last_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=nb_hidden_features,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, yr, doy, era5_hourly, era5_daily):
        # Define forward pass

        # ---------------------------
        # Example of a model that only takes doy and era5_hourly
        # and applies distinct layers of 1D convolutions

        # doy repeated to go from (batch, 1) to (batch, 1, 24)
        doy = doy.repeat(1, 24).unsqueeze(1)
        yr = yr.repeat(1, 24).unsqueeze(1)
        # concatenate doy with era_hourly as feature (batch, nfeatures + 2, 24)
        # X = torch.cat([doy, yr], 1)
        X = torch.cat([era5_hourly, doy, yr], 1)

        out = self.layers(X)
        out = self.last_layer(out)
        out = (
            5 * out
        )  # car la dernière couche de sigmoid force 0 < out < 1, making possible to generate count data between 10^0 and 10^5

        # Force count to be zero between 0-? and ?-24 hr
        pred_mask = np.array([1 for i in range(24)])
        pred_mask[:6] = 0
        pred_mask[21:] = 0
        pred_mask = torch.FloatTensor(pred_mask).repeat(out.shape[0], 1).unsqueeze(1)
        out = out * pred_mask

        return out  # (batch, 1, 24)


class ConvNetplus(nn.Module):
    def __init__(
        self,
        nb_input_features,
        nb_hidden_features_hourly,
        nb_layer_hourly,
        nb_hidden_features_daily,
        nb_layer_daily,
        dropout,
    ):
        super(ConvNetplus, self).__init__()

        self.nb_input_features = nb_input_features

        # Hourly Network --------------------------
        self.nb_layer_hourly = nb_layer_hourly
        self.nb_hidden_features_hourly = nb_hidden_features_hourly
        layers_h = []
        for n in range(nb_layer_hourly):
            if n == 0:
                layers_h.append(
                    nn.Conv1d(
                        in_channels=nb_input_features,
                        out_channels=nb_hidden_features_hourly,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                    )
                )
            else:
                layers_h.append(
                    nn.Conv1d(
                        in_channels=nb_hidden_features_hourly,
                        out_channels=nb_hidden_features_hourly,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                    )
                )
            layers_h.append(nn.BatchNorm1d(num_features=nb_hidden_features_hourly))
            layers_h.append(nn.ReLU())
            if dropout:
                layers_h.append(nn.Dropout(0.3))

        self.layers_h = nn.Sequential(*layers_h)

        self.last_layer_h = nn.Sequential(
            nn.Conv1d(
                in_channels=nb_hidden_features_hourly,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.Sigmoid(),
        )

        # Daily Network --------------------------
        self.nb_layer_daily = nb_layer_daily
        self.nb_hidden_features_daily = nb_hidden_features_daily
        layers_d = []
        for n in range(nb_layer_daily):
            if n == 0:
                layers_d.append(
                    nn.Conv1d(
                        in_channels=nb_input_features,
                        out_channels=nb_hidden_features_daily,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                    )
                )
            else:
                layers_d.append(
                    nn.Conv1d(
                        in_channels=nb_hidden_features_daily,
                        out_channels=nb_hidden_features_daily,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                    )
                )
            layers_d.append(nn.BatchNorm1d(num_features=nb_hidden_features_daily))
            layers_d.append(nn.ReLU())
            if dropout:
                layers_d.append(nn.Dropout(0.3))

        self.layers_d = nn.Sequential(*layers_d)

        self.last_layer_d = nn.Sequential(
            nn.Linear(nb_hidden_features_daily, 1),
            nn.Sigmoid(),
        )

    def forward(self, yr, doy, era5_hourly, era5_daily):
        # Define forward pass

        # ---------------------------
        # Example of a model that only takes doy and era5_hourly
        # and applies distinct layers of 1D convolutions
        doy_ = doy.repeat(1, 24).unsqueeze(1)
        yr_ = yr.repeat(1, 24).unsqueeze(1)
        X_h = torch.cat([era5_hourly, doy_, yr_], 1)

        out_h = self.layers_h(X_h)
        out_h = self.last_layer_h(out_h)

        doy_ = doy.repeat(1, 7).unsqueeze(1)
        # dd = torch.arange(-7/366, 0, step = 1/ 366)
        # doy_ = doy_ + dd
        yr_ = yr.repeat(1, 7).unsqueeze(1)
        X_d = torch.cat([era5_daily, doy_, yr_], 1)
        out_d = torch.mean(self.layers_d(X_d), dim=2)
        out_d = self.last_layer_d(out_d).unsqueeze(1)

        out = 5 * out_h * out_d

        # Force count to be zero between 0-? and ?-24 hr
        pred_mask = np.array([1 for i in range(24)])
        pred_mask[:6] = 0
        pred_mask[21:] = 0
        pred_mask = torch.FloatTensor(pred_mask).repeat(out.shape[0], 1).unsqueeze(1)
        out = out * pred_mask

        return out  # (batch, 1, 24)
