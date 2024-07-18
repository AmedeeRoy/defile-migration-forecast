import numpy as np
import torch
import torch.nn as nn


class UNetplus(nn.Module):
    def __init__(self, nb_input_features, nb_hidden_features_daily, nb_layer_daily, dropout):
        super(UNetplus, self).__init__()

        self.nb_input_features = nb_input_features

        # Hourly Network --------------------------
        self.cnn_input_1 = nn.Sequential(
            nn.BatchNorm1d(self.nb_input_features),
            nn.Conv1d(self.nb_input_features, 8, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
        )

        self.pooling_1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2, dilation=1)
        )

        self.cnn_input_2 = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
        )

        self.pooling_2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2, dilation=1)
        )

        self.cnn_input_3 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
        )

        # create the decoder pathway and add to a list
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=2, padding=2, dilation=1)
        )

        self.cnn_output_2 = nn.Sequential(
            nn.BatchNorm1d(2 * 16),
            nn.Conv1d(2 * 16, 16, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
        )

        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=6, stride=2, padding=2, dilation=1)
        )

        self.cnn_output_1 = nn.Sequential(
            nn.BatchNorm1d(2 * 8),
            nn.Conv1d(2 * 8, 8, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(2, 1, kernel_size=5, stride=1, padding=2, dilation=1),
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

        # Hourly UNet
        doy_ = doy.repeat(1, 24).unsqueeze(1)
        yr_ = yr.repeat(1, 24).unsqueeze(1)
        X_h = torch.cat([era5_hourly, doy_, yr_], 1)

        out_1 = self.cnn_input_1(X_h)
        out = self.pooling_1(out_1)
        out_2 = self.cnn_input_2(out)
        out = self.pooling_2(out_2)
        out = self.cnn_input_3(out)

        out = self.upconv_2(out)
        out = torch.cat((out, out_2), 1)
        out = self.cnn_output_2(out)

        out = self.upconv_1(out)
        out = torch.cat((out, out_1), 1)
        out_h = self.cnn_output_1(out)

        # Daily CNNet
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
