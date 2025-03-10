import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class DownConv(nn.Module):
    """A helper Module that performs 2 convolutions and 1 MaxPool.

    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                self.out_channels,
                self.out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
            ),
            nn.ReLU(),
        )

        if self.pooling:
            self.pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """A helper Module that performs 2 convolutions and 1 UpConvolution.

    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=6, stride=2, padding=2, dilation=1
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(2 * out_channels),
            nn.Conv1d(
                2 * out_channels,
                out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
            ),
            nn.ReLU(),
        )

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetplus(nn.Module):
    def __init__(
        self,
        nb_input_features_hourly,
        nb_hidden_features_hourly,
        nb_layer_hourly,
        nb_hidden_features_daily,
        nb_input_features_daily,
        nb_layer_daily,
        nb_output_features: int = 1,
        dropout: bool = True,
    ):
        super(UNetplus, self).__init__()

        self.nb_input_features_hourly = nb_input_features_hourly
        self.nb_hidden_features_hourly = nb_hidden_features_hourly
        self.nb_layer_hourly = nb_layer_hourly

        # Hourly Network --------------------------
        # create the encoder pathway and add to a list
        self.down_convs = nn.ModuleList()
        for i in range(nb_layer_hourly):
            ins = self.nb_input_features_hourly if i == 0 else outs
            outs = self.nb_hidden_features_hourly * (2**i)
            pooling = True if i < nb_layer_hourly - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        self.up_convs = nn.ModuleList()
        for i in range(nb_layer_hourly - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = nn.Sequential(
            nn.Conv1d(outs, 4, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(
                4, nb_output_features, kernel_size=5, stride=1, padding=2, dilation=1
            ),
            nn.Sigmoid(),  # force output between 0-1
            # nn.ReLU(),  # force output >= 0
        )

        # Daily Network --------------------------
        self.nb_input_features_daily = nb_input_features_daily
        self.nb_layer_daily = nb_layer_daily
        self.nb_hidden_features_daily = nb_hidden_features_daily
        layers_d = []
        for n in range(nb_layer_daily):
            if n == 0:
                layers_d.append(
                    nn.Conv1d(
                        in_channels=nb_input_features_daily,
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
            layers_d.append(nn.ReLU())
            layers_d.append(nn.BatchNorm1d(num_features=nb_hidden_features_daily))
            if dropout:
                layers_d.append(nn.Dropout(0.3))

        self.layers_d = nn.Sequential(*layers_d)

        self.last_layer_d = nn.Sequential(
            nn.Linear(nb_hidden_features_daily, 1),
            nn.Sigmoid(),  # force output between 0-1
            # nn.ReLU(),  # force output >= 0
        )

    def forward(self, yr, doy, era5_main, era5_hourly, era5_daily):
        # Define forward pass
        # ---------------------------

        # Hourly weather
        doy_ = doy.repeat(1, 24).unsqueeze(1)
        yr_ = yr.repeat(1, 24).unsqueeze(1)
        era5_hourly = rearrange(era5_hourly, "b f t x -> b (f x) t")
        era5_main = era5_main.squeeze()
        out_h = torch.cat([era5_main, era5_hourly, doy_, yr_], 1)

        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            out_h, before_pool = module(out_h)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            out_h = module(before_pool, out_h)
        out_h = self.conv_final(out_h)

        # Daily weather
        doy_ = doy.repeat(1, 7).unsqueeze(1)
        yr_ = yr.repeat(1, 7).unsqueeze(1)
        era5_daily = rearrange(era5_daily, "b f t x -> b (f x) t")
        X_d = torch.cat([era5_daily, doy_, yr_], 1)

        out_d = torch.mean(self.layers_d(X_d), dim=2)
        out_d = self.last_layer_d(out_d).unsqueeze(1)

        out = 8 * out_h * out_d  # exp(8)-1 = 2979
        # out = out_h + out_d

        # Force count to be zero during the hours of day with no data
        pred_mask = np.array([1 for i in range(24)])
        pred_mask[:5] = 0
        pred_mask[19:] = 0
        pred_mask = torch.FloatTensor(pred_mask).repeat(out.shape[0], 1).unsqueeze(1)
        out = out * pred_mask.to(out.device)

        return out  # (batch, 1, 24)
