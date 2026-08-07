import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# Floor added before `log(prior_shape)` (see UNetplus.forward). Not zero: a `prior_shape`
# hour of exactly 0 (deep night) would otherwise make that hour's softmax logit -inf,
# making it architecturally *impossible* to ever predict anything there regardless of
# real evidence -- precisely the rigidity the old hardcoded dawn/dusk zero-mask caused,
# which this project already found and fixed once (see DECISIONS.md -> Model
# architecture). This keeps night heavily discouraged, not forbidden.
PRIOR_LOG_EPS = 1e-6


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
        nb_lag_day,
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

        # No trailing Sigmoid: this now produces raw logits `z`, combined with the
        # phenology shape prior's log-probability before a softmax (see forward()) --
        # out_h is a genuine distribution over the 24 hours (sums to 1) now, not 24
        # independent (0, 1) values as before.
        self.conv_final = nn.Sequential(
            nn.Conv1d(outs, 4, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(
                4, nb_output_features, kernel_size=5, stride=1, padding=2, dilation=1
            ),
        )
        # Near-zero (not exactly zero) initialized, so the network starts training at
        # out_h ~= prior_shape -- see DECISIONS.md -> Model architecture for why
        # anchoring the default this way, rather than an arbitrary initial output, is
        # the actual fix for out_h's flat-collapse failure mode: there is no longer a
        # nearby degenerate state for training to fall into when it has little else to
        # go on.
        #
        # Deliberately not exactly zero: for a linear/conv layer, d(output)/d(input) = W,
        # so W == 0 exactly would make the gradient reaching every upstream layer (the
        # whole encoder/decoder before this one) exactly zero too, on the very first
        # step -- the layer's own weights still get a valid gradient and move off zero
        # after step 1, so this "cold start" is transient, but it measurably matters
        # here: with this repo's early-stopping patience of 3 (configs/callbacks/
        # early_stopping.yaml), a training run can plateau and stop before the network
        # has warmed back up, leaving out_d stuck near 0 and every prediction capped far
        # below its true ceiling (confirmed directly: max predicted count across an
        # entire test set was ~3.4 birds/hr against true single-day counts over 1000,
        # with `out = 8 * out_h * out_d` capable of reaching 8). A small random init
        # keeps out_h close to prior_shape at start while letting gradient reach the
        # rest of the network immediately.
        nn.init.normal_(self.conv_final[-1].weight, std=1e-3)
        nn.init.zeros_(self.conv_final[-1].bias)

        # Daily Network --------------------------
        self.nb_lag_day = nb_lag_day
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

        # No hard hour mask here, deliberately. This used to force the output to zero at
        # UTC hours 0-4 and 19-23, ahead of and independent of the real per-sample survey
        # coverage mask that the loss applies. Real surveys start before 05:00 UTC on 145
        # of 4,900 days (3%, concentrated in July-August dawn starts under CEST): on those
        # days the loss's mask correctly said "score this hour" while the network was
        # architecturally forced to emit zero there, biasing the prediction down on exactly
        # the days where the early hours carry birds. `applyMask` in
        # src/models/criterion.py already restricts the loss to the hours each survey
        # actually covered, per sample, which is both correct and sufficient.
        #
        # Dropping that hard mask left deep night (which no survey has ever covered) with
        # no gradient at all, and out_h free to settle anywhere -- including a flat,
        # input-invariant collapse observed directly on some seeds after retraining (see
        # DECISIONS.md -> Model architecture). Fixed not by reintroducing a hard mask, but
        # by anchoring out_h's *default* to a smooth, physically-informed shape (the
        # `prior_shape` argument below) that real weather evidence can still override --
        # see forward()'s softmax step.

    def forward(self, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily):
        # Define forward pass
        # ---------------------------

        # Hourly weather
        doy_ = doy.repeat(1, 24).unsqueeze(1)
        yr_ = yr.repeat(1, 24).unsqueeze(1)
        era5_hourly = rearrange(era5_hourly, "b f t x -> b (f x) t")
        # squeeze(-1) drops the single-location axis only. A bare squeeze() would also
        # drop the batch axis whenever the batch holds a single sample (e.g. a trailing
        # batch of size 1), corrupting the concatenation below.
        era5_main = era5_main.squeeze(-1)
        out_h = torch.cat([era5_main, era5_hourly, doy_, yr_], 1)

        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            out_h, before_pool = module(out_h)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            out_h = module(before_pool, out_h)
        z = self.conv_final(out_h)  # (batch, nb_output_features, 24) raw logits

        # out_h is now a genuine distribution over the 24 hours (sums to 1): the
        # network's logits `z`, biased by the phenology prior's log-probability. At
        # init (z == 0 everywhere, by the zero-init above) this is exactly `prior_shape`;
        # as training proceeds, `z` is a learned residual that shifts probability mass
        # away from the prior wherever the hourly weather inputs actually justify it.
        # `out_d` (below) is a single scalar shared by every hour of one sample, so it
        # cancels out of this softmax entirely -- this supervises shape only.
        log_prior = torch.log(prior_shape + PRIOR_LOG_EPS).unsqueeze(1)
        out_h = torch.softmax(z + log_prior, dim=-1)

        # Daily weather
        doy_ = doy.repeat(1, self.nb_lag_day).unsqueeze(1)
        yr_ = yr.repeat(1, self.nb_lag_day).unsqueeze(1)
        era5_daily = rearrange(era5_daily, "b f t x -> b (f x) t")
        X_d = torch.cat([era5_daily, doy_, yr_], 1)

        out_d = torch.mean(self.layers_d(X_d), dim=2)
        out_d = self.last_layer_d(out_d).unsqueeze(1)

        # out_h now sums to 1 across the 24 hours (a genuine shape distribution, not 24
        # independent (0, 1) values); out_d is between 0 and 1. The `8x` scale (giving a
        # max reachable count of exp(8)-1 = 2979) was tuned against the old
        # unconstrained-per-hour out_h and will settle at a different effective value
        # once retrained against this parameterisation -- expected, not a bug, checked
        # post-retrain like any architecture change here.
        out = 8 * out_h * out_d
        # out = out_h + out_d

        return out  # (batch, 1, 24)
