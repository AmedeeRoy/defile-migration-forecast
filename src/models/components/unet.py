import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# Floor used when turning `prior_shape` into a per-hour logit bias (see
# `prior_shape_to_logit_bias`). Not zero: an hour whose prior is exactly 0 (deep night)
# would otherwise get a -inf bias, making it architecturally *impossible* to ever predict
# anything there regardless of real evidence -- precisely the rigidity the old hardcoded
# dawn/dusk zero-mask caused, which this project already found and fixed once (see
# DECISIONS.md -> Model architecture). This keeps night heavily discouraged, not forbidden:
# logit(1e-6) is about -13.8, so a learned logit has to genuinely fight for that hour.
PRIOR_LOG_EPS = 1e-6

# What `out_h` is at its peak hour when the network adds nothing (raw logit z == 0), i.e.
# the prior-only default. 0.5 keeps the initial scale of `out_h` comparable to the old
# free-sigmoid architecture (whose small random init also sat near 0.5), so `out_d`'s
# learned scale and the `8 *` constant in `forward` stay in the range they were tuned for.
PRIOR_PEAK_TARGET = 0.5


def prior_shape_to_logit_bias(prior_shape, peak_target=PRIOR_PEAK_TARGET, eps=PRIOR_LOG_EPS):
    """Turns a normalised 24h shape (sums to 1, from `Phenology.hourly_shape`) into a
    per-hour logit bias for `out_h`'s sigmoid -- `(batch, 24) -> (batch, 24)`.

    Deliberately *not* a softmax over the prior's log-probabilities. That was the first
    attempt and it is wrong for this model: a softmax forces the 24 hours to sum to 1, so
    `out_h` becomes a pure shape and every bit of magnitude has to come from `out_d`, a
    single scalar fed only by *daily*-scale inputs. That is a real capacity cut, not a
    reparameterisation -- the hourly branch can then only redistribute a total it has no
    way to influence, even though hourly weather (a rain band, a wind shift) genuinely
    changes how many birds pass in an hour, not just when. Measured directly: a 3-seed
    sweep under the softmax form held `season_total_ratio` at ~0.14 (vs ~1.34 for the
    unconstrained baseline), unmoved by 2x the epochs or 5x the early-stopping patience.

    The sigmoid form keeps the anchoring without the constraint. The prior is rescaled so
    its peak hour maps to `peak_target` and its zero hours to ~0, then converted to a
    logit; `out_h = sigmoid(z + bias)` is `peak_target` at the peak when `z == 0`, and each
    hour is free in (0, 1) independently -- so `out_h` still carries absolute magnitude,
    exactly as it did before any prior existed.
    """
    peak = prior_shape.amax(dim=-1, keepdim=True).clamp_min(eps)
    target = (prior_shape / peak) * peak_target
    return torch.log(target + eps) - torch.log1p(-target + eps)


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

        # No trailing Sigmoid here: it moved into `forward`, where the phenology prior's
        # logit bias is added first so the sigmoid's default is the prior's shape rather
        # than an arbitrary init. `out_h`'s range and meaning are otherwise unchanged --
        # still 24 independent values in (0, 1) that carry magnitude as well as shape.
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
        # see forward()'s sigmoid step and `prior_shape_to_logit_bias`.

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

        # out_h stays what it always was -- 24 independent values in (0, 1), free to carry
        # absolute magnitude, not just the day's shape -- but its *default* is now the
        # phenology prior instead of an arbitrary init. `z` is the network's learned
        # per-hour logit; the bias term makes `sigmoid(z + bias)` sit at the prior's
        # rescaled shape when `z == 0` (see `prior_shape_to_logit_bias`, including why a
        # softmax here was wrong). Hourly weather can push any hour up or down in absolute
        # terms, so the hourly branch keeps its share of the magnitude prediction and
        # `out_d` is not left carrying all of it alone.
        prior_bias = prior_shape_to_logit_bias(prior_shape).unsqueeze(1)
        out_h = torch.sigmoid(z + prior_bias)

        # Daily weather
        doy_ = doy.repeat(1, self.nb_lag_day).unsqueeze(1)
        yr_ = yr.repeat(1, self.nb_lag_day).unsqueeze(1)
        era5_daily = rearrange(era5_daily, "b f t x -> b (f x) t")
        X_d = torch.cat([era5_daily, doy_, yr_], 1)

        out_d = torch.mean(self.layers_d(X_d), dim=2)
        out_d = self.last_layer_d(out_d).unsqueeze(1)

        # out_h and out_d are both between 0 and 1 -- unchanged from before the prior was
        # introduced, so the `8 *` scale (max reachable count exp(8)-1 = 2979) still means
        # what it did when it was tuned. Both branches contribute magnitude: out_d damps
        # the whole day from daily-scale inputs, out_h sets each hour's level from
        # hourly-scale inputs.
        out = 8 * out_h * out_d
        # out = out_h + out_d

        return out  # (batch, 1, 24)
