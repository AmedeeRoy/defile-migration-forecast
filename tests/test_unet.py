"""Tests for UNetplus's out_h shape-prior anchoring (src/models/components/unet.py) --
see DECISIONS.md -> Model architecture.

`out_h = sigmoid(z + prior_shape_to_logit_bias(prior_shape))`, with `z`'s producing layer
near-zero-initialized: at construction, before any training, the network outputs the
prior's rescaled shape rather than an arbitrary one -- that guarantee, not a loss term, is
what removes the flat-collapse failure mode this replaces.

Critically, `out_h` is *not* normalised to sum to 1: each hour stays independent in (0, 1)
so the hourly branch keeps its share of the absolute-magnitude prediction. The tests below
pin that down, since the first attempt at this used a softmax and silently cost the model
that capacity (see `prior_shape_to_logit_bias`'s docstring).
"""

import torch

from src.models.components.unet import (
    PRIOR_PEAK_TARGET,
    UNetplus,
    prior_shape_to_logit_bias,
)


def _make_net(nb_input_features_hourly=7, nb_input_features_daily=6):
    return UNetplus(
        nb_input_features_hourly=nb_input_features_hourly,
        nb_hidden_features_hourly=4,
        nb_layer_hourly=2,
        nb_lag_day=3,
        nb_hidden_features_daily=8,
        nb_input_features_daily=nb_input_features_daily,
        nb_layer_daily=2,
        nb_output_features=1,
        dropout=False,
    )


def _normalised_shape(batch_size):
    shape = torch.rand(batch_size, 24)
    return shape / shape.sum(dim=1, keepdim=True)


def _random_inputs(batch_size, prior_shape=None):
    return dict(
        yr=torch.rand(batch_size, 1),
        doy=torch.rand(batch_size, 1),
        prior_shape=_normalised_shape(batch_size) if prior_shape is None else prior_shape,
        era5_main=torch.randn(batch_size, 4, 24, 1),
        era5_hourly=torch.randn(batch_size, 1, 24, 1),
        era5_daily=torch.randn(batch_size, 4, 3, 1),
    )


def test_logit_bias_puts_the_peak_hour_at_the_target_and_zero_hours_near_zero():
    prior_shape = torch.zeros(1, 24)
    prior_shape[0, 10] = 0.6  # peak
    prior_shape[0, 11] = 0.4

    out_h_at_z_zero = torch.sigmoid(prior_shape_to_logit_bias(prior_shape))

    assert torch.isclose(out_h_at_z_zero[0, 10], torch.tensor(PRIOR_PEAK_TARGET), atol=1e-4)
    # The second hour is 0.4/0.6 of the peak, scaled by the same target.
    assert torch.isclose(
        out_h_at_z_zero[0, 11], torch.tensor(PRIOR_PEAK_TARGET * 0.4 / 0.6), atol=1e-4
    )
    assert (out_h_at_z_zero[0, :10] < 1e-4).all()  # prior-zero hours suppressed


def test_out_h_is_not_normalised_so_it_can_carry_magnitude():
    """The whole point of using sigmoid rather than softmax: two samples with the *same*
    prior shape must still be able to differ in overall level, not just in distribution.
    A softmax would force both to sum to 1 and make this impossible."""
    prior_shape = torch.zeros(2, 24)
    prior_shape[:, 8:16] = 1.0 / 8  # identical, normalised prior for both samples
    bias = prior_shape_to_logit_bias(prior_shape)

    quiet_hour_logits = torch.full((2, 24), -3.0)
    busy_hour_logits = torch.full((2, 24), 3.0)

    quiet = torch.sigmoid(quiet_hour_logits + bias)
    busy = torch.sigmoid(busy_hour_logits + bias)

    assert busy.sum() > 3 * quiet.sum()  # magnitude genuinely free to move
    assert not torch.isclose(busy.sum(dim=1), torch.ones(2)).any()


def test_out_h_follows_the_prior_shape_at_init():
    net = _make_net()
    net.eval()
    batch_size = 5
    prior_shape = _normalised_shape(batch_size)
    inputs = _random_inputs(batch_size, prior_shape=prior_shape)

    with torch.no_grad():
        out = net(**inputs)

    # UNetplus returns only the combined `out = 8 * out_h * out_d`. out_d is a single
    # scalar shared by every hour of a sample, so normalising `out` across hours recovers
    # out_h's *shape* (not its level) for comparison against the prior.
    out_shape = out[:, 0, :] / out.sum(dim=2, keepdim=True)[:, 0, :]
    expected = torch.sigmoid(prior_shape_to_logit_bias(prior_shape))
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(out_shape, expected, atol=1e-2)


def test_prior_shape_with_a_zero_hour_is_discouraged_not_forbidden():
    """A hard 0 in prior_shape must not make that hour impossible (the old dawn/dusk
    mask's exact failure mode) -- PRIOR_LOG_EPS keeps the bias finite, so a large enough
    learned logit can still push probability there."""
    prior_shape = torch.zeros(1, 24)
    prior_shape[0, 12] = 1.0

    bias = prior_shape_to_logit_bias(prior_shape)
    assert torch.isfinite(bias).all()

    z = torch.zeros(1, 24)
    z[0, 0] = 50.0  # strong evidence at an hour the prior says is impossible
    out_h = torch.sigmoid(z + bias)
    assert out_h[0, 0] > 0.99


def test_forward_runs_with_multi_sample_batch():
    """Guards the einops/broadcast plumbing around the prior_shape argument for a batch
    size > 1 (a single-sample batch can silently hide a squeezed batch axis)."""
    net = _make_net()
    out = net(**_random_inputs(batch_size=4))
    assert out.shape == (4, 1, 24)
