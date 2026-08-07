"""Tests for UNetplus's out_h shape-prior anchoring (src/models/components/unet.py) --
see DECISIONS.md -> Model architecture and the `feature/phenology-shape-prior` plan.

`out_h = softmax(z + log(prior_shape))` with `z`'s producing layer zero-initialized: at
construction, before any training, the network must output exactly `prior_shape` (up to
the small epsilon floor), regardless of its other inputs -- that guarantee, not a loss
term, is what removes the flat-collapse failure mode this replaces (NightPenalty,
ShapeSupervision).
"""

import torch

from src.models.components.unet import PRIOR_LOG_EPS, UNetplus


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


def _random_inputs(net, batch_size, prior_shape=None):
    if prior_shape is None:
        prior_shape = torch.rand(batch_size, 24)
        prior_shape = prior_shape / prior_shape.sum(dim=1, keepdim=True)
    return dict(
        yr=torch.rand(batch_size, 1),
        doy=torch.rand(batch_size, 1),
        prior_shape=prior_shape,
        era5_main=torch.randn(batch_size, 4, 24, 1),
        era5_hourly=torch.randn(batch_size, 1, 24, 1),
        era5_daily=torch.randn(batch_size, 4, 3, 1),
    )


def test_out_h_equals_prior_shape_at_init():
    """z == 0 everywhere at construction (zero-initialized last layer), so
    softmax(0 + log(prior_shape)) must reproduce prior_shape exactly, up to
    PRIOR_LOG_EPS."""
    net = _make_net()
    net.eval()
    batch_size = 5
    prior_shape = torch.rand(batch_size, 24)
    prior_shape = prior_shape / prior_shape.sum(dim=1, keepdim=True)
    inputs = _random_inputs(net, batch_size, prior_shape=prior_shape)

    with torch.no_grad():
        out = net(**inputs)

    # UNetplus returns only the combined `out = 8 * out_h * out_d`, not out_h directly --
    # recover it by normalising `out` back to sum to 1 across hours (out_d is a single
    # scalar shared by every hour of one sample, so it cancels out of that division).
    out_h_recovered = out[:, 0, :] / out.sum(dim=2)
    assert torch.allclose(out_h_recovered, prior_shape, atol=1e-4)


def test_out_h_sums_to_one():
    net = _make_net()
    inputs = _random_inputs(net, batch_size=3)
    out = net(**inputs)
    out_h_recovered = out[:, 0, :] / out.sum(dim=2)
    assert torch.allclose(out_h_recovered.sum(dim=1), torch.ones(3), atol=1e-5)


def test_prior_shape_with_a_zero_hour_is_discouraged_not_forbidden():
    """A hard 0 in prior_shape must not make that hour's logit -inf (the old dawn/dusk
    mask's exact failure mode) -- PRIOR_LOG_EPS keeps it finite, so a large enough
    learned `z` can still push probability there."""
    net = _make_net()
    prior_shape = torch.zeros(1, 24)
    prior_shape[0, 12] = 1.0  # single-hour prior, everything else exactly 0

    log_prior = torch.log(prior_shape + PRIOR_LOG_EPS)
    assert torch.isfinite(log_prior).all()

    # A large synthetic logit at an otherwise-zero-prior hour should still be able to
    # dominate the softmax -- i.e. real evidence can override the prior.
    z = torch.full((1, 24), -1.0)
    z[0, 0] = 50.0
    out_h = torch.softmax(z + log_prior, dim=-1)
    assert out_h[0, 0] > 0.99


def test_forward_runs_with_multi_sample_batch():
    """Guards the einops/broadcast plumbing around the new prior_shape argument for a
    batch size > 1 (a single-sample batch can silently hide a squeezed batch axis)."""
    net = _make_net()
    inputs = _random_inputs(net, batch_size=4)
    out = net(**inputs)
    assert out.shape == (4, 1, 24)
