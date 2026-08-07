"""Tests for the `prior_shape` plumbing in src/data/defile_datamodule.py -- see
DECISIONS.md -> Model architecture. Previously only exercised by live smoke training
runs; these pin down the two things that actually broke during that process (a missing
tuple slot, and a symlink-shaped assumption about `data/weather` -- neither of which a
live run always catches quickly).
"""

import numpy as np
import pandas as pd

from src.data.defile_datamodule import DefileDataset, ForecastDataset, _prior_shape_lookup


def test_prior_shape_lookup_shape_and_normalisation():
    lookup = _prior_shape_lookup("data", "Common Buzzard")
    assert lookup.shape == (366, 24)
    assert lookup.dtype == np.float32
    # Every day's shape sums to 1 (Phenology.hourly_shape's own contract) -- the
    # architecture no longer requires this of prior_shape, but it's still what the
    # committed phenology file provides, and prior_shape_to_logit_bias only cares about
    # relative values, so this should hold unchanged.
    assert np.allclose(lookup.sum(axis=1), 1.0, atol=1e-4)


def test_defile_dataset_getitem_includes_prior_shape_at_the_right_position():
    """DefileDataset.__getitem__'s tuple order is a contract with UNetplus.forward and
    DefileLitModule's batch-unpacking -- both index into it positionally, not by name."""
    count = pd.DataFrame(
        {
            "count": [1.0],
            "year_used_trans": [0.5],
            "doy_trans": [0.1],
            "doy": [200],
        }
    )
    ds = object.__new__(DefileDataset)
    ds.count = count
    ds.mask = np.zeros((24, 1))
    ds.prior_shape_lookup = _prior_shape_lookup("data", "Common Buzzard")
    ds.return_original = False
    ds._main_trans_arr = np.zeros((3, 1))
    ds._hourly_trans_arr = np.zeros((3, 1))
    ds._daily_trans_arr = np.zeros((3, 1))
    ds._main_idx = np.array([0])
    ds._hourly_idx = np.array([0])
    ds._daily_idx = np.array([0])

    sample = ds.__getitem__(0)

    # (count, year_used_trans, doy_trans, prior_shape, era5_main, era5_hourly, era5_daily, mask)
    assert len(sample) == 8
    prior_shape_tensor = sample[3]
    assert prior_shape_tensor.shape == (24,)
    expected = ds.prior_shape_lookup[200 - 1]
    assert np.allclose(prior_shape_tensor.numpy(), expected)


def test_forecast_dataset_getitem_matches_predict_step_unpacking():
    """The non-return_original path is padded with zero placeholders to match
    DefileDataset's tuple length -- see DefileLitModule.predict_step, which unpacks 8
    names regardless of dataset."""
    ds = object.__new__(ForecastDataset)
    ds.count = pd.DataFrame({"date": [pd.Timestamp("2020-07-20")], "doy": [202]})
    ds.prior_shape_lookup = _prior_shape_lookup("data", "Common Buzzard")
    ds.return_original = False
    ds.era5_main_trans = _FakeXr()
    ds.era5_hourly_trans = _FakeXr()
    ds.era5_daily_trans = _FakeXr()
    ds.count["year_used_trans"] = [0.5]
    ds.count["doy_trans"] = [0.1]

    sample = ds.__getitem__(0)

    # matches `_, yr, doy, prior_shape, era5_main, era5_hourly, era5_daily, _ = batch`
    assert len(sample) == 8


class _FakeXr:
    """Minimal stand-in for an xarray Dataset's `.sel(date=...)`, just enough for
    `sample2tensor`'s `hasattr(s, "to_array")` branch to see something array-like."""

    def sel(self, date):
        return self

    def to_array(self):
        return self

    @property
    def values(self):
        return np.zeros(3)
