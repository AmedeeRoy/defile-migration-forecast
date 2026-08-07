"""Tests for `DefileLitModule._apply_trend_correction` (src/models/defile_module.py).

The trend multiplier lives on the raw birds/hr scale but the model's prediction array is
log1p(birds/hr) -- these tests check that conversion is actually applied (a bug here would
silently under- or over-scale every published forecast), that the flag disables it
cleanly, and that a missing trend file falls back to "no correction" rather than raising
(the forecast job must not fail just because a species has no trend file yet).
"""

import logging

import numpy as np
import pytest
from lightning import LightningModule

from src.models.defile_module import DefileLitModule


class _FakeDataModule:
    def __init__(self, species="Test Species", data_dir="/nonexistent"):
        self.species = species
        self.data_dir = data_dir


class _FakeTrainer:
    def __init__(self, datamodule):
        self.datamodule = datamodule


def make_module(apply_trend_correction=True, species="Test Species", data_dir="/nonexistent"):
    """A bare `DefileLitModule` with just enough state for `_apply_trend_correction`.

    Bypasses `__init__` (which needs a real net/optimizer/criterion) since only the
    trend-correction path is under test here.
    """
    module = object.__new__(DefileLitModule)
    # `DefileLitModule.__init__` needs a real net/optimizer/criterion, which this test
    # doesn't have -- but `LightningModule.__init__` alone sets up everything
    # `_apply_trend_correction`/`trend` actually touch: `_parameters`/`_modules` (from
    # `nn.Module`) plus `_trainer`/`_jit_is_scripting`/`_fabric` that the `trainer`
    # property getter/setter dereference.
    LightningModule.__init__(module)
    module.apply_trend_correction = apply_trend_correction
    module._trend = None
    module.trainer = _FakeTrainer(_FakeDataModule(species, data_dir))
    return module


def test_disabled_flag_returns_input_unchanged():
    module = make_module(apply_trend_correction=False)
    pred = np.array([[0.1, 0.2, 0.3]])
    out = module._apply_trend_correction(pred, np.array(["2026-09-01"], dtype="datetime64[D]"))
    assert np.array_equal(out, pred)


def test_missing_trend_file_falls_back_to_unchanged(caplog):
    module = make_module(apply_trend_correction=True, data_dir="/nonexistent")
    pred = np.array([[0.1, 0.2, 0.3]])
    with caplog.at_level(logging.WARNING):
        out = module._apply_trend_correction(pred, np.array(["2026-09-01"], dtype="datetime64[D]"))
    assert np.array_equal(out, pred)
    assert "Skipping trend correction" in caplog.text


def test_multiplier_is_applied_on_the_birds_per_hour_scale(tmp_path, monkeypatch):
    """A multiplier of 2x should double birds/hr, not double log1p(birds/hr)."""
    data_dir = tmp_path / "data"
    (data_dir / "count").mkdir(parents=True)

    species = "Test Species"
    # Fabricate a Trend whose multiplier is exactly 2.0 for 2026 by construction: level -
    # baseline_log_rate = log(2), trend = 0, so forecast_log_rate - baseline == log(2).
    import json

    record = {
        "species": species,
        "year": [2024, 2025],
        "log_rate": [0.0, 0.0],
        "level_path": [0.0, 0.0],
        "baseline_log_rate": 0.0,
        "level": float(np.log(2.0)),
        "trend": 0.0,
        "alpha": 0.4,
        "beta": 0.15,
        "phi": 0.7,
        "last_year": 2025,
    }
    with open(data_dir / "count" / "species_year_statistics.json", "w") as f:
        json.dump([record], f)

    module = make_module(apply_trend_correction=True, species=species, data_dir=str(data_dir))
    assert module.trend.multiplier(2026) == pytest.approx(2.0)

    birds_per_hr = np.array([[1.0, 5.0, 10.0]])
    pred_log = np.log1p(birds_per_hr)
    dates = np.array(["2026-09-01"] * pred_log.shape[0], dtype="datetime64[D]")

    out_log = module._apply_trend_correction(pred_log, dates)
    out_birds_per_hr = np.expm1(out_log)

    assert out_birds_per_hr == pytest.approx(2.0 * birds_per_hr, rel=1e-6)
