"""Tests for the annual trend correction (src/trend.py).

Covers the two things that matter for a forecast-time correction: that the damped
extrapolation actually damps (a multi-year-ahead call converges rather than running away,
the failure mode the rolling-origin backtest found in an unconstrained linear trend -- see
DECISIONS.md), and that the multiplier is centred so a species with no real trend gets a
multiplier of ~1.
"""

import json

import numpy as np
import pytest

from src.trend import Trend

SPECIES = "Test Species"


def write_trend_file(tmp_path, record):
    data_dir = tmp_path / "data"
    (data_dir / "count").mkdir(parents=True)
    with open(data_dir / "count" / "species_year_statistics.json", "w") as f:
        json.dump([record], f)
    return str(data_dir)


def make_record(year, log_rate, level, trend, phi=0.7, alpha=0.4, beta=0.15):
    level_path = list(log_rate)  # not used by any test below except as a lookup table
    return {
        "species": SPECIES,
        "year": [int(y) for y in year],
        "log_rate": [float(v) for v in log_rate],
        "level_path": level_path,
        "baseline_log_rate": float(np.mean(level_path)),
        "level": float(level),
        "trend": float(trend),
        "alpha": alpha,
        "beta": beta,
        "phi": phi,
        "last_year": int(year[-1]),
    }


def test_load_raises_key_error_for_unknown_species(tmp_path):
    data_dir = write_trend_file(tmp_path, make_record(range(2000, 2010), np.zeros(10), 0.0, 0.0))
    with pytest.raises(KeyError, match="Unknown Species"):
        Trend.load(data_dir, "Unknown Species")


def test_one_step_forecast_matches_phi_times_trend(tmp_path):
    record = make_record(range(2000, 2010), np.zeros(10), level=1.0, trend=0.1, phi=0.7)
    trend = Trend.load(write_trend_file(tmp_path, record), SPECIES)
    # h=1: level + phi * trend
    assert trend.forecast_log_rate(2010) == pytest.approx(1.0 + 0.7 * 0.1)


def test_damped_forecast_converges_instead_of_running_away(tmp_path):
    """A multi-year-ahead call must plateau, not scale linearly with the horizon.

    This is the property that made an unconstrained linear-trend extrapolation the worst
    performer (by far) in the rolling-origin backtest: it has no such ceiling.
    """
    record = make_record(range(2000, 2010), np.zeros(10), level=1.0, trend=0.1, phi=0.7)
    trend = Trend.load(write_trend_file(tmp_path, record), SPECIES)

    forecasts = [trend.forecast_log_rate(2010 + h) for h in range(1, 30)]
    # Monotonically increasing (trend is positive)...
    assert all(b >= a for a, b in zip(forecasts, forecasts[1:]))
    # ...but bounded: the total lift converges to trend * phi / (1 - phi), not h * trend.
    ceiling = 1.0 + 0.1 * 0.7 / (1 - 0.7)
    assert forecasts[-1] == pytest.approx(ceiling, abs=1e-4)
    # A naive (undamped) linear extrapolation would have reached level + 29*trend = 3.9 by
    # h=29; the damped forecast must stay far below that.
    assert forecasts[-1] < 1.0 + 5 * 0.1


def test_multiplier_is_one_when_forecast_equals_baseline(tmp_path):
    record = make_record(range(2000, 2010), np.zeros(10), level=1.0, trend=0.0, phi=0.7)
    record["baseline_log_rate"] = 1.0  # forecast_log_rate(2010) == level == baseline
    trend = Trend.load(write_trend_file(tmp_path, record), SPECIES)
    assert trend.multiplier(2010) == pytest.approx(1.0)


def test_multiplier_reflects_a_real_upward_trend(tmp_path):
    """Mirrors the real fit for Red Kite/Kestrel: a genuine multi-year rise should give a
    multiplier well above 1, not be washed out by the damping."""
    years = np.arange(1993, 2026)
    log_rate = np.log(0.5) + 0.05 * (years - years[0])  # steady ~5%/yr growth in log-space
    from scripts.build_trend_stats import damped_trend_fit

    level_path, level, trend_ = damped_trend_fit(log_rate)
    record = make_record(years, log_rate, level, trend_)
    record["level_path"] = level_path.tolist()
    record["baseline_log_rate"] = float(np.mean(level_path))
    trend = Trend.load(write_trend_file(tmp_path, record), SPECIES)
    assert trend.multiplier(2026) > 1.5


def test_multiplier_is_clamped_to_the_sanity_bound(tmp_path, caplog):
    """A species whose fit implies a huge correction (e.g. Red Kite's raw 3.41x) must be
    held at the MAX_LOG_MULTIPLIER bound, not applied unclamped -- see that constant's
    docstring: this covers the real blind spot where a species' entire fitted history moves
    in one direction (no historical test case for what happens when it eventually turns)."""
    import logging

    record = make_record(range(2000, 2010), np.zeros(10), level=10.0, trend=0.0, phi=0.7)
    record["baseline_log_rate"] = 0.0  # forecast - baseline = 10, exp(10) is way above 3x
    trend = Trend.load(write_trend_file(tmp_path, record), SPECIES)
    with caplog.at_level(logging.WARNING):
        mult = trend.multiplier(2010)
    assert mult == pytest.approx(3.0)
    assert "clamped" in caplog.text


def test_forecasting_a_year_within_the_fitted_history_uses_its_own_level(tmp_path):
    year = list(range(2000, 2010))
    level_path = [float(i) for i in range(10)]  # distinct, easy-to-check values
    record = make_record(year, level_path, level=100.0, trend=100.0)
    record["level_path"] = level_path
    trend = Trend.load(write_trend_file(tmp_path, record), SPECIES)
    assert trend.forecast_log_rate(2005) == pytest.approx(5.0)
