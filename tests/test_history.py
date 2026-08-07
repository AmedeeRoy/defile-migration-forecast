"""Tests for within-season history features (src/data/history.py).

The failure modes that matter here are all leakage/boundary bugs: a day's history must
never include its own count, must never reach across a year boundary, and a missing
calendar day must become a real zero-coverage row rather than silently letting a window
reach further back in time than it should.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.history import add_history_features, n_history_channels


def make_count(rows):
    """rows: list of (year, date_str, count_raw, duration), one row per observation period."""
    df = pd.DataFrame(rows, columns=["year", "date", "count_raw", "duration"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_noop_when_nothing_requested():
    count = make_count([(2020, "2020-07-15", 10, 5)])
    out = add_history_features(count, windows=(), cumulative=False)
    assert out is count  # literally unchanged, not just equal


def test_lag_window_excludes_the_row_own_day():
    count = make_count(
        [
            (2020, "2020-07-15", 10, 5),
            (2020, "2020-07-16", 20, 5),
            (2020, "2020-07-17", 30, 5),
        ]
    )
    out = add_history_features(count, windows=(1,), cumulative=False)
    # day 3's lag-1 rate must reflect only day 2 (20/5=4), never day 3's own (30, 5)
    day3 = out[out.date == "2020-07-17"].iloc[0]
    assert day3["history_lag1_value"] == pytest.approx(np.log1p(20 / 5))


def test_lag_window_resets_at_year_boundary():
    count = make_count(
        [
            (2020, "2020-12-01", 999, 5),  # last day of 2020 season, huge count
            (2021, "2021-07-15", 10, 5),  # first day of 2021 season
        ]
    )
    out = add_history_features(count, windows=(3,), cumulative=False)
    first_2021 = out[out.date == "2021-07-15"].iloc[0]
    # No prior data exists in 2021 -- must not see 2020's tail.
    assert first_2021["history_lag3_value"] == pytest.approx(0.0)
    assert first_2021["history_lag3_hours"] == pytest.approx(0.0)


def test_missing_calendar_day_becomes_zero_coverage_not_a_skipped_gap():
    """A 3-day window computed the day after a 1-day gap must include that gap as a real
    zero, not silently reach back a 4th day to compensate."""
    count = make_count(
        [
            (2020, "2020-07-15", 10, 5),
            (2020, "2020-07-16", 20, 5),
            # 2020-07-17 missing entirely (no session that day)
            (2020, "2020-07-18", 30, 5),
        ]
    )
    out = add_history_features(count, windows=(3,), cumulative=False)
    day18 = out[out.date == "2020-07-18"].iloc[0]
    # window = days 15,16,17 = 10+20+0 = 30 over 5+5+0=10 hours
    assert day18["history_lag3_value"] == pytest.approx(np.log1p(30 / 10))
    assert day18["history_lag3_hours"] == pytest.approx(np.log1p(10))


def test_zero_hours_window_gives_zero_value_not_nan_or_inf():
    count = make_count([(2020, "2020-07-15", 0, 0), (2020, "2020-07-16", 5, 5)])
    out = add_history_features(count, windows=(1,), cumulative=False)
    day16 = out[out.date == "2020-07-16"].iloc[0]
    assert day16["history_lag1_value"] == pytest.approx(0.0)
    assert np.isfinite(day16["history_lag1_value"])


def test_cumulative_accumulates_from_season_start():
    count = make_count(
        [
            (2020, "2020-07-15", 10, 5),
            (2020, "2020-07-16", 20, 5),
            (2020, "2020-07-17", 30, 5),
        ]
    )
    out = add_history_features(count, windows=(), cumulative=True)
    day3 = out[out.date == "2020-07-17"].iloc[0]
    # cumulative through day 2 only: (10+20)/(5+5)
    assert day3["history_cum_value"] == pytest.approx(np.log1p(30 / 10))


def test_multiple_periods_on_the_same_date_are_summed_not_double_counted():
    """Two sessions on the same date (e.g. two hourly blocks) must contribute their sum to
    the next day's lag window, not be treated as two separate days."""
    count = make_count(
        [
            (2020, "2020-07-15", 10, 5),
            (2020, "2020-07-15", 5, 1),  # a second period, same date
            (2020, "2020-07-16", 0, 5),
        ]
    )
    out = add_history_features(count, windows=(1,), cumulative=False)
    day16 = out[out.date == "2020-07-16"].iloc[0]
    assert day16["history_lag1_value"] == pytest.approx(np.log1p(15 / 6))


def test_row_count_and_order_preserved():
    count = make_count(
        [
            (2020, "2020-07-16", 10, 5),
            (2020, "2020-07-15", 5, 5),  # deliberately out of order
        ]
    )
    out = add_history_features(count, windows=(1,), cumulative=False)
    assert len(out) == len(count)
    assert list(out["date"]) == list(count["date"])


def test_n_history_channels():
    assert n_history_channels([], False) == 0
    assert n_history_channels([1], False) == 2
    assert n_history_channels([1, 3, 7], False) == 6
    assert n_history_channels([1, 3, 7], True) == 8
