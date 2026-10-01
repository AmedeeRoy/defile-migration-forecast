"""Tests for `season_level` (src/metrics.py): the season total ratio is pooled over years."""

import numpy as np
import pandas as pd

from src.metrics import era_of, season_level


def _daily(totals: dict) -> pd.DataFrame:
    """One row per year: {year: (obs, pred, phen)} as that year's only day."""
    years = np.array(sorted(totals))
    obs, pred, phen = np.array([totals[y] for y in years], dtype=float).T
    return pd.DataFrame(
        {
            "year": years,
            "doy": 250,
            "era": era_of(years),
            "obs": obs,
            "pred": pred,
            "phen": phen,
        }
    )


def test_total_ratio_is_pooled_not_mean_of_years():
    # A near-empty year with a huge ratio must not dominate: mean of ratios would be
    # (50 + 1) / 2 = 25.5, the pooled ratio is (50 + 1000) / (1 + 1000).
    out, per_year = season_level(_daily({1967: (1, 50, 1), 2019: (1000, 1000, 1000)}))
    assert np.isclose(out["season_total_ratio"], 1050 / 1001)
    assert np.isclose(per_year.set_index("year").loc[1967, "total_ratio"], 50)


def test_phenology_total_ratio_reported_alongside():
    out, _ = season_level(_daily({2001: (100, 150, 120), 2005: (100, 150, 80)}))
    assert np.isclose(out["season_total_ratio"], 1.5)
    assert np.isclose(out["season_total_ratio_phen"], 1.0)


def test_total_ratio_nan_when_nothing_observed():
    out, _ = season_level(_daily({2001: (0, 5, 5)}))
    assert np.isnan(out["season_total_ratio"])
