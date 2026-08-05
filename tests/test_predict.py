"""Tests for the season guard (DEVELOPMENT.md 4.11).

Training is restricted to a day-of-year window; the daily cron runs every day of the year
regardless. `is_in_season` is what stops the job from publishing a confident extrapolation
for the seven months a year the model has never seen.
"""

from datetime import datetime, timezone

import pytest

from src.predict import is_in_season

DOY_RANGE = (196, 335)  # matches configs/data/defile.yaml


@pytest.mark.parametrize(
    "date_str, expected",
    [
        ("2026-01-15", False),  # deep off-season
        ("2026-07-14", False),  # day before the season starts (doy 195)
        ("2026-07-15", True),  # first day of the season (doy 196)
        ("2026-09-15", True),  # mid-season
        ("2026-12-01", True),  # last day of the season (doy 335)
        ("2026-12-02", False),  # day after the season ends
        ("2026-12-31", False),  # deep off-season, end of year
    ],
)
def test_is_in_season_boundaries(date_str, expected):
    date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    assert is_in_season(DOY_RANGE, date) is expected


def test_is_in_season_accepts_a_plain_list_like_omegaconf_gives():
    """Hydra hands `cfg.data.doy` over as an OmegaConf ListConfig, not a tuple."""
    date = datetime(2026, 9, 15, tzinfo=timezone.utc)
    assert is_in_season([196, 335], date) is True
