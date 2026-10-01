"""Tests for `Phenology.hourly_shape` (src/phenology.py) -- the smooth, astronomically night-
anchored 24h shape used as UNetplus's out_h prior (see DECISIONS.md -> Model architecture and the
`feature/phenology-shape-prior` plan).

Built entirely from
`Phenology.hourly_rate`, already tested indirectly via the real data file, plus
`night_mask_by_doy_hour` -- these tests use a small synthetic `Phenology` instead, so they
don't depend on `data/count/species_doy_statistics.json`'s actual content.
"""

import numpy as np
import pytest

from src.data.weather import night_mask_by_doy_hour
from src.phenology import RATIO_HOURS, Phenology


def _make_phenology(mean_by_doy: dict) -> Phenology:
    """A minimal synthetic Phenology: flat ratio=1 across RATIO_HOURS for every doy."""
    doy = np.array(sorted(mean_by_doy))
    mean = np.array([mean_by_doy[d] for d in doy], dtype=float)
    ratio = np.ones((len(doy), len(RATIO_HOURS)))
    return Phenology(
        species="Test Species",
        doy=doy,
        mean=mean,
        quantile_levels=np.array([50.0]),
        quantiles=mean[:, None],
        ratio=ratio,
    )


def test_hourly_shape_sums_to_one():
    phenology = _make_phenology({200: 10.0, 250: 5.0, 330: 2.0})
    shape = phenology.hourly_shape([200, 250, 330])
    assert np.allclose(shape.sum(axis=1), 1.0)


def test_hourly_shape_is_zero_at_astronomical_night():
    """Midwinter (doy 1) has the shortest day at this latitude -- RATIO_HOURS' fixed 6-17 window
    includes hours that are astronomically night that far into winter, and hourly_shape must zero
    those out even though hourly_rate (pre-existing, unchanged) does not."""
    phenology = _make_phenology({1: 10.0})
    shape = phenology.hourly_shape([1])[0]
    rate = phenology.hourly_rate([1])[0]

    # hourly_rate (unchanged, pre-existing behaviour) is nonzero across all of RATIO_HOURS
    assert (rate[RATIO_HOURS] > 0).all()
    # hourly_shape must have zeroed out whichever of those hours are actually night
    night_within_window = ~np.isclose(shape[RATIO_HOURS], rate[RATIO_HOURS] / rate.sum())
    assert night_within_window.any()  # some daytime-labelled hour actually got zeroed
    assert np.isclose(shape.sum(), 1.0)


def test_hourly_shape_falls_back_to_uniform_on_a_zero_rate_day():
    phenology = _make_phenology({200: 0.0})
    shape = phenology.hourly_shape([200])[0]
    assert np.isclose(shape.sum(), 1.0)
    assert np.allclose(shape[RATIO_HOURS], shape[RATIO_HOURS][0])  # uniform
    assert (shape[RATIO_HOURS] > 0).all()


@pytest.mark.parametrize("doy", [200, 260, 330])
def test_hourly_shape_is_zero_exactly_at_astronomical_night(doy):
    """The sun mask is the only thing that zeros an hour: every night hour is 0 and every daylight
    hour is not -- including daylight hours outside RATIO_HOURS (05/18 UTC in summer), which are
    surveyed and carry birds."""
    phenology = _make_phenology({doy: 10.0})
    shape = phenology.hourly_shape([doy])[0]
    night = night_mask_by_doy_hour()[doy - 1]
    assert np.array_equal(shape == 0, night)


def test_daylight_shoulder_hours_hold_the_nearest_fitted_hour():
    """Doy 200 (mid-July): 05 and 18 UTC are daylight but outside RATIO_HOURS -- they take the edge
    hour's value rather than 0."""
    doy = 200
    phenology = _make_phenology({doy: 10.0})
    shape = phenology.hourly_shape([doy])[0]
    night = night_mask_by_doy_hour()[doy - 1]
    first, last = RATIO_HOURS[0], RATIO_HOURS[-1]
    assert not night[first - 1] and not night[last + 1]  # premise: both are daylight in July
    assert np.isclose(shape[first - 1], shape[first])
    assert np.isclose(shape[last + 1], shape[last])
