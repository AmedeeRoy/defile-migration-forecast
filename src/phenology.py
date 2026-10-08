"""The day-of-year phenology baseline: `data/count/species_doy_statistics.json`.

Built by `scripts/build_phenology_stats.py`, one record per species, each holding a
7-day-smoothed distribution of the *daily* count rate (birds/hr) by day of year plus a
GAM-fitted hour-of-day activity `ratio` (hourly rate / that day's rate).

Two consumers read this file:

- `src.metrics` (this repo's evaluation module) -- day-of-year phenology is the naive
  baseline every skill score in the test report is computed against. If the model doesn't
  beat it, the weather features aren't contributing anything.
- defileViz's uncertainty bands -- the model's own uncertainty channel was dropped as
  untrained, so the frontend shows this file's quantile spread instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.weather import night_mask_by_doy_hour

PHENOLOGY_FILE = os.path.join("count", "species_doy_statistics.json")

# `ratio` in the phenology file is a GAM fit of (hourly period rate / that day's rate)
# over this hour grid -- see `PhenologyBuilder.fit_hourly_ratio` in
# `scripts/build_phenology_stats.py`, which generated the file. 04-18 UTC is every hour with
# survey data to fit (hour 4: ~40 survey days per species, hour 19: 1-2); it was 06-17,
# which left the surveyed 05 and 18 UTC to a copy of the edge hour that overstated both.
RATIO_HOURS: np.ndarray = np.arange(4, 19)

# GAM spline counts for the doy/hour ratio surface. `notebooks/phenology_baseline.ipynb` has an
# AIC-search cell over a handful of (k0, k1) combinations that never fed back into these
# defaults -- worth revisiting there before changing these, rather than guessing new ones.
DOY_SPLINES = 4
HOUR_SPLINES = 12

# Regularization ladder for the PIRLS fit, weakest first. `ratio` is a heavily
# zero-inflated, long-tailed quantity (rare species can be >70% zero with a handful of
# ratios in the 10-40 range), and PoissonGAM's PIRLS diverges outright for some species at
# the historical (100, 10) strength -- Hen Harrier and Merlin, tried while writing this
# script, neither of which appears in `notebooks/phenology_baseline.ipynb`'s exploratory cells
# (only Osprey and European Honey Buzzard were ever fitted there). Trying progressively
# stronger regularization and keeping the first one that converges reproduces the
# historical fit exactly for species that were already well-behaved, and still produces a
# usable (slightly smoother) surface for the ones that were not, rather than crashing an
# 11-species run over the one species that needed it.
GAM_LAM_LADDER = [(100, 10), (1000, 100), (10_000, 1_000), (100_000, 10_000)]

# Each day's hourly-ratio samples are weighted by (that day's bird count) ** this in the GAM
# fit. 0 is the historical fit -- every day equal, so a 2-bird day shapes the curve as much as
# a 2 000-bird one; 1 is the unbiased estimator of the *expected* hourly rate the model
# predicts, but lets a handful of huge days dominate. 0.5 was best on held-out years (fit on
# even years, L1 to odd years' count-weighted hourly profile, 7 species): 0.253 / 0.213 /
# 0.229 for 0 / 0.5 / 1 -- 1 lost on Honey Buzzard, whose shape a few huge days dominate.
RATIO_WEIGHT_POWER = 0.5


def fit_ratio_surface(
    doy: np.ndarray,
    hour: np.ndarray,
    ratio: np.ndarray,
    weights: np.ndarray,
    doy_grid: np.ndarray,
    hours: np.ndarray,
    k0: int = DOY_SPLINES,
    k1: int = HOUR_SPLINES,
    interaction: bool = True,
    lam_ladder=GAM_LAM_LADDER,
    label: str = "",
) -> np.ndarray:
    """Poisson GAM of `ratio` (a period's rate / its day's rate) on (doy, hour), predicted on
    `doy_grid` x `hours`: shape `(len(doy_grid), len(hours))`.

    The one fit of the time-of-day shape, shared by the model's phenology baseline
    (`scripts/build_phenology_stats.py`, UTC hours of the model's periods) and the Explore export
    (`src/explore/`, local clock hours). `interaction` fits `te(doy, hour)`, else `s(doy) +
    s(hour)`, a doy-invariant shape once normalised (see `PhenologyBuilder.fit_hourly_ratio`). If
    PIRLS diverges, the fit is retried with progressively stronger regularization from `lam_ladder`
    (see `GAM_LAM_LADDER`), and so is one whose prediction overflows.
    """
    from pygam import PoissonGAM, s, te
    from pygam.utils import OptimizationError

    term = te(0, 1, n_splines=[k0, k1]) if interaction else s(0, n_splines=k0) + s(1, n_splines=k1)
    X = np.column_stack([doy, hour])
    grid = np.column_stack([np.repeat(doy_grid, len(hours)), np.tile(hours, len(doy_grid))])
    for lam in lam_ladder:
        try:
            gam = PoissonGAM(term, lam=list(lam)).fit(X, ratio, weights=weights)
        except OptimizationError:
            continue
        surface = gam.predict(grid)
        if np.isfinite(surface).all():  # PIRLS can also overflow without raising
            break
    else:
        raise OptimizationError(f"{label}: PIRLS did not converge even at lam={lam_ladder[-1]}")
    if lam != lam_ladder[0]:
        print(f"  {label}: PIRLS needed lam={lam} to converge")
    return surface.reshape(len(doy_grid), len(hours))


@dataclass
class Phenology:
    """Day-of-year phenology for one species, the primary naive baseline.

    Loaded from `data/count/species_doy_statistics.json`, which holds, per day of year, a 7-day-
    smoothed distribution of the *daily* count rate (birds/hr) plus a fitted hourly activity
    `ratio`.

    Known caveat, carried from DEVELOPMENT.md: the file has no `year` field, so it is pooled over
    all years including whichever ones land in the test split. That is a mild leakage risk on the
    *baseline* side -- it can only make the baseline look better and the model's skill score worse,
    so it is conservative, not flattering. Worth rebuilding per-split if a skill score ever looks
    suspiciously good.
    """

    species: str
    doy: np.ndarray  # (D,)
    mean: np.ndarray  # (D,) daily count rate, birds/hr
    quantile_levels: np.ndarray  # (Q,) percent
    quantiles: np.ndarray  # (D, Q)
    ratio: np.ndarray  # (D, len(RATIO_HOURS)) hourly rate / daily rate

    @classmethod
    def load(cls, data_dir: str, species: str) -> "Phenology":
        path = os.path.join(data_dir, PHENOLOGY_FILE)
        with open(path) as f:
            entries = json.load(f)

        for entry in entries:
            if entry["species"] == species:
                return cls(
                    species=species,
                    doy=np.asarray(entry["doy"], dtype=int),
                    mean=np.asarray(entry["mean"], dtype=float),
                    quantile_levels=np.asarray(entry["quantile_levels"], dtype=float),
                    quantiles=np.asarray(entry["quantiles"], dtype=float),
                    ratio=np.asarray(entry["ratio"], dtype=float),
                )

        available = ", ".join(sorted(e["species"] for e in entries))
        raise KeyError(f"No phenology for species {species!r} in {path}. Available: {available}")

    def _positions(self, doy: Sequence[int]) -> np.ndarray:
        """Index of each requested doy in the phenology grid, clipped to its range.

        Clipping rather than raising keeps the baseline defined at the season edges: the
        grid covers the trained season only, and a doy one day outside it is far better
        served by the nearest fitted value than by a NaN that silently drops the row from
        every skill score.
        """
        return np.clip(np.searchsorted(self.doy, np.asarray(doy)), 0, len(self.doy) - 1)

    def daily_rate(self, doy: Sequence[int]) -> np.ndarray:
        """Phenological mean count rate (birds/hr) for each day of year."""
        return self.mean[self._positions(doy)]

    def quantile(self, doy: Sequence[int], level: float) -> np.ndarray:
        """Phenological quantile of the daily rate, e.g. `level=90` for the p90 threshold.

        Used as the per-doy event threshold at metric level 2: "a big day for this species,
        at this point in the season" rather than one fixed count for the whole season.
        """
        q = int(np.argmin(np.abs(self.quantile_levels - level)))
        return self.quantiles[self._positions(doy), q]

    def hourly_rate(self, doy: Sequence[int]) -> np.ndarray:
        """Phenological hourly profile, shape `(len(doy), 24)`, in birds/hr.

        The daily rate scaled by the fitted hour-of-day `ratio`. Hours outside `RATIO_HOURS` were
        never fitted and are returned as zero.

        `ratio` and `doy`/`mean` are aligned index-for-index (`scripts/build_phenology_stats.py`
        builds both over the same inclusive doy range). Positions are still clipped separately
        against `ratio`'s own length as defence-in-depth: a `species_doy_statistics.json` built by
        something other than that script -- or an older copy of it -- is not guaranteed to have
        fixed the historical one-day-short `ratio` array this once had, and a doy landing on a
        missing last day should get the nearest available fit rather than an IndexError.
        """
        pos = self._positions(doy)
        ratio_pos = np.clip(pos, 0, len(self.ratio) - 1)
        profile = np.zeros((len(pos), 24), dtype=float)
        profile[:, RATIO_HOURS] = self.ratio[ratio_pos] * self.mean[pos][:, None]
        return profile

    def hourly_shape(self, doy: Sequence[int]) -> np.ndarray:
        """The day's diurnal *shape* -- `hourly_rate`, normalised to sum to 1 across the 24 hours
        -- with every astronomically-night hour forced to exactly 0 first.

        Deep night is forced by `night_mask_by_doy_hour` (`src/data/weather.py`), not
        fitted: `ratio`'s GAM already only ever sees real hourly-bin data (`RATIO_HOURS`,
        4-18), and per DEVELOPMENT.md there is essentially none at night to fit against
        anyway (2 individuals total, dataset-wide, across all 11 modelled species) --
        a deterministic astronomical fact is both cheaper and more reliable than
        extrapolating a spline into territory it has no real data for.

        That mask is the *only* thing that zeros an hour. The few daylight hours outside
        `RATIO_HOURS` (03 and 19 UTC in July, never surveyed) hold the nearest fitted hour's
        value -- already close to 0 there -- instead of 0, because `UNetplus.forward` forbids
        any prediction where the prior is 0, and only the sun should decide that.

        Used as the shape prior `UNetplus`'s `out_h` defaults to before any weather
        evidence shifts it (see DECISIONS.md -> Model architecture) -- unlike every other
        method on this class, this one is read at training/forecast time, not only at
        evaluation time.

        Days with a phenological rate of exactly zero (season edges) fall back to a
        uniform shape over that day's daylight hours, rather than an all-zero vector that is
        undefined once normalised -- so on every day, an hour is zero iff it is night, which
        `UNetplus.forward` relies on to mask its output.
        """
        profile = self.hourly_rate(doy)
        first, last = RATIO_HOURS[0], RATIO_HOURS[-1]
        profile[:, :first] = profile[:, [first]]
        profile[:, last + 1 :] = profile[:, [last]]
        doy_arr = np.clip(np.asarray(doy), 1, 366)
        profile = np.where(night_mask_by_doy_hour()[doy_arr - 1], 0.0, profile)

        total = profile.sum(axis=1)
        zero_days = total == 0
        profile[zero_days] = ~night_mask_by_doy_hour()[doy_arr[zero_days] - 1]
        total[zero_days] = profile[zero_days].sum(axis=1)

        return profile / total[:, None]
