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

PHENOLOGY_FILE = os.path.join("count", "species_doy_statistics.json")

# `ratio` in the phenology file is a GAM fit of (hourly period rate / that day's rate)
# over this hour grid -- see `PhenologyBuilder.fit_hourly_ratio` in
# `scripts/build_phenology_stats.py`, which generated the file. Hours outside it were
# never fitted, so the hourly phenology is zero there rather than extrapolated.
RATIO_HOURS: np.ndarray = np.arange(6, 18)


@dataclass
class Phenology:
    """Day-of-year phenology for one species, the primary naive baseline.

    Loaded from `data/count/species_doy_statistics.json`, which holds, per day of year, a
    7-day-smoothed distribution of the *daily* count rate (birds/hr) plus a fitted hourly
    activity `ratio`.

    Known caveat, carried from DEVELOPMENT.md: the file has no `year` field, so it is
    pooled over all years including whichever ones land in the test split. That is a mild
    leakage risk on the *baseline* side -- it can only make the baseline look better and
    the model's skill score worse, so it is conservative, not flattering. Worth rebuilding
    per-split if a skill score ever looks suspiciously good.
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

        The daily rate scaled by the fitted hour-of-day `ratio`. Hours outside
        `RATIO_HOURS` were never fitted and are returned as zero.

        `ratio` and `doy`/`mean` are aligned index-for-index (`scripts/build_phenology_stats.py`
        builds both over the same inclusive doy range). Positions are still clipped
        separately against `ratio`'s own length as defence-in-depth: a `species_doy_statistics.json`
        built by something other than that script -- or an older copy of it -- is not
        guaranteed to have fixed the historical one-day-short `ratio` array this once had,
        and a doy landing on a missing last day should get the nearest available fit
        rather than an IndexError.
        """
        pos = self._positions(doy)
        ratio_pos = np.clip(pos, 0, len(self.ratio) - 1)
        profile = np.zeros((len(pos), 24), dtype=float)
        profile[:, RATIO_HOURS] = self.ratio[ratio_pos] * self.mean[pos][:, None]
        return profile
