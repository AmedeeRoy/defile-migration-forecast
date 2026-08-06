#!/usr/bin/env python3
"""Builds the per-species day-of-year phenology statistics used as the naive baseline.

Writes `data/count/species_doy_statistics.json`: for each species, a 7-day-smoothed
distribution (mean, min, max, quantiles) of the daily count rate (birds/hr) by day of
year, plus a GAM-fitted hour-of-day activity `ratio` (hourly rate / that day's rate).

Two consumers read this file, so its schema is a contract, not free to change casually:

- `src.phenology.Phenology` (this repo) -- the day-of-year / persistence baseline that
  every skill score in the test report is computed against.
- defileViz's uncertainty bands (see the restore commit c11f723) -- the model's own
  uncertainty channel was dropped as untrained, so the frontend shows this file's
  quantile spread instead.

This used to exist only as `notebooks/phenology_baseline.ipynb`, which is why it drifted:
`pygam` was never a pinned dependency (nothing forced anyone to have it installed to run
training), and the notebook's own quantile/min/max statistics list had been edited since
whatever version actually produced the currently-committed JSON -- running the notebook
top to bottom today would *not* reproduce that file's schema. This script is the fix:
`pygam` is now pinned in `pyproject.toml`, the species list is read from
`configs/experiment/*.yaml` (the actual set of modelled species) instead of a second
hardcoded copy, and the hour grid for `ratio` (`RATIO_HOURS`) is imported from
`src.phenology` instead of a third copy of `range(6, 18)` -- so a change to either can no
longer silently disagree with this file's shape.

Usage:
    python scripts/build_phenology_stats.py                 # all 11 modelled species
    python scripts/build_phenology_stats.py --species "Osprey" "Red Kite"
    python scripts/build_phenology_stats.py --dry-run        # fit and report, don't write

`notebooks/phenology_baseline.ipynb` runs this script and plots what it produced. For
exploring or tuning the fit itself (GAM diagnostic plots, spline-count search, raw
pre-smoothing ratio samples), use this module's `PhenologyBuilder` directly.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import rootutils
import yaml
from pygam import PoissonGAM, s
from pygam.utils import OptimizationError

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.phenology import PHENOLOGY_FILE, RATIO_HOURS  # noqa: E402

# Percentiles baked into the currently-committed file (`quantile_levels`); kept as the
# default rather than re-derived so a re-run without `--quantiles` reproduces the same
# shape `Phenology` and defileViz already expect.
QUANTILE_LEVELS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)

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
# (only Osprey and European Honey-buzzard were ever fitted there). Trying progressively
# stronger regularization and keeping the first one that converges reproduces the
# historical fit exactly for species that were already well-behaved, and still produces a
# usable (slightly smoother) surface for the ones that were not, rather than crashing an
# 11-species run over the one species that needed it.
GAM_LAM_LADDER = [(100, 10), (1000, 100), (10_000, 1_000), (100_000, 10_000)]

SMOOTH_WINDOW = 7


def species_from_experiments(configs_dir: str) -> list:
    """The set of modelled species, read from `configs/experiment/*.yaml`.

    Deliberately not a hardcoded list living in this script: those 11 files are already
    the authoritative "what species does this project model" (AGENTS.md "Species /
    experiment pattern"), and a second copy here is exactly the kind of drift that left
    this generator out of sync with its own output in the first place.
    """
    species = []
    for path in sorted(glob.glob(os.path.join(configs_dir, "experiment", "*.yaml"))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        name = cfg.get("data", {}).get("species")
        if name is None:
            raise ValueError(f"{path} has no data.species entry")
        species.append(name)
    return species


def default_doy_range(configs_dir: str) -> list:
    """The trained season, read from `configs/data/defile.yaml`'s `doy` field.

    Fitting this file over a different season than the model is trained on would make it
    a baseline for a question nobody is asking; reading the value rather than copying it
    keeps the two from silently diverging.
    """
    with open(os.path.join(configs_dir, "data", "defile.yaml")) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["doy"])


class PhenologyBuilder:
    """Loads count data once and computes per-species day-of-year phenology statistics."""

    def __init__(self, data_dir: str, years, doy):
        self.data_dir = data_dir
        self.doy = list(doy)

        all_count = pd.read_csv(
            os.path.join(data_dir, "count", "all_count_processed.csv"),
            parse_dates=["date", "start", "end"],
        )
        all_count["doy"] = all_count["date"].dt.day_of_year
        all_count["year"] = all_count["date"].dt.year
        all_count = all_count[
            all_count["date"].dt.year.isin(years)
            & all_count["date"].dt.day_of_year.between(self.doy[0], self.doy[1])
        ]
        self.all_count = all_count

        # Every observation period regardless of species, deduplicated -- the frame every
        # species' zero-filled count is built against (a period with no record of a given
        # species in `all_count` is a real zero, not a missing row).
        self.all_periods = all_count[
            [c for c in all_count.columns if c not in ("species", "count")]
        ].drop_duplicates()
        self.all_periods["duration"] = (
            self.all_periods["end"] - self.all_periods["start"]
        ).dt.total_seconds() / 3600

        self.trektellen_id = self._load_trektellen_ids(data_dir)

    @staticmethod
    def _load_trektellen_ids(data_dir: str) -> dict:
        taxonomy = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
        taxonomy["trektellen_species_id"] = pd.to_numeric(
            taxonomy["trektellen_species_id"], errors="coerce"
        )
        return {
            name: int(tid)
            for name, tid in taxonomy[["English name", "trektellen_species_id"]].itertuples(
                index=False
            )
            if not (isinstance(tid, float) and math.isnan(tid))
        }

    def _count_species(self, species: str) -> pd.DataFrame:
        """Every observation period, zero-filled for periods with no record of `species`."""
        observed = (
            self.all_count[self.all_count["species"] == species][["date", "count", "start", "end"]]
            .groupby(["date", "start", "end"], as_index=False)["count"]
            .sum()
        )
        if observed.empty:
            raise ValueError(f"No data for species {species!r} in the selected years/doy.")

        count = self.all_periods.merge(observed, how="left")
        count["count"] = count["count"].fillna(0)
        count["count_rate"] = count["count"] / count["duration"]
        return count.dropna(subset=["count_rate"])

    def _daily_count(self, species: str) -> pd.DataFrame:
        """One row per date, summing same-day periods (rare, but the data has some)."""
        count = self._count_species(species)
        daily = (
            count.groupby(["date", "doy", "year"])
            .agg(count=("count", "sum"), duration=("duration", "sum"))
            .reset_index()
        )
        daily["count_rate"] = daily["count"] / daily["duration"]
        return daily

    def _hourly_ratio_samples(self, species: str) -> pd.DataFrame:
        """Rows usable for fitting the hour-of-day ratio: (doy, hour, ratio) triples.

        Only dates split into several periods carry hour-of-day information at all -- a
        date recorded as one dawn-to-dusk block constrains the day's *total*, not its
        shape, so it cannot supply a ratio and would only add noise if it did.
        """
        count = self._count_species(species)
        daily = self._daily_count(species)[["date", "count_rate"]]
        df = count.merge(daily, on="date", suffixes=("", "_daily"))

        df["ratio"] = df["count_rate"] / df["count_rate_daily"]
        df["hour"] = (df["start"] + (df["end"] - df["start"]) / 2).dt.hour
        df["n_periods"] = df.groupby("date")["count_rate"].transform("count")

        return df.loc[df["n_periods"] > 1, ["doy", "hour", "ratio"]].dropna()

    def fit_hourly_ratio(
        self, species: str, k0: int = DOY_SPLINES, k1: int = HOUR_SPLINES
    ) -> np.ndarray:
        """GAM-fitted `ratio(doy, hour)` over the full `self.doy` range x `RATIO_HOURS`.

        Shape `(n_doy, n_hours)`, aligned index-for-index with the day-of-year grid the
        other statistics use (`np.arange(doy[0], doy[1] + 1)`, inclusive) -- the source of
        the bug this script fixes: the notebook's version built this grid with
        `np.arange(doy[0], doy[1])`, one day short, so `ratio` silently carried one fewer
        row than `mean`/`doy` in every species already in the committed file. Every
        consumer that pairs them up by position (`src.phenology.Phenology.hourly_rate`) is
        now robust to that historical mismatch, but a freshly built file should not have
        it in the first place.
        """
        samples = self._hourly_ratio_samples(species)
        X, y = samples[["doy", "hour"]].to_numpy(), samples["ratio"].to_numpy()

        gam = None
        for lam in GAM_LAM_LADDER:
            try:
                gam = PoissonGAM(s(0, n_splines=k0) + s(1, n_splines=k1), lam=list(lam)).fit(X, y)
                if lam != GAM_LAM_LADDER[0]:
                    print(f"  {species}: PIRLS needed lam={lam} to converge")
                break
            except OptimizationError:
                continue
        if gam is None:
            raise OptimizationError(
                f"{species}: PIRLS did not converge even at lam={GAM_LAM_LADDER[-1]}"
            )

        doy_grid = np.arange(self.doy[0], self.doy[1] + 1)
        grid = np.column_stack(
            [np.repeat(doy_grid, len(RATIO_HOURS)), np.tile(RATIO_HOURS, len(doy_grid))]
        )
        return gam.predict(grid).reshape(len(doy_grid), len(RATIO_HOURS))

    def build(self, species: str, quantile_levels=QUANTILE_LEVELS, window=SMOOTH_WINDOW) -> dict:
        """The full per-species record: smoothed daily-rate statistics plus the hourly ratio."""
        daily = self._daily_count(species)

        by_doy = daily.groupby("doy")["count_rate"]
        stats = pd.DataFrame(
            {
                "doy": by_doy.mean().index,
                "min": by_doy.min().to_numpy(),
                "max": by_doy.max().to_numpy(),
                "mean": by_doy.mean().to_numpy(),
                "count_observations": by_doy.apply(lambda x: int((x > 0).sum())).to_numpy(),
                "quantiles": list(
                    by_doy.quantile([q / 100 for q in quantile_levels]).unstack().to_numpy()
                ),
            }
        )

        # Re-index onto the full doy range before smoothing: a day of year with zero
        # observed dates (possible at the season edges in the sparsest years) would
        # otherwise be silently dropped from the grid instead of interpolated across, and
        # every downstream consumer indexes this array by position, not by doy value.
        full_doy = pd.DataFrame({"doy": np.arange(self.doy[0], self.doy[1] + 1)})
        stats = full_doy.merge(stats, on="doy", how="left")
        stats["count_observations"] = stats["count_observations"].fillna(0)

        smoothed = self._smooth(stats, window=window)

        trektellen_id = self.trektellen_id.get(species)

        return {
            "species": species,
            "doy": smoothed["doy"].tolist(),
            "min": smoothed["min"].tolist(),
            "max": smoothed["max"].tolist(),
            "mean": smoothed["mean"].tolist(),
            "count_observations": stats["count_observations"].astype(int).tolist(),
            "quantile_levels": list(quantile_levels),
            "quantiles": np.stack(smoothed["quantiles"].to_numpy()).tolist(),
            "trektellen_species_id": trektellen_id,
            "ratio": self.fit_hourly_ratio(species).tolist(),
        }

    @staticmethod
    def _smooth(stats: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling mean over doy for every statistic except the raw observation count.

        `quantiles` is a column of arrays rather than a scalar, so it is smoothed
        separately by stacking into a 2-D array, rolling, and unstacking back -- pandas'
        own `.rolling().mean()` does not average object-dtype columns elementwise.
        """
        stats = stats.sort_values("doy").reset_index(drop=True)
        scalar_cols = ["min", "max", "mean"]

        rolled = (
            stats.set_index("doy")[scalar_cols]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .reset_index()
        )

        quantile_stack = np.stack(stats["quantiles"].to_numpy())  # (n_doy, n_levels)
        smoothed_quantiles = (
            pd.DataFrame(quantile_stack, index=stats["doy"])
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        rolled["quantiles"] = list(smoothed_quantiles.to_numpy())

        return rolled


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--configs-dir", default="configs", help="Hydra configs directory")
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Species to build (default: every species in configs/experiment/*.yaml)",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        default=[1966, 2030],
        metavar=("START", "END"),
        help="Year range, end exclusive (default: 1966 2030 -- pool every year available; "
        "this is a baseline, not a model, so more history is strictly better)",
    )
    parser.add_argument(
        "--doy",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Day-of-year range, inclusive (default: read from configs/data/defile.yaml)",
    )
    parser.add_argument(
        "--out", default=None, help=f"Output path (default: <data-dir>/{PHENOLOGY_FILE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fit and report, but do not write the file"
    )
    args = parser.parse_args()

    species_list = args.species or species_from_experiments(args.configs_dir)
    doy = args.doy or default_doy_range(args.configs_dir)
    out_path = args.out or os.path.join(args.data_dir, PHENOLOGY_FILE)

    print(f"Species ({len(species_list)}): {', '.join(species_list)}")
    print(f"Years: [{args.years[0]}, {args.years[1]}) | doy: {doy}")

    builder = PhenologyBuilder(data_dir=args.data_dir, years=range(*args.years), doy=doy)

    missing_ids = [s for s in species_list if builder.trektellen_id.get(s) is None]
    if missing_ids:
        print(f"Warning: no trektellen_species_id for: {', '.join(missing_ids)}", file=sys.stderr)

    records = []
    for species in species_list:
        print(f"Fitting {species}...")
        records.append(builder.build(species))

    if args.dry_run:
        print(f"Dry run: built {len(records)} species, not writing {out_path}")
        return 0

    # Write to a temp file and rename into place: this file is read by every training/eval
    # run and, once deployed, by defileViz -- a reader must never see a half-written file.
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp_path, out_path)

    print(f"Wrote {len(records)} species to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
