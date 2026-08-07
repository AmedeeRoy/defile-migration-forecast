#!/usr/bin/env python3
"""Builds the per-species annual trend statistics used for the trend correction.

Writes `data/count/species_year_statistics.json`: for each species, the effort-corrected
annual seasonal rate (birds/observer-hour, summed over the trained season) for every year
of reliable coverage, plus a Holt damped-trend fit of that series. Read by
`src.trend.Trend`, which turns it into a scalar multiplier applied to the trained model's
output for the current forecast year.

Only years from 1993 onward are used: `data/count/readme.md` documents that real daily
systematic monitoring only started in 1993 (before that, coverage was sporadic and
pigeon-focused), matching this project's own `ERA_EDGES = (1993, 2014)` convention in
`src/metrics.py`.

The effort denominator is *site-level* watch-hours (the longest session logged by any
species on a given date), not a given species' own logged hours: since 2021, Trektellen
only creates a session row for a species actually seen that hour, so a scarce species' own
summed duration undercounts true watch time (verified: Hen Harrier's own hours drop to
~10-40h/season from 2021 on even though the site is watched ~12h/day all season). Sessions
also sometimes carry several separate rows for the exact same (species, date, start, end)
window (multiple sighting submissions within one nominal hour); those are collapsed to a
single window (summing count) before summing duration, or duration is overcounted 4-8x on
busy days -- verified against 2022+ Common Buzzard data, where naive summation gave
~1700-2300h/season instead of the real ~300-400h.

The Holt damping hyperparameters (alpha, beta, phi) are fixed, not fitted per species --
see `src.trend` for why. They were chosen by a rolling-origin one-year-ahead backtest
(DECISIONS.md) comparing this damped-trend method against a flat/no-trend baseline, a
3-year persistence average, and an unconstrained linear-trend extrapolation: damped trend
and persistence both cut pooled RMSE by ~43% versus the no-trend baseline, while
unconstrained linear extrapolation only managed ~12% and had the worst downside (-37% on
species with no real trend) -- exactly the runaway-extrapolation failure mode a damped
trend is meant to avoid.

Usage:
    python scripts/build_trend_stats.py                 # all 11 modelled species
    python scripts/build_trend_stats.py --species "Osprey" "Red Kite"
    python scripts/build_trend_stats.py --dry-run        # fit and report, don't write
"""

import argparse
import glob
import os
import sys
import json

import numpy as np
import pandas as pd
import rootutils
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.trend import ALPHA, BETA, PHI, TREND_FILE  # noqa: E402

MIN_YEAR = 1993
MIN_HOURS_PER_YEAR = 100  # drop species-years with too little site-level watch effort
MIN_YEARS = 15  # fewer than this and a multi-year trend isn't a meaningful fit


def species_from_experiments(configs_dir: str) -> list:
    """The set of modelled species, read from `configs/experiment/*.yaml`.

    Mirrors `scripts/build_phenology_stats.py`'s `species_from_experiments`: the 11
    experiment configs are the authoritative species list, not a second hardcoded copy.
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
    """The trained season, read from `configs/data/defile.yaml`'s `doy` field."""
    with open(os.path.join(configs_dir, "data", "defile.yaml")) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["doy"])


def damped_trend_fit(log_rate: np.ndarray, alpha=ALPHA, beta=BETA, phi=PHI):
    """Holt's damped double-exponential smoothing, fit by sequential updating.

    Returns `(level_path, final_level, final_trend)`: `level_path[t]` is the fitted level
    after seeing `log_rate[0..t]`, and `(final_level, final_trend)` is the state after the
    last observation, from which `src.trend.Trend.forecast_log_rate` extrapolates.
    """
    n = len(log_rate)
    level_path = np.empty(n)
    level = log_rate[0]
    trend = log_rate[1] - log_rate[0] if n > 1 else 0.0
    level_path[0] = level
    for t in range(1, n):
        prev_level = level
        level = alpha * log_rate[t] + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
        level_path[t] = level
    return level_path, level, trend


class TrendBuilder:
    """Loads count data once and computes per-species annual trend statistics."""

    def __init__(self, data_dir: str, doy, min_year=MIN_YEAR, min_hours=MIN_HOURS_PER_YEAR):
        self.doy = list(doy)
        self.min_year = min_year
        self.min_hours = min_hours

        count = pd.read_csv(
            os.path.join(data_dir, "count", "all_count_processed.csv"),
            parse_dates=["date", "start", "end"],
        )
        count["doy"] = count["date"].dt.day_of_year
        count["year"] = count["date"].dt.year
        count = count[count["doy"].between(self.doy[0], self.doy[1])]
        count = count[count["end"] > count["start"]]

        # Collapse duplicate (species, date, start, end) windows before summing duration --
        # see the module docstring.
        window = count.groupby(["species", "date", "start", "end"], as_index=False)["count"].sum()
        window["duration_h"] = (window["end"] - window["start"]).dt.total_seconds() / 3600
        window["year"] = window["date"].dt.year

        self.daily = (
            window.groupby(["species", "year", "date"])
            .agg(count=("count", "sum"), hours=("duration_h", "sum"))
            .reset_index()
        )
        site_hours = self.daily.groupby(["year", "date"])["hours"].max().reset_index()
        self.site_hours_year = site_hours.groupby("year")["hours"].sum()

    def annual_series(self, species: str) -> pd.DataFrame:
        """Effort-corrected annual rate for `species`, restricted to reliable years."""
        d = self.daily[self.daily["species"] == species]
        annual = d.groupby("year")["count"].sum().rename("total_count").reset_index()
        annual = annual.merge(self.site_hours_year.rename("total_hours"), on="year", how="left")
        annual["rate"] = annual["total_count"] / annual["total_hours"]
        annual = annual[
            (annual["total_hours"] >= self.min_hours) & (annual["year"] >= self.min_year)
        ]
        return annual.sort_values("year")

    def build(self, species: str) -> dict:
        annual = self.annual_series(species)
        if len(annual) < MIN_YEARS:
            raise ValueError(
                f"{species}: only {len(annual)} usable years (>= {self.min_hours}h, "
                f">= {self.min_year}); need >= {MIN_YEARS} for a trend fit."
            )

        year = annual["year"].to_numpy()
        log_rate = np.log(annual["rate"].to_numpy())
        level_path, level, trend = damped_trend_fit(log_rate)

        return {
            "species": species,
            "year": year.tolist(),
            "log_rate": log_rate.tolist(),
            "level_path": level_path.tolist(),
            "baseline_log_rate": float(np.mean(level_path)),
            "level": float(level),
            "trend": float(trend),
            "alpha": ALPHA,
            "beta": BETA,
            "phi": PHI,
            "last_year": int(year[-1]),
        }


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
        "--doy",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Day-of-year range, inclusive (default: read from configs/data/defile.yaml)",
    )
    parser.add_argument(
        "--out", default=None, help=f"Output path (default: <data-dir>/{TREND_FILE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fit and report, but do not write the file"
    )
    args = parser.parse_args()

    species_list = args.species or species_from_experiments(args.configs_dir)
    doy = args.doy or default_doy_range(args.configs_dir)
    out_path = args.out or os.path.join(args.data_dir, TREND_FILE)

    print(f"Species ({len(species_list)}): {', '.join(species_list)}")
    print(f"doy: {doy} | years >= {MIN_YEAR} | hours >= {MIN_HOURS_PER_YEAR}")

    builder = TrendBuilder(data_dir=args.data_dir, doy=doy)

    records = []
    skipped = []
    for species in species_list:
        try:
            record = builder.build(species)
        except ValueError as e:
            print(f"  skipping {species}: {e}", file=sys.stderr)
            skipped.append(species)
            continue
        print(
            f"  {species}: {len(record['year'])} years ({record['year'][0]}-{record['last_year']}), "
            f"multiplier(next year)={np.exp(record['level'] + record['phi'] * record['trend'] - record['baseline_log_rate']):.2f}"
        )
        records.append(record)

    if args.dry_run:
        print(f"Dry run: built {len(records)} species ({len(skipped)} skipped), not writing {out_path}")
        return 0

    # Write to a temp file and rename into place: this file is read at prediction time and
    # a reader must never see a half-written file.
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp_path, out_path)

    print(f"Wrote {len(records)} species to {out_path}" + (f" ({len(skipped)} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
