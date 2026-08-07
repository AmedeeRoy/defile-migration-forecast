#!/usr/bin/env python3
"""Aggregates `*_metrics.json` across seeds into mean +/- std, per species and era.

Model-quality comparisons in this project (loss_weighting variants, year-subset ladder,
location/variable ablation, ...) must never be judged from a single training run. Two
Red Kite runs that differed *only* in `seed` (weight init / batch order -- same data,
same split, same everything else) moved event CSI by 44% and POD by 47% with nothing
about the model or data changed. A config comparison that shows a smaller difference
than that is indistinguishable from seed noise.

The standard from here on: run every config that's being judged with >=3 seeds and
aggregate before comparing, rather than reading one run's numbers:

    uv run python src/train.py --multirun hydra=seed_sweep experiment=red_kite seed=0,1,2
    uv run python scripts/aggregate_seeds.py \
        "logs/train/multiruns/<timestamp>/*/*/Red_Kite_metrics.json"

`hydra=seed_sweep` is required for a seed sweep: the default sweep subdir is just the
species name, so without it every seed of the same species overwrites the previous
seed's metrics.json/report.pdf (only checkpoints get auto-versioned) and this script
silently aggregates one run repeated, not three.

or, for runs launched individually (one `train.py` invocation per seed, e.g. because
they're spread across separate background processes rather than one `--multirun`),
just pass each run's metrics.json path directly:

    uv run python scripts/aggregate_seeds.py \
        logs/train/runs/2026-08-06_15-29-30/Red_Kite/Red_Kite_metrics.json \
        logs/train/runs/2026-08-06_16-18-42/Red_Kite/Red_Kite_metrics.json \
        logs/train/runs/2026-08-06_.../Red_Kite/Red_Kite_metrics.json

Groups inputs by the `species` field inside each file (not by filename), so metrics
from several species' seed sweeps can be passed in one call and it reports one table
per species.

Usage:
    python scripts/aggregate_seeds.py <metrics.json> [<metrics.json> ...]
    python scripts/aggregate_seeds.py "logs/train/multiruns/*/*/​*_metrics.json"
    python scripts/aggregate_seeds.py ... --out summary.json
"""

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Same headline set src/plots/report.py puts on the summary page, so this table reads
# like the report's numbers, just averaged over seeds instead of from one run.
HEADLINE = [
    "n_rows",
    "mae",
    "bias",
    "mae_skill_phen",
    "mae_skill_persistence",
    "event_csi",
    "event_pod",
    "event_far",
    "event_csi_skill_persistence",
    "shape_peak_hour_mae",
    "shape_emd",
    "season_median_date_mae",
    "season_total_ratio",
]

ERA_HEADLINE = ["mae", "mae_skill_phen", "event_csi", "season_median_date_mae"]


def _mean_std(values: list) -> tuple:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], float("nan")
    return statistics.mean(clean), statistics.stdev(clean)


def _resolve_paths(patterns: list) -> list:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def load_runs(patterns: list) -> dict:
    """Reads every metrics.json and groups by `species`. Returns {species: [run_dict, ...]}."""
    by_species = defaultdict(list)
    for path in _resolve_paths(patterns):
        p = Path(path)
        if not p.is_file():
            print(f"skipping (not found): {path}")
            continue
        run = json.loads(p.read_text())
        run["_path"] = str(p)
        by_species[run["species"]].append(run)
    return by_species


def summarize_species(runs: list) -> dict:
    """One species' seed group -> {scalars: {key: (mean, std)}, by_era: {era: {key: (mean, std)}}}."""
    scalars = {k: _mean_std([r["scalars"].get(k) for r in runs]) for k in HEADLINE}

    eras = defaultdict(list)
    for r in runs:
        for era_row in r.get("by_era", []):
            eras[era_row["era"]].append(era_row)
    by_era = {
        era: {k: _mean_std([row.get(k) for row in era_rows]) for k in ERA_HEADLINE}
        for era, era_rows in eras.items()
    }
    return {"n_seeds": len(runs), "scalars": scalars, "by_era": by_era}


def print_summary(species: str, summary: dict) -> None:
    n = summary["n_seeds"]
    print(f"\n{species} (n={n} seed{'s' if n != 1 else ''})")
    if n < 3:
        print("  ** fewer than 3 seeds -- treat this as provisional, not a settled comparison **")
    for key, (mean, std) in summary["scalars"].items():
        std_str = f"{std:8.4f}" if not math.isnan(std) else "     n/a"
        print(f"  {key:<28} {mean:9.4f}  +/- {std_str}")
    for era, era_stats in summary["by_era"].items():
        print(f"    era={era}")
        for key, (mean, std) in era_stats.items():
            std_str = f"{std:8.4f}" if not math.isnan(std) else "     n/a"
            print(f"      {key:<24} {mean:9.4f}  +/- {std_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="metrics.json paths or glob patterns")
    parser.add_argument("--out", type=str, default=None, help="write JSON summary here")
    args = parser.parse_args()

    by_species = load_runs(args.paths)
    if not by_species:
        raise SystemExit("No metrics.json files matched.")

    summaries = {}
    for species, runs in sorted(by_species.items()):
        summary = summarize_species(runs)
        summaries[species] = summary
        print_summary(species, summary)

    if args.out:
        Path(args.out).write_text(json.dumps(summaries, indent=2))
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
