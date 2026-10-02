#!/usr/bin/env python3
"""Builds `data/count/all_count_processed.csv` from the defile-dataset tables, plus a QA report.

The counts are built and documented in the separate defile-dataset repo, which publishes
`surveys.csv`, `observations.csv` and `metadata.json`. `--dataset <dir>` copies those three
files from a defile-dataset build (its `output/` folder, or an unpacked release) into
`data/count/dataset/`; without it, the copy already there is used. This script then applies the
model's processing (`src/data/counts.py`: hourly splitting, zero-fill, the flagged rows the
model cannot use) and writes the model's count file, read by `DefileDataModule.read_counts`
and `scripts/build_phenology_stats.py`. Rebuild the phenology statistics and retrain after it
changes.

Every run (including --dry-run) also writes an HTML report, by default to
`logs/qa/counts/count_processing_report.html`: the checks (pass/warn/fail), each processing step
with the rows and birds it removed or moved, birds in vs. out per year, survey effort, and a
few example days. Data-entry errors and the dataset's own checks are in defile-dataset's report.

Usage:
    python scripts/build_counts.py --dataset ../defile-dataset/output
    python scripts/build_counts.py             # re-run on data/count/dataset/
    python scripts/build_counts.py --dry-run   # report only
"""

import argparse
import os
import shutil
import sys

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts._species_stats_common import species_from_experiments  # noqa: E402
from src.data import counts as C  # noqa: E402
from src.plots.count_report import render  # noqa: E402

ROOT = rootutils.find_root(__file__, indicator=".project-root")
DEFAULT_REPORT = os.path.join(ROOT, "logs", "qa", "counts", "count_processing_report.html")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--dataset", help="defile-dataset output folder to copy the tables from")
    ap.add_argument("--report", default=DEFAULT_REPORT, help='HTML report path; "" to skip')
    ap.add_argument("--dry-run", action="store_true", help="don't write the CSV")
    args = ap.parse_args(argv)

    dataset_dir = os.path.join(args.data_dir, C.DATASET_DIR)
    if args.dataset:
        os.makedirs(dataset_dir, exist_ok=True)
        for f in C.DATASET_FILES:
            shutil.copy2(os.path.join(args.dataset, f), os.path.join(dataset_dir, f))
        print(f"Copied {', '.join(C.DATASET_FILES)} from {args.dataset}")
    out_path = os.path.join(args.data_dir, "count", "all_count_processed.csv")

    surveys, observations, metadata = C.read_dataset(args.data_dir)
    print(
        f"Dataset: {len(surveys):,} surveys, {len(observations):,} observations "
        f"(built {metadata.get('built_at')}, defile-dataset {metadata.get('git_sha') or '?'})"
    )

    mc = C.build_model_counts(surveys, observations)
    checks = C.check_model_counts(mc)
    for c in checks:
        print(f"  [{c.status:4s}] {c.name}: {c.detail}")

    if args.dry_run:
        print(f"--dry-run: {len(mc.counts)} rows, not written.")
    else:
        tmp = out_path + ".tmp"
        mc.counts.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        print(f"Wrote {out_path} ({len(mc.counts)} rows).")

    if args.report:
        inputs = {
            "Dataset": f"{dataset_dir}: {len(surveys):,} surveys, {len(observations):,} "
            f"observations, years {metadata.get('years')}",
            "defile-dataset": f"built {metadata.get('built_at')}, commit "
            f"{metadata.get('git_sha') or '?'}",
            "Output": out_path + (" (dry run, not written)" if args.dry_run else ""),
        }
        species = species_from_experiments(os.path.join(ROOT, "configs"))
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(render(mc, checks, observations, species, inputs))
        print(f"Wrote report {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
