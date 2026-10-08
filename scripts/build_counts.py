#!/usr/bin/env python3
"""Builds `data/count/all_count_processed.csv` from the defile-dataset tables, plus a QA report.

The counts are built and documented in the separate defile-dataset repo, whose release holds
`dataset/{count,survey,taxonomy}.csv` with `dataset/datapackage.json`, and the build's
`metadata.json`. `--dataset <dir>` copies those files from a defile-dataset build (its `output/`
folder, or an unpacked release) into `data/count/dataset/`, with the recorded times of the entries
the release keeps at day level, from the build's `interim/processed/observations.csv`
(`entry_times.csv`); without it, the copy already there is used. This script then applies the model's processing (`src/data/counts.py`: the surveys and counts
the model cannot use, hourly splitting, zero-fill) and writes the model's count file, read by
`DefileDataModule.read_counts` and `scripts/build_phenology_stats.py`. Rebuild the phenology
statistics and retrain after it changes.

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

import pandas as pd
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
            shutil.copy2(os.path.join(args.dataset, "dataset", f), os.path.join(dataset_dir, f))
        shutil.copy2(
            os.path.join(args.dataset, C.METADATA_FILE),
            os.path.join(dataset_dir, C.METADATA_FILE),
        )
        print(f"Copied {', '.join(C.DATASET_FILES)} and {C.METADATA_FILE} from {args.dataset}")
        # Recorded times of the entries the release keeps at day level (timed outside their
        # survey), so the model's 10-min tolerance can be applied to them.
        interim = os.path.join(args.dataset, "..", "interim", "processed", "observations.csv")
        times_path = os.path.join(dataset_dir, C.ENTRY_TIMES_FILE)
        if os.path.exists(interim):
            obs = pd.read_csv(interim, usecols=["observation_id", "datetime_original"])
            count = pd.read_csv(os.path.join(dataset_dir, "count.csv"), low_memory=False)
            times = C.entry_times(obs, count)
            times.to_csv(times_path, index=False)
            print(f"Wrote {C.ENTRY_TIMES_FILE}: {len(times)} recorded times from {interim}")
        elif os.path.exists(times_path):
            os.remove(times_path)  # stale: from another dataset build
            print(f"No {interim}: entries timed outside their survey will be dropped.")
    out_path = os.path.join(args.data_dir, "count", "all_count_processed.csv")

    surveys, counts, _, metadata = C.read_dataset(args.data_dir)
    print(
        f"Dataset: {len(surveys):,} surveys, {len(counts):,} counts "
        f"(built {metadata.get('built_at')}, defile-dataset {metadata.get('git_sha') or '?'})"
    )

    mc = C.build_model_counts(surveys, counts)
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
            "Dataset": f"{dataset_dir}: {len(surveys):,} surveys, {len(counts):,} counts, "
            f"years {metadata.get('years')}",
            "defile-dataset": f"built {metadata.get('built_at')}, commit "
            f"{metadata.get('git_sha') or '?'}, schema {metadata.get('consolidated_schema')}",
            "Output": out_path + (" (dry run, not written)" if args.dry_run else ""),
        }
        species = species_from_experiments(os.path.join(ROOT, "configs"))
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(render(mc, checks, counts, species, inputs))
        print(f"Wrote report {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
