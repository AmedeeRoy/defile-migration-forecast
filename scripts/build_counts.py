#!/usr/bin/env python3
"""Builds `data/count/all_count_processed.csv`, the model's count file, from a defile-dataset
release.

The count data is published by the defile-dataset project as GitHub release assets (`count.csv`,
`survey.csv`, `taxonomy.csv`, `datapackage.json`, `metadata.json`, `SHA256SUMS`). This script
downloads a release into `data/count/dataset/`, verifies its checksums, then applies the model's
processing (`src/data/counts.py`, rules in `data/count/readme.md`) and writes the count file read
by `DefileDataModule.read_counts` and `scripts/build_phenology_stats.py`. Rebuild the phenology
statistics and retrain whenever the count file changes.

By default it uses the release pinned in `src.data.counts.DATASET_RELEASE`, downloading it only
if `data/count/dataset/` does not already hold it. `--release <tag>` picks another release.
`--dataset <dir>` is for development only: it copies the tables from a local defile-dataset
build (its `output/` folder) instead of downloading.

Every run (including --dry-run) also writes an HTML report, by default to
`logs/qa/counts/count_processing_report.html`: the checks (pass/warn/fail), each processing step
with the rows and birds it removed or moved, birds in vs. out per year, survey effort, and a
few example days.

Usage:
    python scripts/build_counts.py                          # the pinned release
    python scripts/build_counts.py --release v1.0.0-beta.1  # a specific release
    python scripts/build_counts.py --dataset ../defile-dataset/output   # local build (development)
    python scripts/build_counts.py --dry-run                # report only, write nothing
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts._species_stats_common import species_from_experiments  # noqa: E402
from src.data import counts as C  # noqa: E402
from src.plots.count_report import render  # noqa: E402

ROOT = rootutils.find_root(__file__, indicator=".project-root")
DEFAULT_REPORT = os.path.join(ROOT, "logs", "qa", "counts", "count_processing_report.html")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_release(tag: str, dataset_dir: str) -> None:
    """Fetch the release assets into `dataset_dir`, replacing it only if every checksum matches."""
    base = f"https://github.com/{C.DATASET_REPO}/releases/download/{tag}"
    tmp = dataset_dir + ".download"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        names = (*C.DATASET_FILES, C.METADATA_FILE, C.DATASET_CHECKSUMS_FILE)
        for name in names:
            print(f"Downloading {base}/{name}")
            urllib.request.urlretrieve(f"{base}/{name}", os.path.join(tmp, name))
        with open(os.path.join(tmp, C.DATASET_CHECKSUMS_FILE)) as f:
            expected = dict(reversed(line.split(None, 1)) for line in f if line.strip())
        expected = {k.strip().lstrip("*"): v for k, v in expected.items()}
        for name in (*C.DATASET_FILES, C.METADATA_FILE):
            if name not in expected:
                raise RuntimeError(f"{name} is not listed in {C.DATASET_CHECKSUMS_FILE}")
            if _sha256(os.path.join(tmp, name)) != expected[name]:
                raise RuntimeError(f"Checksum mismatch for {name} in release {tag}")
        with open(os.path.join(tmp, C.DATASET_RELEASE_MARKER), "w") as f:
            f.write(tag + "\n")
        shutil.rmtree(dataset_dir, ignore_errors=True)
        os.replace(tmp, dataset_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Verified checksums; {tag} is in {dataset_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument(
        "--release",
        metavar="TAG",
        help=f"defile-dataset release tag to use (default: {C.DATASET_RELEASE})",
    )
    ap.add_argument(
        "--dataset", help="development: copy the tables from a local defile-dataset output folder"
    )
    ap.add_argument("--report", default=DEFAULT_REPORT, help='HTML report path; "" to skip')
    ap.add_argument("--dry-run", action="store_true", help="don't write the CSV")
    args = ap.parse_args(argv)

    dataset_dir = os.path.join(args.data_dir, C.DATASET_DIR)
    if args.release and args.dataset:
        ap.error("--release and --dataset are mutually exclusive")
    marker = os.path.join(dataset_dir, C.DATASET_RELEASE_MARKER)
    if args.dataset:
        os.makedirs(dataset_dir, exist_ok=True)
        for f in (*C.DATASET_FILES, C.METADATA_FILE):
            src = os.path.join(
                args.dataset, f if f == C.METADATA_FILE else os.path.join("dataset", f)
            )
            shutil.copy2(src, os.path.join(dataset_dir, f))
        if os.path.exists(marker):
            os.remove(marker)  # a local build is not a release
        print(f"Copied the dataset tables from {args.dataset} (local build)")
    else:
        tag = args.release or C.DATASET_RELEASE
        have = open(marker).read().strip() if os.path.exists(marker) else None
        if have == tag:
            print(f"Using release {tag} already in {dataset_dir}")
        else:
            download_release(tag, dataset_dir)
    out_path = os.path.join(args.data_dir, "count", "all_count_processed.csv")

    surveys, counts, _, metadata = C.read_dataset(args.data_dir)
    print(
        f"Dataset: {len(surveys):,} surveys, {len(counts):,} counts "
        f"(built {metadata.get('built_at')}, defile-dataset {metadata.get('git_sha') or '?'})"
    )

    release = open(marker).read().strip() if os.path.exists(marker) else "local build"
    mc = C.build_model_counts(surveys, counts)
    checks = C.check_model_counts(mc)
    for c in checks:
        print(f"  [{c.status:4s}] {c.name}: {c.detail}")

    failed = any(c.status == "fail" for c in checks)
    if failed:
        print("Failed checks: model counts were not written.")
    elif args.dry_run:
        print(f"--dry-run: {len(mc.counts)} rows, not written.")
    else:
        tmp = out_path + ".tmp"
        mc.counts.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        unknown_path = os.path.join(args.data_dir, "count", C.UNKNOWN_DATES_FILE)
        mc.unknown_dates.to_csv(unknown_path, index=False)
        print(f"Wrote {unknown_path} ({len(mc.unknown_dates)} unknown species-days).")
        print(f"Wrote {out_path} ({len(mc.counts)} rows).")

    if args.report:
        inputs = {
            "Dataset": f"{dataset_dir}: {len(surveys):,} surveys, {len(counts):,} counts, "
            f"years {metadata.get('years')}",
            "defile-dataset": f"release {release}, built {metadata.get('built_at')}, commit "
            f"{metadata.get('git_sha') or '?'}, schema {metadata.get('consolidated_schema')}",
            "Output": out_path + (" (dry run, not written)" if args.dry_run else ""),
        }
        species = species_from_experiments(os.path.join(ROOT, "configs"))
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(render(mc, checks, counts, species, inputs))
        print(f"Wrote report {args.report}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
