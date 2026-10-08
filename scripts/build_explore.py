#!/usr/bin/env python3
"""Builds the Explore export for defileViz from the defile-dataset release tables.

Reads `data/count/dataset/` (copied by `python scripts/build_counts.py --dataset <dir>`) and writes
`data/explore/`: `manifest.json`, `taxa.json`, `effort.json`, `reports.json` and
`species/<taxon_id>.json`, with French names from the eBird taxonomy (downloaded once, no key).
`src/explore/` documents each file; every value is an
aggregation of the release, with no model processing. Copy the folder to defileViz's
`public/data/explore/` to publish it.

Usage:
    python scripts/build_explore.py
    python scripts/build_explore.py --out ../defileViz/public/data/explore
"""

import argparse
import os
import shutil
import subprocess
import urllib.request

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.explore import export as E  # noqa: E402
from src.explore import profile as P  # noqa: E402

ROOT = rootutils.find_root(__file__, indicator=".project-root")


def write(path: str, obj) -> int:
    with open(path, "w", encoding="utf-8") as f:
        f.write(E.dumps(obj))
    return os.path.getsize(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "explore"))
    ap.add_argument(
        "--refresh-names", action="store_true", help="download the eBird taxonomy again"
    )
    args = ap.parse_args(argv)

    names_path = os.path.join(args.data_dir, E.EBIRD_TAXONOMY_FILE)
    if args.refresh_names or not os.path.exists(names_path):
        urllib.request.urlretrieve(E.EBIRD_TAXONOMY_URL, names_path)
        print(f"Downloaded the eBird taxonomy (French names) to {names_path}")
    ebird = pd.read_csv(names_path)

    surveys, counts, taxonomy, reports, metadata = E.read_release(args.data_dir)
    effort = E.build_effort(surveys)
    days = E.daily_counts(counts)
    hourly = E.hourly_counts(counts)
    taxa = E.build_taxa(taxonomy, days, ebird)
    profiles, source = P.build_profiles(taxa, days, hourly, effort)
    taxa["profile"] = taxa["taxon_id"].map(source)
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.join(args.out, "species"))
    sizes = {
        "manifest.json": write(
            os.path.join(args.out, "manifest.json"), E.manifest(metadata, days, effort, git_sha)
        ),
        "taxa.json": write(os.path.join(args.out, "taxa.json"), E.records(taxa)),
        "effort.json": write(
            os.path.join(args.out, "effort.json"),
            {"days": E.columns(effort), "annual": E.records(E.annual_effort(effort))},
        ),
        "reports.json": write(
            os.path.join(args.out, "reports.json"),
            E.records(reports[reports["category"] != E.REPORT_SPECIES]),
        ),
    }
    species_bytes = 0
    texts = reports[reports["category"] == E.REPORT_SPECIES]
    starts = taxa.set_index("taxon_id")["start_year"]
    for taxon_id, d in days.groupby("taxon_id"):
        profile = profiles[taxon_id if source[taxon_id] == "own" else source[taxon_id]]
        c = P.coverage(effort, profile)
        start = starts[taxon_id]
        d = P.adjust_days(d.drop(columns="taxon_id"), effort, c, start)
        a = E.annual(d).merge(
            P.annual_index(d, effort, c, start)[["year", "c_mean", "index"]], on="year", how="left"
        )
        h = hourly[hourly["taxon_id"] == taxon_id].drop(columns="taxon_id")
        t = texts[texts["key"] == taxon_id][["year", "text"]]
        species_bytes += write(
            os.path.join(args.out, "species", f"{taxon_id}.json"),
            {
                "taxon_id": taxon_id,
                "days": E.columns(d),
                "hourly": E.columns(h),
                "annual": E.records(a),
                "profile": {"source": source[taxon_id], **P.profile_table(profile)},
                "reports": E.records(t),
            },
        )
    sizes[f"species/ ({days['taxon_id'].nunique()} files)"] = species_bytes

    print(f"Explore export -> {args.out}")
    for name, n in sizes.items():
        print(f"  {name:28s} {n / 1e3:9.1f} kB")
    print("  tiers:", taxa["tier"].value_counts().to_dict())
    print("  profiles:", taxa["profile"].value_counts().to_dict())
    full = taxa[taxa["tier"] == "full"]
    print("  start years (full tier):", full["start_year"].value_counts().to_dict())
    missing = taxa.loc[taxa["french_name"].isna(), "english_name"].tolist()
    if missing:
        print(f"  no French name ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
