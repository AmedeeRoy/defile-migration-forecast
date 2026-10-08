#!/usr/bin/env python3
"""Builds the Explore export for defileViz from the defile-dataset release tables.

Reads `data/count/dataset/` (copied by `python scripts/build_counts.py --dataset <dir>`) and writes
`data/explore/`: `manifest.json`, `taxa.json`, `effort.json`, `reports.json` and
`species/<taxon_id>.json`, with French names from the eBird taxonomy (downloaded once, no key).
`src/explore/` documents each file. Raw values are aggregations of the release; the species files'
effort-adjusted values and `trend` (the GAM, about 4 min for the full tier on 11 cores) are
labelled as such. Copy the folder to defileViz's `public/data/explore/` to publish it.

Usage:
    python scripts/build_explore.py
    python scripts/build_explore.py --skip-trend   # seconds, without the trend model
    python scripts/build_explore.py --out ../defileViz/public/data/explore
"""

import argparse
import os
import shutil
import subprocess
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.explore import export as E  # noqa: E402
from src.explore import profile as P  # noqa: E402
from src.explore import trend as T  # noqa: E402

ROOT = rootutils.find_root(__file__, indicator=".project-root")


def write(path: str, obj) -> int:
    with open(path, "w", encoding="utf-8") as f:
        f.write(E.dumps(obj))
    return os.path.getsize(path)


def trend_of(args) -> tuple[str, dict]:
    taxon_id, kwargs = args
    t = T.taxon_trend(**kwargs)
    return taxon_id, {
        "model": t["model"],
        "first_year": t["first_year"],
        "last_year": t["last_year"],
        "theta": t["theta"],
        "kappa": t["kappa"],
        "annual": E.records(t["annual"]),
        "passage": E.records(t["passage"]),
        "season": t["season"],
    }


def trends(taxa, days, hourly, effort, profiles, source, last_year: int, workers: int) -> dict:
    """The GAM trend (`src.explore.trend.taxon_trend`) of every full-tier taxon of a rank in
    `TREND_RANKS`, in parallel."""
    full = taxa[(taxa["tier"] == "full") & taxa["taxon_rank"].isin(T.TREND_RANKS)]
    jobs = [
        (
            t["taxon_id"],
            {
                "days": days[days["taxon_id"] == t["taxon_id"]].drop(columns="taxon_id"),
                "hourly": hourly[hourly["taxon_id"] == t["taxon_id"]].drop(columns="taxon_id"),
                "effort": effort,
                "profile": profiles[
                    t["taxon_id"] if source[t["taxon_id"]] == "own" else source[t["taxon_id"]]
                ],
                "first_year": int(t["start_year"]),
                "last_year": last_year,
            },
        )
        for _, t in full.iterrows()
    ]
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(trend_of, jobs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "explore"))
    ap.add_argument(
        "--refresh-names", action="store_true", help="download the eBird taxonomy again"
    )
    ap.add_argument("--skip-trend", action="store_true", help="no trend model (fast build)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
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
    taxonomy, days, hourly = E.add_combined(taxonomy, days, hourly)
    taxa = E.build_taxa(taxonomy, days, ebird)
    profiles, source = P.build_profiles(taxa, days, hourly, effort)
    taxa["profile"] = taxa["taxon_id"].map(source)
    last_year = int(days["date"].max().year) - len(E.partial_years(days))
    trend = (
        {}
        if args.skip_trend
        else trends(taxa, days, hourly, effort, profiles, source, last_year, args.workers)
    )
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
                "trend": trend.get(taxon_id),
            },
        )
    sizes[f"species/ ({days['taxon_id'].nunique()} files)"] = species_bytes

    print(f"Explore export -> {args.out}")
    for name, n in sizes.items():
        print(f"  {name:28s} {n / 1e3:9.1f} kB")
    print("  tiers:", taxa["tier"].value_counts().to_dict())
    print("  profiles:", taxa["profile"].value_counts().to_dict())
    print(f"  trends: {len(trend)} taxa (full tier, {'/'.join(T.TREND_RANKS)}), to {last_year}")
    full = taxa[taxa["tier"] == "full"]
    print("  start years (full tier):", full["start_year"].value_counts().to_dict())
    combined = taxa[taxa["taxon_rank"] == E.COMBINED_RANK]
    print("  combined:", dict(zip(combined["english_name"], combined["start_year"])))
    missing = taxa.loc[taxa["french_name"].isna(), "english_name"].tolist()
    if missing:
        print(f"  no French name ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
