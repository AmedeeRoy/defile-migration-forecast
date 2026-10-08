#!/usr/bin/env python3
"""Compares effort normalisations for the Explore page's annual series (issue #55).

For a few species: the time-of-day profile p(h | doy) at three dates; its stability between hourly
sheets (2014-2020) and Trektellen entry times (2021-2026), the only timed counts; and four annual series in the default window, each scaled to its own
2010-2025 mean: raw total, birds per counted hour, the uniform-profile index and the
species-profile index (Σ birds / Σ coverage). Writes `logs/qa/explore/effort_normalisation.pdf`
and prints a summary table. The decision it supports is in DECISIONS.md -> Explore.

Usage:
    python scripts/analyse_explore_effort.py
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rootutils  # noqa: E402

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.explore import export as E  # noqa: E402
from src.explore import profile as P  # noqa: E402
from src.metrics import ERA_EDGES  # noqa: E402

ROOT = rootutils.find_root(__file__, indicator=".project-root")
SPECIES = (
    "Black Kite",
    "European Honey Buzzard",
    "Red Kite",
    "Common Buzzard",
    "Common Wood Pigeon",
    "Eurasian Chaffinch",
)
REFERENCE_YEARS = (2010, 2025)  # each series is scaled to its mean over these years
# Timed counts exist from 2014 only: hourly sheets (spreadsheet, Naturalist) vs Trektellen entries.
STABILITY_PERIODS = ((2014, 2020), (2021, 2026))
PROFILE_DATES = ("08-15", "09-15", "10-15")
SERIES = ("raw", "per_hour", "uniform", "profile")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "logs", "qa", "explore", "effort_normalisation.pdf")
    )
    args = ap.parse_args(argv)

    surveys, counts, taxonomy, _, _ = E.read_release(args.data_dir)
    effort = E.build_effort(surveys)
    days = E.daily_counts(counts)
    hourly = E.hourly_counts(counts)
    ids = taxonomy.set_index("english_name")["taxon_id"]
    taxa = taxonomy[taxonomy["english_name"].isin(SPECIES)]
    profiles, source = P.build_profiles(taxa, days, hourly, effort)
    doy_grid = np.arange(P.PROFILE_DOY[0], P.PROFILE_DOY[1] + 1)
    light = P.daylight(doy_grid)
    c_uniform = P.coverage(effort, profiles["uniform"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(len(SPECIES), 3, figsize=(16, 3.2 * len(SPECIES)))
    summary = []
    for row, name in zip(axes, SPECIES):
        tid = ids[name]
        d, h = days[days["taxon_id"] == tid], hourly[hourly["taxon_id"] == tid]
        p = profiles[tid] if source[tid] == "own" else profiles[source[tid]]

        # profile at three dates
        for md, color in zip(PROFILE_DATES, ("C0", "C1", "C2")):
            i = pd.Timestamp(f"2025-{md}").dayofyear - P.PROFILE_DOY[0]
            row[0].plot(np.arange(24), p[i], color=color, label=md)
            row[0].plot(np.arange(24), profiles["uniform"][i], color=color, ls=":", lw=0.8)
        row[0].set(title=f"{name}: profile ({source[tid]}), dotted uniform", xlim=(4, 22))
        row[0].legend(fontsize=7)

        # stability between two periods
        stab = []
        for (y0, y1), ls in zip(STABILITY_PERIODS, ("-", "--")):
            sel = d["date"].dt.year.between(y0, y1)
            selh = h["date"].dt.year.between(y0, y1)
            samples = P.profile_samples(d[sel], h[selh], effort)
            pp = P.fit_profile(samples, doy_grid, light, label=f"{name} {y0}-{y1}")
            stab.append(pp)
            i = pd.Timestamp(f"2025-{PROFILE_DATES[1]}").dayofyear - P.PROFILE_DOY[0]
            row[1].plot(np.arange(24), pp[i], ls=ls, color="k", label=f"{y0}-{y1}")
        tv = 0.5 * np.abs(stab[0] - stab[1]).sum(axis=1)  # total variation per day
        row[1].set(
            title=f"{PROFILE_DATES[1]}; TV distance median {np.median(tv):.2f}", xlim=(4, 22)
        )
        row[1].legend(fontsize=7)

        # annual series
        a = P.annual_index(d, effort, P.coverage(effort, p)).set_index("year")
        u = P.annual_index(d, effort, c_uniform).set_index("year")
        s = pd.DataFrame(
            {
                "raw": a["birds"],
                "per_hour": a["birds"] / a["hours"],
                "uniform": u["index"],
                "profile": a["index"],
            }
        )
        ref = s.loc[REFERENCE_YEARS[0] : REFERENCE_YEARS[1]].mean()
        scaled = s / ref
        for col, style in zip(SERIES, ("C7", "C3", "C0", "k")):
            row[2].plot(scaled.index, scaled[col], color=style, lw=1.6 if col == "profile" else 1)
        ax2 = row[2].twinx()
        cov = (P.coverage(effort, p).groupby(effort["date"].dt.year).mean()).reindex(a.index)
        ax2.bar(a.index, cov, color="C2", alpha=0.15)
        ax2.set(ylim=(0, 1), ylabel="mean c per day")
        row[2].axvline(ERA_EDGES[0] - 0.5, color="grey", lw=0.5)
        row[2].set(title=f"{name}: annual, scaled to {REFERENCE_YEARS} mean", yscale="log")
        row[2].legend(SERIES, fontsize=7, loc="upper left")

        early = scaled.loc[: ERA_EDGES[0] - 1].median()
        summary.append(
            {
                "species": name,
                "source": source[tid],
                "timed_birds": int(h["count"].sum()),
                "tv_periods": round(float(np.median(tv)), 3),
                **{f"pre{ERA_EDGES[0]}_{k}": round(float(v), 2) for k, v in early.items()},
                "c_median_pre": round(float(cov.loc[: ERA_EDGES[0] - 1].median()), 2),
                "c_median_post2015": round(float(cov.loc[2015:].median()), 2),
            }
        )
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=70)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
