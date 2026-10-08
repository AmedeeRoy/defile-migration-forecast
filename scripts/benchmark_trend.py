#!/usr/bin/env python3
"""Benchmarks the Explore trend models (`src.explore.trend`: GAM vs Gaussian process, with the
season-only reference) on a few taxa, before either is exported (issue #55).

Two tests, each on the taxon's years from its start year to the last complete season:

1. Gap transplant (gap filling, what Explore shows): a well-counted recent year (mean coverage of
   counted window days >= `TARGET_MIN_COVERAGE`) is given an older year's gaps, hour by hour and
   day by day (`DONOR_YEARS`), refitted with the full fit's hyperparameters, and its hidden birds
   filled. The estimate (birds kept + the posterior predictive of the hidden coverage) is compared
   with the birds the year actually counted. Estimates are posterior predictive medians: for
   irruptive taxa the mean is a tail. Also: the ratio estimator, the Explore index applied
   to the hidden coverage (birds kept / coverage kept, same day or not).
2. Recent years (trend extrapolation, what the forecast would use): fitted on the seasons up to
   `RECENT_FROM - 1` (hyperparameters refitted), the next seasons' counted birds predicted from the
   model alone, as annual totals and as each day's log predictive density.

Writes `logs/qa/explore/trend_benchmark.pdf` (one page per taxon and a summary), the per-trial
tables as CSV next to it, and prints the summary.

Usage:
    python scripts/benchmark_trend.py
    python scripts/benchmark_trend.py --taxa "Red Kite" --workers 1
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rootutils  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.explore import export as E  # noqa: E402
from src.explore import profile as P  # noqa: E402
from src.explore import trend as T  # noqa: E402

ROOT = rootutils.find_root(__file__, indicator=".project-root")
TAXA = (
    "Black Kite",
    "European Honey Buzzard",
    "Red Kite",
    "Common Buzzard",
    "Osprey",
    "All pigeons (Columba)",
    "Eurasian Chaffinch",
)
TARGET_MIN_COVERAGE = 0.8
TARGET_FROM = 2014  # timed counts, so the birds of hidden hours are known
DONOR_YEARS = (1993, 2013)  # their gaps (hours and days not counted) are transplanted
RECENT_FROM = 2023
METHODS = ("ratio", *T.VARIANTS)
COLORS = {"ratio": "C7", "season": "C2", "gam": "C0", "gp": "C3"}


# --- data -------------------------------------------------------------------


def load(data_dir: str, names) -> dict:
    """Everything the trials need, per taxon: frame, profile, hourly birds, effort, start year."""
    surveys, counts, taxonomy, _, _ = E.read_release(data_dir)
    effort = E.build_effort(surveys)
    days = E.daily_counts(counts)
    hourly = E.hourly_counts(counts)
    taxonomy, days, hourly = E.add_combined(taxonomy, days, hourly)
    taxa = E.build_taxa(taxonomy, days)
    profiles, source = P.build_profiles(taxa, days, hourly, effort)
    last_year = int(days["date"].max().year) - len(E.partial_years(days))
    out = {}
    for name in names:
        t = taxa[taxa["english_name"] == name].iloc[0]
        tid = t["taxon_id"]
        profile = profiles[tid if source[tid] == "own" else source[tid]]
        out[name] = {
            "taxon_id": tid,
            "start": int(t["start_year"]),
            "last": last_year,
            "profile": profile,
            "days": days[days["taxon_id"] == tid].drop(columns="taxon_id"),
            "hourly": hourly[hourly["taxon_id"] == tid].drop(columns="taxon_id"),
            "effort": effort,
        }
    return out


def hourly_effort(effort: pd.DataFrame) -> pd.Series:
    """Hours counted per local clock hour, by date, on counted days."""
    e = effort[effort["state"] == "counted"]
    return pd.Series(list(e["hourly"]), index=e["date"])


# --- test 1: gap transplant -------------------------------------------------


def transplant(target: pd.DataFrame, donor_year: int, ctx: dict) -> pd.DataFrame:
    """The target year's rows with the donor year's gaps: `c` and `y` become the coverage and birds
    kept, `c_target` the coverage the target counted.

    Hidden birds are known from their hour; untimed birds are kept in proportion to the coverage
    kept.
    """
    eff = hourly_effort(ctx["effort"])
    hourly = {d: h.set_index("hour")["count"] for d, h in ctx["hourly"].groupby("date")}
    zero = np.zeros(E.HOURS)
    rows = []
    for _, r in target.iterrows():
        tgt = eff.get(r["date"], zero) if r["c"] > 0 else zero
        donor_date = r["date"].replace(year=donor_year)
        kept = np.minimum(tgt, eff.get(donor_date, zero))
        p = ctx["profile"][int(np.clip(r["doy"], *P.PROFILE_DOY)) - P.PROFILE_DOY[0]]
        c_kept = float((p * kept).sum())
        birds_h = np.zeros(E.HOURS)
        if r["y"] > 0 and r["date"] in hourly:
            h = hourly[r["date"]]
            birds_h[h.index.to_numpy()] = h.to_numpy()
        share = np.divide(kept, tgt, out=np.zeros(E.HOURS), where=tgt > 0)
        untimed = max(r["y"] - birds_h.sum(), 0.0)
        y_kept = (birds_h * share).sum() + (untimed * c_kept / r["c"] if r["c"] > 0 else 0.0)
        extra = r["extra"]
        c_target = r["c"]
        if c_kept < T.MIN_COVERAGE:  # as `model_frame` would: kept as counted, not modelled
            extra, c_target = extra + y_kept, c_target - c_kept
            c_kept = y_kept = 0.0
        rows.append({"c": c_kept, "y": y_kept, "extra": extra, "c_target": c_target})
    out = target.drop(columns=["c", "y", "extra"]).reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def gap_trials(name: str, ctx: dict, frame: pd.DataFrame, fits: dict) -> pd.DataFrame:
    years = np.arange(ctx["start"], ctx["last"] + 1)
    doy_range = (int(frame["doy"].min()), int(frame["doy"].max()))
    counted = frame[frame["c"] > 0].groupby("year")["c"].mean()
    targets = [
        y
        for y in counted.index
        if y >= max(TARGET_FROM, ctx["start"]) and counted[y] >= TARGET_MIN_COVERAGE
    ]
    rows = []
    for ty in targets:
        target = frame[frame["year"] == ty]
        truth = target["y"].sum() + target["extra"].sum()
        for dy in range(DONOR_YEARS[0], DONOR_YEARS[1] + 1):
            masked = transplant(target, dy, ctx)
            base = {
                "taxon": name,
                "target": ty,
                "donor": dy,
                "truth": truth,
                "kept": masked["y"].sum() + masked["extra"].sum(),
                "hidden_coverage": 1 - masked["c"].sum() / masked["c_target"].sum(),
            }
            ratio = masked["y"].sum() / max(masked["c"].sum(), 1e-9)
            est = masked["y"].sum() + masked["extra"].sum()
            est += ratio * (masked["c_target"] - masked["c"]).sum()
            rows.append({**base, "method": "ratio", "estimate": est})
            others = frame[frame["year"] != ty]
            train = pd.concat([others, masked[others.columns]], ignore_index=True)
            for v, full in fits.items():
                f = T.fit(v, train, years, doy_range, hyper=full.hyper)
                draws = T.fill_draws(f, masked, masked["c_target"].to_numpy(), seed=dy)
                total = draws.sum(axis=1)
                q = np.quantile(total, [0.025, 0.1, 0.5, 0.9, 0.975])
                rows.append(
                    {
                        **base,
                        "method": v,
                        "estimate": q[2],
                        "q2.5": q[0],
                        "q10": q[1],
                        "q90": q[3],
                        "q97.5": q[4],
                    }
                )
    return pd.DataFrame(rows)


# --- test 2: recent years ---------------------------------------------------


def recent_trials(name: str, ctx: dict, frame: pd.DataFrame) -> pd.DataFrame:
    years = np.arange(ctx["start"], ctx["last"] + 1)
    doy_range = (int(frame["doy"].min()), int(frame["doy"].max()))
    future = frame["year"] >= RECENT_FROM
    train = frame.assign(c=frame["c"].where(~future, 0.0), y=frame["y"].where(~future, 0.0))
    test = frame[future & (frame["c"] > 0)]
    rows = []
    for v in T.VARIANTS:
        f = T.fit(v, train, years, doy_range)
        unseen = test.assign(c=0.0, y=0.0)
        draws = T.fill_draws(f, unseen, test["c"].to_numpy(), seed=1)
        beta = f.draws(T.DRAWS, np.random.default_rng(2))
        mu = np.exp(T.eta_draws(f, test["year"], test["doy"], beta)) * test["c"].to_numpy()
        ll = logsumexp(T.nb_loglik(test["y"].to_numpy(), mu, f.theta), axis=0) - np.log(len(mu))
        for yr in sorted(test["year"].unique()):
            sel = (test["year"] == yr).to_numpy()
            total = draws[:, sel].sum(axis=1)
            q = np.quantile(total, [0.025, 0.1, 0.5, 0.9, 0.975])
            rows.append(
                {
                    "taxon": name,
                    "method": v,
                    "year": yr,
                    "truth": test["y"].to_numpy()[sel].sum(),
                    "estimate": q[2],
                    "q2.5": q[0],
                    "q10": q[1],
                    "q90": q[3],
                    "q97.5": q[4],
                    "log_score": ll[sel].mean(),
                    "days": int(sel.sum()),
                }
            )
    return pd.DataFrame(rows)


# --- one taxon --------------------------------------------------------------


def run_taxon(args) -> dict:
    name, ctx = args
    t0 = time.time()
    frame = T.model_frame(
        ctx["days"],
        ctx["effort"],
        P.coverage(ctx["effort"], ctx["profile"]),
        ctx["start"],
        ctx["last"],
    )
    years = np.arange(ctx["start"], ctx["last"] + 1)
    doy_range = (int(frame["doy"].min()), int(frame["doy"].max()))
    fits = {v: T.fit(v, frame, years, doy_range) for v in T.VARIANTS}
    annual = pd.concat([T.annual_totals(f, frame).assign(method=v) for v, f in fits.items()])
    mid = int((ctx["start"] + ctx["last"]) / 2)
    seasons = pd.concat(
        [
            T.season_curve(f, yr).assign(method=v, year=yr)
            for v, f in fits.items()
            for yr in (ctx["start"], mid, ctx["last"])
        ]
    )
    peaks = pd.concat(
        [T.peak_doy(f, years).assign(method=v) for v, f in fits.items() if v != "season"]
    )
    info = pd.DataFrame(
        [
            {
                "taxon": name,
                "method": v,
                "log_marginal": f.log_marginal,
                "theta": f.theta,
                **{
                    f"h_{k}": round(float(np.exp(x)), 4)
                    for k, x in f.hyper.items()
                    if k != "theta"
                },
            }
            for v, f in fits.items()
        ]
    )
    gaps = gap_trials(name, ctx, frame, fits)
    recent = recent_trials(name, ctx, frame)
    print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
    return {
        "name": name,
        "start": ctx["start"],
        "annual": annual.assign(taxon=name),
        "seasons": seasons.assign(taxon=name),
        "peaks": peaks.assign(taxon=name),
        "info": info,
        "gaps": gaps,
        "recent": recent,
    }


# --- summary and figures ----------------------------------------------------


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Per taxon and method: median |log(estimate / truth)|, median log ratio (bias) and how often
    the 80% and 95% intervals hold the truth."""
    d = df[df["truth"] > 0].assign(err=lambda x: np.log(x["estimate"] / x["truth"]))
    d["in80"] = (d["truth"] >= d.get("q10")) & (d["truth"] <= d.get("q90"))
    d["in95"] = (d["truth"] >= d.get("q2.5")) & (d["truth"] <= d.get("q97.5"))
    has_q = d.get("q10").notna() if "q10" in d else False
    g = d.groupby(["taxon", "method"])
    s = pd.DataFrame(
        {
            "abs_log_err": g["err"].apply(lambda x: x.abs().median()),
            "bias": g["err"].median(),
            "cover80": d[has_q].groupby(["taxon", "method"])["in80"].mean(),
            "cover95": d[has_q].groupby(["taxon", "method"])["in95"].mean(),
            "n": g.size(),
        }
    )
    return s


def page(pdf, r: dict, gaps_score: pd.DataFrame):
    name = r["name"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    a = r["annual"]
    obs = a[a["method"] == "gam"]
    ax[0, 0].bar(obs["year"], obs["observed"], color="0.85", label="counted")
    for v in T.VARIANTS:
        x = a[a["method"] == v]
        off = {"season": -0.25, "gam": 0, "gp": 0.25}[v]
        ax[0, 0].errorbar(
            x["year"] + off,
            x["q50"],
            yerr=[x["q50"] - x["q2.5"], x["q97.5"] - x["q50"]],
            fmt="o",
            ms=3,
            lw=0.8,
            color=COLORS[v],
            label=f"{v} gap-filled",
        )
        if v != "season":
            ax[0, 0].plot(x["year"], x["smooth"], color=COLORS[v], lw=1.5)
            ax[0, 0].fill_between(
                x["year"], x["smooth_q2.5"], x["smooth_q97.5"], color=COLORS[v], alpha=0.12
            )
    ax[0, 0].set(yscale="log", title=f"{name} (from {r['start']}): window total, smooth trend")
    ax[0, 0].legend(fontsize=7)

    s = r["seasons"]
    for (v, yr), x in s[s["method"] != "season"].groupby(["method", "year"]):
        ls = {0: ":", 1: "--", 2: "-"}[sorted(s["year"].unique()).index(yr)]
        ax[0, 1].plot(x["doy"], x["mid"], color=COLORS[v], ls=ls, label=f"{v} {yr}")
    x = s[(s["method"] == "season")].drop_duplicates("doy")
    ax[0, 1].plot(x["doy"], x["mid"], color=COLORS["season"], lw=0.8, label="season only")
    ax[0, 1].set(title="expected birds per full day", xlabel="day of year")
    ax[0, 1].legend(fontsize=7)

    p = r["peaks"]
    for v, x in p.groupby("method"):
        ax[1, 0].plot(x["year"], x["mid"], color=COLORS[v], label=v)
        ax[1, 0].fill_between(x["year"], x["lo"], x["hi"], color=COLORS[v], alpha=0.15)
    ax[1, 0].set(title="median passage date (80% band)", ylabel="day of year")
    ax[1, 0].legend(fontsize=7)

    g = r["gaps"][r["gaps"]["truth"] > 0]
    data = [
        np.log(g.loc[g["method"] == m, "estimate"] / g.loc[g["method"] == m, "truth"])
        for m in METHODS
    ]
    ax[1, 1].boxplot(data, tick_labels=METHODS, showfliers=False)
    ax[1, 1].axhline(0, color="k", lw=0.5)
    sc = gaps_score.loc[name]
    txt = "\n".join(
        f"{m:6s} |err| {sc.loc[m, 'abs_log_err']:.3f}  bias {sc.loc[m, 'bias']:+.3f}"
        + (
            f"  80% {sc.loc[m, 'cover80']:.2f}  95% {sc.loc[m, 'cover95']:.2f}"
            if m != "ratio"
            else ""
        )
        for m in METHODS
    )
    ax[1, 1].text(
        0.02, 0.02, txt, transform=ax[1, 1].transAxes, fontsize=7, family="monospace", va="bottom"
    )
    ax[1, 1].set(title="test 1, gap transplant: log(estimate / counted)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "logs", "qa", "explore", "trend_benchmark.pdf")
    )
    ap.add_argument("--taxa", nargs="+", default=list(TAXA))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    ctxs = load(args.data_dir, args.taxa)
    print(f"Loaded {len(ctxs)} taxa; fitting...", flush=True)
    jobs = list(ctxs.items())
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(run_taxon, jobs))
    else:
        results = [run_taxon(j) for j in jobs]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stem = args.out.removesuffix(".pdf")
    tables = {
        k: pd.concat([r[k] for r in results], ignore_index=True)
        for k in ("annual", "info", "gaps", "recent", "peaks")
    }
    for k, df in tables.items():
        df.to_csv(f"{stem}_{k}.csv", index=False)
    gaps_score = score(tables["gaps"])
    recent_score = score(tables["recent"]).join(
        tables["recent"].groupby(["taxon", "method"])["log_score"].mean()
    )
    info = tables["info"].set_index(["taxon", "method"])[["log_marginal", "theta"]]
    with PdfPages(args.out) as pdf:
        for r in results:
            page(pdf, r, gaps_score)
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.axis("off")
        text = (
            "TEST 1, gap transplant\n"
            + gaps_score.round(3).to_string()
            + "\n\nTEST 2, recent years\n"
            + recent_score.round(3).to_string()
            + "\n\nFits\n"
            + info.round(2).to_string()
        )
        ax.text(0, 1, text, family="monospace", fontsize=6, va="top")
        pdf.savefig(fig)
        plt.close(fig)
    pd.set_option("display.width", 200)
    print("\nTEST 1, gap transplant\n", gaps_score.round(3).to_string())
    print(
        "\n",
        gaps_score.groupby("method")[["abs_log_err", "bias", "cover80", "cover95"]]
        .mean()
        .round(3),
    )
    print("\nTEST 2, recent years\n", recent_score.round(3).to_string())
    print(
        "\n",
        recent_score.groupby("method")[["abs_log_err", "bias", "cover80", "cover95", "log_score"]]
        .mean()
        .round(3),
    )
    print("\nFits\n", info.round(2).to_string())
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
