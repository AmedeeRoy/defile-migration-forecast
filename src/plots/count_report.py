"""Self-contained HTML QA report of the count processing (`src.data.counts`).

Rendering only: every number shown comes from the `ModelCounts` (output, `ProcessingLog` steps) and
the `Check`s computed in `src.data.counts`, so this module can change freely without affecting the
data. Built by `scripts/build_counts.py`.

Times in tables are shown in local time (Europe/Paris), as in the raw files, so a row can be looked
up there directly.
"""

import base64
import datetime as dt
import html
import io
import re

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.data.counts import (
    MIN_PERIOD_DURATION,
    SOURCES,
    TIMEZONE,
    Check,
    ModelCounts,
    source_of,
)
from src.plots.panels import C_OBS, C_PHEN, C_PRED

MAX_TABLE_ROWS = 200
SOURCE_COLORS = {"historical": C_PHEN, "trektellen": C_PRED}
STATUS_COLORS = {"pass": "#1a7f37", "warn": "#9a6700", "fail": "#cf222e"}
EXAMPLE_DAYS_PER_YEAR = 1
EXAMPLE_SEED = 0

CSS = """
:root { --fg:#1f2328; --muted:#59636e; --bg:#ffffff; --line:#d1d9e0; --soft:#f6f8fa; }
body { font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg);
       background: var(--bg); max-width: 1180px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 24px; margin-bottom: 4px; } h2 { margin-top: 40px; border-bottom: 1px solid
var(--line); padding-bottom: 4px; } h3 { margin-top: 24px; }
.muted { color: var(--muted); }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 12.5px; }
th, td { border: 1px solid var(--line); padding: 3px 8px; text-align: right; vertical-align: top; }
th { background: var(--soft); } td.l, th.l { text-align: left; }
.table-wrap { overflow-x: auto; max-width: 100%; }
.badge { display: inline-block; min-width: 40px; text-align: center; color: #fff;
         border-radius: 10px; padding: 0 8px; font-size: 12px; font-weight: 600; }
details { margin: 4px 0 12px; } summary { cursor: pointer; color: var(--muted); }
img { max-width: 100%; }
code { background: var(--soft); padding: 1px 4px; border-radius: 4px; }
"""


def _fig_to_img(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return df


def _table(df: pd.DataFrame, index=False, max_rows=MAX_TABLE_ROWS) -> str:
    if df is None or df.empty:
        return '<p class="muted">No rows.</p>'
    more = len(df) - max_rows
    out = _local(df.head(max_rows)).to_html(index=index, na_rep="", border=0, escape=True)
    note = f'<p class="muted">… and {more} more rows.</p>' if more > 0 else ""
    return f'<div class="table-wrap">{out}</div>{note}'


def _badge(status: str) -> str:
    return f'<span class="badge" style="background:{STATUS_COLORS[status]}">{status}</span>'


def _species_label(rows: pd.DataFrame) -> pd.Series:
    return rows["species"].fillna(rows.get("taxon_name_original", "(no species)"))


def _breakdown(rows: pd.DataFrame) -> str:
    """Rows and birds per species and per year, for a step's rows."""
    if rows.empty or "count" not in rows or "species" not in rows:
        return ""
    r = rows.assign(_species=_species_label(rows), _year=rows["date"].dt.year)
    by_sp = (
        r.groupby("_species")["count"]
        .agg(rows="size", birds="sum")
        .sort_values("birds", ascending=False)
        .rename_axis("species")
        .reset_index()
    )
    by_y = r.groupby("_year")["count"].agg(rows="size", birds="sum").rename_axis("year")
    return (
        '<div style="display:flex;gap:24px;flex-wrap:wrap">'
        f"<div><b>By species</b>{_table(by_sp, max_rows=20)}</div>"
        f"<div><b>By year</b>{_table(by_y.reset_index(), max_rows=40)}</div></div>"
    )


# --- sections --------------------------------------------------------------------------


def _section_checks(checks: list[Check]) -> str:
    parts = ["<h2>Checks</h2>"]
    parts.append(
        '<table><tr><th class="l">Check</th><th>Status</th><th class="l">Detail</th></tr>'
        + "".join(
            f'<tr><td class="l">{html.escape(c.name)}</td><td>{_badge(c.status)}</td>'
            f'<td class="l">{html.escape(c.detail)}</td></tr>'
            for c in checks
        )
        + "</table>"
    )
    for c in checks:
        if c.rows.empty:
            continue
        extra = ""
        if c.name == "Surveyed hours with no row":
            local = c.rows["start"].dt.tz_convert(TIMEZONE)
            extra = (
                "<b>By local hour</b>"
                + _table(local.dt.hour.value_counts().sort_index().rename("slots").to_frame().T)
                + "<b>By year</b>"
                + _table(local.dt.year.value_counts().sort_index().rename("slots").to_frame().T)
            )
        parts.append(
            f"<details><summary>{html.escape(c.name)}: {len(c.rows)} rows</summary>"
            f"{extra}{_table(c.rows)}</details>"
        )
    return "".join(parts)


def _section_steps(mc: ModelCounts) -> str:
    parts = [
        "<h2>Processing steps</h2>",
        '<p class="muted">In order of application, per source. <i>Birds</i> is shown for '
        "steps that remove or move birds; open a step to see which species, years and rows "
        "it touched, and check them against the raw files.</p>",
    ]
    for src in SOURCES:
        steps = [s for s in mc.log.steps if s.source == src]
        parts.append(f"<h3>{src.capitalize()}</h3>")
        parts.append(
            '<table><tr><th>#</th><th class="l">Step</th><th class="l">Action</th>'
            '<th class="l">Rule</th><th>Rows</th><th>Birds</th></tr>'
        )
        for i, s in enumerate(steps, 1):
            birds = f"{s.birds:,.0f}" if s.action in ("removed", "modified") else ""
            parts.append(
                f'<tr><td>{i}</td><td class="l">{html.escape(s.name)}</td>'
                f'<td class="l">{s.action}</td><td class="l">{html.escape(s.rule)}</td>'
                f"<td>{s.n_rows:,}</td><td>{birds}</td></tr>"
            )
        parts.append("</table>")
        for i, s in enumerate(steps, 1):
            if s.rows.empty or s.action == "merged":
                continue
            body = _breakdown(s.rows) if s.action != "added" else ""
            parts.append(
                f"<details><summary>{i}. {html.escape(s.name)} ({s.action}, "
                f"{s.n_rows:,} rows)</summary>{body}<b>Rows</b>"
                f"{_table(s.rows)}</details>"
            )
    return "".join(parts)


def _section_balance(mc: ModelCounts) -> str:
    rows = []
    for src in SOURCES:
        c = mc.counts[source_of(mc.counts["date"], _trektellen_from(mc)) == src]
        df = pd.DataFrame(
            {
                "read": mc.raw_birds[src],
                "removed": mc.log.removed_birds(src),
                "output": c.groupby(c["date"].dt.year)["count"].sum(),
            }
        ).fillna(0)
        df["removed %"] = (100 * df["removed"] / df["read"].where(df["read"] > 0)).round(2)
        rows.append(df.assign(source=src))
    df = pd.concat(rows).rename_axis("year").reset_index()
    df = df[["source", "year", "read", "removed", "removed %", "output"]]
    return (
        "<h2>Birds read vs. output, per year</h2>"
        '<p class="muted">All species. Trektellen birds are <code>direction1</code> '
        "(main migration direction) only, as in the dataset's <code>count</code>.</p>"
        + _table(df.astype({"read": int, "removed": int, "output": int}), max_rows=100)
    )


def _trektellen_from(mc: ModelCounts) -> int:
    return int(mc.raw_birds["trektellen"].index.min())


def _periods(mc: ModelCounts) -> pd.DataFrame:
    p = mc.counts[["date", "start", "end"]].drop_duplicates()
    p = p.assign(
        hours=(p["end"] - p["start"]).dt.total_seconds() / 3600,
        source=source_of(p["date"], _trektellen_from(mc)),
        year=p["date"].dt.year,
        doy=p["date"].dt.day_of_year,
    )
    return p


def _section_effort(mc: ModelCounts) -> str:
    p = _periods(mc)
    years = np.arange(p["year"].min(), p["year"].max() + 1)

    fig = Figure(figsize=(11, 3.6))
    ax = fig.subplots()
    hourly = p[p["hours"] <= 1].groupby("year")["hours"].sum().reindex(years, fill_value=0)
    longer = p[p["hours"] > 1].groupby("year")["hours"].sum().reindex(years, fill_value=0)
    ax.bar(years, hourly, color=C_PRED, label="in periods ≤ 1 h (hourly resolution)")
    ax.bar(years, longer, bottom=hourly, color="#bbbbbb", label="in periods > 1 h")
    ax.set_ylabel("surveyed hours")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig_bar = _fig_to_img(fig)

    grid = (p.groupby(["year", "doy"])["hours"].sum().unstack("doy").reindex(index=years)).reindex(
        columns=np.arange(1, 367)
    )
    fig = Figure(figsize=(11, 6))
    ax = fig.subplots()
    im = ax.imshow(
        grid.values,
        aspect="auto",
        cmap="viridis",
        interpolation="none",
        extent=(0.5, 366.5, years[-1] + 0.5, years[0] - 0.5),
    )
    ax.set_xlabel("day of year")
    ax.set_ylabel("year")
    fig.colorbar(im, ax=ax, label="surveyed hours per day")
    fig_grid = _fig_to_img(fig)

    fig = Figure(figsize=(11, 3))
    axs = fig.subplots(1, 2)
    for ax, src in zip(axs, SOURCES):
        h = p.loc[p["source"] == src, "hours"]
        ax.hist(h * 60, bins=np.arange(0, 16 * 60 + 15, 15), color=SOURCE_COLORS[src])
        ax.set_yscale("log")
        ax.set_title(f"{src}: {len(h):,} periods", fontsize=10)
        ax.set_xlabel("period duration (min)")
        ax.axvline(MIN_PERIOD_DURATION.total_seconds() / 60, color=C_OBS, lw=0.8, ls="--")
    fig_hist = _fig_to_img(fig)

    return (
        "<h2>Survey effort</h2>"
        "<h3>Surveyed hours per year</h3>"
        + fig_bar
        + "<h3>Surveyed hours per day</h3>"
        + fig_grid
        + "<h3>Period durations</h3>"
        '<p class="muted">Dashed: the 10-min minimum applied to Trektellen periods.</p>' + fig_hist
    )


def _section_species(mc: ModelCounts, species: list[str]) -> str:
    c = mc.counts[mc.counts["species"].isin(species)]
    t = (
        c.groupby([c["date"].dt.year.rename("year"), "species"])["count"]
        .sum()
        .unstack("species")
        .reindex(columns=species)
        .fillna(0)
        .astype(int)
    )
    return (
        "<h2>Modelled species, birds per year</h2>"
        '<p class="muted">The species in <code>configs/experiment/</code>. A sudden jump or '
        "gap between adjacent years is worth checking against the raw files.</p>"
        + _table(t.reset_index(), max_rows=100)
    )


def _section_examples(mc: ModelCounts, sightings: pd.DataFrame, removed: pd.DataFrame) -> str:
    """`sightings`: the dataset's Trektellen observations."""
    """A few Trektellen days split into hours: output periods vs.

    raw sighting timestamps.
    """
    windows = mc.split_windows["trektellen"]
    if windows.empty:
        return ""
    rng = np.random.default_rng(EXAMPLE_SEED)
    days = []
    for _, g in windows.groupby(windows["date"].dt.year):
        days += list(rng.choice(g["date"].unique(), EXAMPLE_DAYS_PER_YEAR, replace=False))
    fig = Figure(figsize=(11, 1.7 * len(days)))
    axs = np.atleast_1d(fig.subplots(len(days), 1))
    for ax, day in zip(axs, days):
        day = pd.Timestamp(day)
        out = mc.counts[mc.counts["date"] == day]
        per = out.groupby(["start", "end"])["count"].sum().reset_index()
        for r in per.itertuples():
            a, b = r.start.tz_convert(TIMEZONE), r.end.tz_convert(TIMEZONE)
            rate = r.count / ((r.end - r.start).total_seconds() / 3600)
            ax.fill_between([a, b], 0, rate, step="pre", color=C_PRED, alpha=0.35, lw=0)
            ax.plot([a, a, b, b], [0, rate, rate, 0], color=C_PRED, lw=0.8)
        s = sightings[(sightings["date"] == day) & sightings["datetime"].notna()]
        ax2 = ax.twinx()
        ax2.vlines(s["datetime"].dt.tz_convert(TIMEZONE), 0, 1, color=C_OBS, lw=0.3, alpha=0.5)
        rm = removed[(removed["date"] == day) & removed["datetime"].notna()]
        ax2.vlines(rm["datetime"].dt.tz_convert(TIMEZONE), 0, 1, color="#cf222e", lw=1.2)
        ax2.set_ylim(0, 6)
        ax2.set_yticks([])
        ax.set_ylabel("birds/h")
        ax.set_title(day.strftime("%Y-%m-%d"), fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TIMEZONE))
    fig.tight_layout()
    return (
        "<h2>Example days (Trektellen, split into hours)</h2>"
        '<p class="muted">One random split day per year. Blue: output periods, height = birds/h '
        "of all species. Ticks at the bottom: raw sighting timestamps (grey), sightings "
        "removed by a step (red). Every surveyed hour is an outlined period, of zero height when "
        "nothing was seen; a gap would be an hour missing as effort.</p>" + _fig_to_img(fig)
    )


def render(
    mc: ModelCounts,
    checks: list[Check],
    observations: pd.DataFrame,
    species: list[str],
    inputs: dict[str, str],
) -> str:
    """The full report as one HTML string."""
    p = _periods(mc)
    removed = pd.concat(
        [s.rows for s in mc.log.steps if s.source == "trektellen" and s.action == "removed"]
    )
    status = {k: sum(c.status == k for c in checks) for k in STATUS_COLORS}
    sightings = observations[observations["source"] == "trektellen"]
    inputs_html = "".join(
        f'<tr><td class="l">{html.escape(k)}</td><td class="l">{html.escape(v)}</td></tr>'
        for k, v in inputs.items()
    )
    body = (
        "<h1>Count processing report</h1>"
        f'<p class="muted">Built {dt.datetime.now():%Y-%m-%d %H:%M} by '
        "<code>scripts/build_counts.py</code>, from <code>src/data/counts.py</code>.</p>"
        f"<p>{len(mc.counts):,} rows, {len(p):,} survey periods, "
        f"{p['hours'].sum():,.0f} surveyed hours, {mc.counts['count'].sum():,.0f} birds. "
        f"Checks: {_badge('pass')} {status['pass']} {_badge('warn')} {status['warn']} "
        f"{_badge('fail')} {status['fail']}</p>"
        '<p class="muted">Model processing only. Data-entry errors, corrections and the '
        "dataset's own checks are in the defile-dataset report.</p>"
        f"<table>{inputs_html}</table>"
        + _section_checks(checks)
        + _section_steps(mc)
        + _section_balance(mc)
        + _section_effort(mc)
        + _section_species(mc, species)
        + _section_examples(mc, sightings, removed)
    )
    n = iter(range(1, 100))
    body = re.sub("<h2>", lambda m: f"<h2>{next(n)}. ", body)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Count processing report</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
