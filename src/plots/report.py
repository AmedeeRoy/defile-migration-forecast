"""The consolidated per-species test report.

One PDF per species per run, replacing the scatter of individual JPEGs that used to be
written next to each other with no ordering and no metrics. This is the artefact every
Phase 2 experiment (year-subset ladder, location/variable ablation) gets judged against,
so it has to be readable side by side with another run's copy: fixed page order, fixed
colour roles, headline numbers before diagnostics.

Rendering notes -- the report is built ~11 times per multirun, so the cost is worth
keeping down:

- Figures are constructed as bare `Figure` objects, never through `pyplot`. pyplot would
  register every figure in a process-global list that is only emptied by an explicit
  `close()`; one missed close per species is a leak that grows across a multirun.
- One `PdfPages` handle is opened for the whole report and each page is written straight
  into it, so intermediate files are never serialised to disk.
- Dense scatters are rasterized at a fixed 150 dpi (see `panels`), which keeps the PDF in
  the low hundreds of kB. Text, axes and lines stay vector, so it prints cleanly.
- `constrained_layout` is set at figure construction rather than calling `tight_layout()`
  per page: same result, one solve instead of two.
"""

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.metrics import ERA_LABELS, Phenology, MetricReport
from src.plots.explanations import draw_explanations_locations, draw_explanations_metrics
from src.plots.panels import (
    draw_counts_distribution,
    draw_cumulative_passage,
    draw_diurnal_profile,
    draw_doy_year,
    draw_peak_hour_scatter,
    draw_residuals,
    draw_sample_day,
    draw_table,
    draw_true_vs_prediction,
)

A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)
RASTER_DPI = 150

# The headline table: one row per scope, in the order DEVELOPMENT.md lists the levels.
# `key` is the name in MetricReport.scalars; `label` is what the printed table says.
HEADLINE_COLUMNS = [
    ("n_rows", "rows"),
    ("mae", "MAE"),
    ("bias", "bias"),
    ("mae_skill_phen", "skill\nvs phen"),
    ("mae_skill_persistence", "skill\nvs pers"),
    ("event_csi", "CSI"),
    ("event_csi_skill_persistence", "CSI skill\nvs pers"),
    ("shape_peak_hour_mae", "peak err\n[h]"),
    ("shape_emd", "EMD\n[h]"),
    ("season_median_date_mae", "date err\n[d]"),
    ("season_total_ratio", "total\nratio"),
]

DIAGNOSTIC_COLUMNS = [
    ("event_n_days", "event\ndays"),
    ("event_base_rate", "base\nrate"),
    ("event_hits", "hits"),
    ("event_misses", "misses"),
    ("event_false_alarms", "false\nalarms"),
    ("event_pod", "POD"),
    ("event_far", "FAR"),
    ("event_csi_persistence", "CSI\npers"),
    ("mae_phen", "MAE\nphen"),
    ("mae_persistence", "MAE\npers"),
    ("shape_n_days", "shape\ndays"),
    ("shape_peak_hour_within_1h", "peak\n±1h"),
]

# Per-year season table. The p10 passage dates are computed and kept in the metrics JSON
# but not shown: two more columns only earn their place once a median error is large
# enough that you need to know which end of the season is driving it.
SEASON_COLUMNS = {
    "year": "year",
    "era": "era",
    "n_days": "days",
    "obs_median_doy": "obs\nmedian doy",
    "pred_median_doy": "pred\nmedian doy",
    "median_error_days": "median err\n[d]",
    "obs_p90_doy": "obs\np90 doy",
    "pred_p90_doy": "pred\np90 doy",
    "total_ratio": "total\nratio",
}


def _new_page(figsize) -> Figure:
    return Figure(figsize=figsize, dpi=RASTER_DPI, constrained_layout=True)


def _finish(pdf: PdfPages, fig: Figure) -> None:
    pdf.savefig(fig)
    fig.clear()


def _scope_table(report: MetricReport, columns) -> pd.DataFrame:
    """Stack the overall row and the per-era rows into one table."""
    rows = [{"scope": "overall", **{lbl: report.scalars.get(key, np.nan) for key, lbl in columns}}]
    for _, era in report.by_era.iterrows():
        rows.append({"scope": era["era"], **{lbl: era.get(key, np.nan) for key, lbl in columns}})
    return pd.DataFrame(rows)


def _page_summary(pdf: PdfPages, report: MetricReport, run_info: Dict[str, str]) -> None:
    # Landscape: the headline/diagnostic tables carry 8-12 metric columns, which need the
    # extra width -- portrait made every column too narrow for its own header.
    fig = _new_page(A4_LANDSCAPE)

    headline = _scope_table(report, HEADLINE_COLUMNS)
    diagnostics = _scope_table(report, DIAGNOSTIC_COLUMNS)
    season = report.per_year.reindex(columns=list(SEASON_COLUMNS)).rename(columns=SEASON_COLUMNS)
    if "year" in season:
        # A year is a label, not a quantity: "2,019" is not a year.
        season["year"] = season["year"].astype(int).astype(str)

    # Each table gets height in proportion to the rows it actually has (+1 for the header,
    # +0.8 for the title), so a species with one test era does not leave two thirds of the
    # page blank and a species with three does not overflow.
    def _rows(df):
        return len(df) + 1.8

    grid = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[5.5, _rows(headline), _rows(diagnostics), _rows(season), 7],
    )

    ax = fig.add_subplot(grid[0])
    ax.axis("off")
    ax.text(0, 1.0, f"{report.species}", fontsize=19, weight="bold", va="top")
    ax.text(
        0,
        0.80,
        "Test-set report  ·  " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        fontsize=8.5,
        color="grey",
        va="top",
    )
    width = max(len(k) for k in run_info) if run_info else 0
    ax.text(
        0,
        0.66,
        "\n".join(f"{k:<{width}}  {v}" for k, v in run_info.items()),
        fontsize=7,
        va="top",
        family="monospace",
        linespacing=1.5,
    )

    draw_table(
        fig.add_subplot(grid[1]), headline, title="Headline metrics — never pooled across eras"
    )
    draw_table(
        fig.add_subplot(grid[2]),
        diagnostics,
        title="Supporting diagnostics — for when a headline number looks wrong",
    )
    draw_table(fig.add_subplot(grid[3]), season, title="Season level, per test year")

    ax = fig.add_subplot(grid[4])
    ax.axis("off")
    ax.text(
        0,
        1.0,
        "Reading this page\n"
        "  MAE / Bias are birds per hour over each survey row. Skill scores are 1 - model/baseline:\n"
        "  1 is perfect, 0 is no better than the naive baseline, negative is worse than it. A model\n"
        "  that does not beat day-of-year phenology is not using the weather.\n"
        "  CSI scores big days (observed rate above the phenological p90 for that day of year);\n"
        "  phenology cannot exceed its own p90, so persistence is the baseline shown.\n"
        "  Peak-hour error and EMD only use dates recorded as several ~1-hour survey rows, the only\n"
        "  ones that carry a true intra-day shape.\n"
        "  Phenology across years rests on the ~3 test years this split provides; treat the season\n"
        "  block as indicative until leave-one-year-out CV lands (DEVELOPMENT.md Phase 1).",
        fontsize=7.5,
        va="top",
        family="monospace",
        color="#333333",
    )

    _finish(pdf, fig)


def _page_row_level(pdf: PdfPages, report: MetricReport) -> None:
    fig = _new_page(A4_LANDSCAPE)
    fig.suptitle(f"{report.species} — level 1 (survey row) and level 2 (day)", fontsize=12)
    axes = fig.subplots(2, 2).ravel()

    draw_true_vs_prediction(axes[0], report.frame["obs"], report.frame["pred"])
    draw_counts_distribution(
        axes[1],
        report.frame["obs"],
        report.frame["pred"],
        report.frame["phen"] if "phen" in report.frame else None,
    )
    draw_residuals(axes[2], report.daily)

    # Observed vs predicted at the daily level, which is the level CSI is scored at.
    draw_true_vs_prediction(axes[3], report.daily["obs"], report.daily["pred"])
    axes[3].set_title("Per day (coverage-weighted)")

    _finish(pdf, fig)


def _page_season(pdf: PdfPages, report: MetricReport, phenology: Phenology) -> None:
    years = sorted(report.daily["year"].unique())
    if not years:
        return

    # One row per test year: a fixed A4 height stops being readable past ~4-5 years (text
    # from adjacent rows collides), so the page grows with the row count instead. PdfPages
    # takes the page size from each figure, so this doesn't need to match A4 at all.
    fig = _new_page((A4_LANDSCAPE[0], max(A4_LANDSCAPE[1], 1.15 * len(years) + 0.6)))
    fig.suptitle(f"{report.species} — season through each test year", fontsize=12)
    grid = fig.add_gridspec(nrows=max(len(years), 1), ncols=2, width_ratios=[2.2, 1])

    for i, year in enumerate(years):
        year_daily = report.daily[report.daily["year"] == year]
        is_last = i == len(years) - 1

        ax = fig.add_subplot(grid[i, 0])
        draw_doy_year(ax, year_daily, phenology=phenology)
        if i == 0:
            ax.legend(fontsize=6, ncol=4, loc="upper left")
        if is_last:
            ax.set_xlabel("Day of year")
        else:
            ax.set_xticklabels([])

        draw_cumulative_passage(
            fig.add_subplot(grid[i, 1]), year_daily, show_xlabel=is_last
        )

    _finish(pdf, fig)


def _page_shape(
    pdf: PdfPages,
    report: MetricReport,
    phenology: Phenology,
    mask: np.ndarray,
    pred_hourly: np.ndarray,
) -> None:
    fig = _new_page(A4_LANDSCAPE)
    fig.suptitle(f"{report.species} — level 3 (intra-day shape)", fontsize=12)
    grid = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[1.3, 1, 1])

    draw_diurnal_profile(fig.add_subplot(grid[0, :2]), report.profiles, phenology)
    draw_peak_hour_scatter(fig.add_subplot(grid[0, 2:]), report.profiles)

    for ax, (date, obs_rows, masks, profile, label) in zip(
        [fig.add_subplot(grid[1 + r, c]) for r in range(2) for c in range(4)],
        _sample_days(report.frame, mask, pred_hourly),
    ):
        draw_sample_day(ax, date, obs_rows, masks, profile, label=label)

    _finish(pdf, fig)


def _sample_days(frame: pd.DataFrame, mask: np.ndarray, pred_hourly: np.ndarray, n: int = 8):
    """Pick a spread of days to show: the biggest, some typical, some quiet.

    Sampling only the busiest days would flatter the model (it is easiest to be roughly
    right when a lot is moving) and sampling uniformly would show eight empty panels for a
    species that is 90% zeros. A seeded generator keeps the same days across reruns of the
    same split, so two reports are actually comparable.
    """
    daily_obs = frame.groupby("date")["obs"].mean().sort_values(ascending=False)
    if daily_obs.empty:
        return []

    rng = np.random.default_rng(0)
    top = list(daily_obs.index[: n // 2])
    rest = daily_obs.index[n // 2 :]
    middle = (
        list(rng.choice(rest, size=min(len(rest), n - len(top)), replace=False))
        if len(rest)
        else []
    )

    pred_count = np.expm1(pred_hourly)
    out = []
    for date in top + middle:
        idx = np.flatnonzero(frame["date"].to_numpy() == np.datetime64(pd.Timestamp(date)))
        if not len(idx):
            continue
        tag = "busiest" if date in top else "sampled"
        out.append(
            (
                date,
                frame["obs"].to_numpy()[idx],
                mask[:, idx],
                pred_count[idx[0]],
                f"{pd.Timestamp(date):%Y-%m-%d}  ({tag})",
            )
        )
    return out


def _page_explanations(pdf: PdfPages, report: MetricReport, datamodule, explanations) -> None:
    fig = _new_page(A4_PORTRAIT)
    fig.suptitle(f"{report.species} — mean saliency attribution", fontsize=12)
    grid = fig.add_gridspec(nrows=5, ncols=1)

    draw_explanations_metrics(
        [fig.add_subplot(grid[i]) for i in range(3)], datamodule, explanations
    )
    draw_explanations_locations(
        [fig.add_subplot(grid[3]), fig.add_subplot(grid[4])], datamodule, explanations
    )

    _finish(pdf, fig)


def build_report(
    path: str,
    report: MetricReport,
    phenology: Phenology,
    mask: np.ndarray,
    pred_hourly: np.ndarray,
    run_info: Dict[str, str],
    datamodule=None,
    explanations=None,
) -> str:
    """Write the whole report to `path` and return it.

    :param report: The scored predictions, from `src.metrics.evaluate`.
    :param mask: `(24, n_rows)` survey coverage, in test-row order.
    :param pred_hourly: `(n_rows, 24)` predicted log1p(birds/hr).
    :param run_info: Free-form key/value lines printed on the summary page (checkpoint,
        split years, dataset sizes) -- what you need to know to trust or reproduce the
        numbers on the same page.
    :param explanations: Optional Captum attributions; the page is skipped without them.
    """
    with PdfPages(path) as pdf:
        _page_summary(pdf, report, run_info)
        _page_row_level(pdf, report)
        _page_season(pdf, report, phenology)
        _page_shape(pdf, report, phenology, mask, pred_hourly)
        if explanations is not None and datamodule is not None:
            _page_explanations(pdf, report, datamodule, explanations)

        pdf.infodict().update(
            {"Title": f"Défilé forecast — {report.species}", "Subject": "Test-set evaluation"}
        )

    return path
