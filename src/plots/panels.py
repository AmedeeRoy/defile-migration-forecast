"""Axes-level panels for the per-species test report.

Every function here draws into an axes that the caller owns and returns nothing. Nothing
in this module touches the pyplot global figure registry, creates a figure, or writes a
file -- `src/plots/report.py` does all three, once, for the whole report. That split is
what makes the panels composable into a single PDF instead of a scatter of JPEGs, and it
keeps a multirun over 11 species from leaking one figure per plot into pyplot's registry.

Dense panels (per-row scatters) are marked `rasterized=True`: the report is a vector PDF,
and a few thousand individually-stroked scatter markers per species is the difference
between a ~200 KB file and a multi-megabyte one that stalls a PDF viewer on scroll.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Shared, colour-blind-safe roles used consistently across every panel, so a colour means
# the same thing on page 2 as it does on page 5.
C_OBS = "#333333"
C_PRED = "#0072B2"
C_PHEN = "#E69F00"
C_PERS = "#009E73"


def _empty(ax, message: str = "no data") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------------------
# Level 1 -- per survey row
# --------------------------------------------------------------------------------------


def draw_true_vs_prediction(ax, obs, pred, log_transformed: bool = True) -> None:
    """Predicted vs observed rate per survey row, with the 1:1 line and an OLS fit."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    if len(obs) == 0:
        return _empty(ax)

    if log_transformed:
        obs, pred = np.log1p(obs), np.log1p(pred)

    ax.scatter(obs, pred, c=C_OBS, s=5, alpha=0.35, rasterized=True, linewidths=0)

    lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.plot(lim, lim, "--", c="black", lw=1, label="1:1")

    # A degenerate fit (all-identical observations) would make polyfit raise rather than
    # tell you anything; the 1:1 line alone is still a useful panel there.
    if np.ptp(obs) > 0:
        x = np.linspace(*lim, 50)
        ax.plot(x, np.polyval(np.polyfit(obs, pred, 1), x), c="red", lw=1.2, label="fit")

    unit = "log1p(birds/hr)" if log_transformed else "birds/hr"
    ax.set_xlabel(f"Observed [{unit}]")
    ax.set_ylabel(f"Predicted [{unit}]")
    ax.set_title("Per survey row")
    ax.legend(fontsize=7, loc="upper left")


def draw_counts_distribution(ax, obs, pred, phen=None) -> None:
    """Histogram of observed vs predicted rates -- does the model reproduce the skew?

    The log y-axis is the point: 61-95% of rows are zero, and the top 1% hold most of the
    birds. On a linear axis the tail that matters is invisible.
    """
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    if len(obs) == 0:
        return _empty(ax)

    bins = np.linspace(0, max(obs.max(), pred.max()), 60)
    ax.hist(obs, bins=bins, label="observed", alpha=0.55, color=C_OBS, edgecolor="none")
    ax.hist(pred, bins=bins, label="predicted", alpha=0.55, color=C_PRED, edgecolor="none")
    if phen is not None:
        ax.hist(
            np.asarray(phen, float),
            bins=bins,
            label="phenology",
            histtype="step",
            color=C_PHEN,
            lw=1.2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Rate over survey [birds/hr]")
    ax.set_ylabel("Rows")
    ax.set_title("Distribution of rates")
    ax.legend(fontsize=7)


def draw_residuals(ax, daily: pd.DataFrame) -> None:
    """Daily prediction error against observed rate -- where the bias actually lives.

    Plotted per day rather than per row so the hourly-recording era does not contribute
    fifteen near-duplicate points for every one from the daily-totals era.
    """
    if daily.empty:
        return _empty(ax)

    obs, pred = daily["obs"].to_numpy(), daily["pred"].to_numpy()
    ax.axhline(0, c="black", lw=1, ls="--")
    ax.scatter(obs, pred - obs, s=8, c=C_PRED, alpha=0.5, rasterized=True, linewidths=0)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Observed daily rate [birds/hr]")
    ax.set_ylabel("Error (pred - obs) [birds/hr]")
    ax.set_title("Daily residuals")


# --------------------------------------------------------------------------------------
# Level 2 / 4 -- day of year and season
# --------------------------------------------------------------------------------------


def draw_doy_year(ax, year_daily: pd.DataFrame, phenology=None, threshold=None) -> None:
    """Observed / predicted daily rate through one season, over the phenological band."""
    if year_daily.empty:
        return _empty(ax)

    g = year_daily.sort_values("doy")
    doy = g["doy"].to_numpy()

    if phenology is not None:
        # p25-p75 of the phenology as context: is a miss unusual for that date, or is
        # the date itself simply variable?
        lo = phenology.quantile(doy, 25)
        hi = phenology.quantile(doy, 75)
        ax.fill_between(doy, lo, hi, color=C_PHEN, alpha=0.25, lw=0, label="phen. p25-p75")
        ax.plot(doy, phenology.daily_rate(doy), c=C_PHEN, lw=1, label="phenology")

    if threshold is not None:
        ax.plot(doy, threshold, c=C_PHEN, lw=0.8, ls=":", label="event threshold")

    ax.plot(doy, g["obs"], c=C_OBS, lw=1.4, label="observed")
    ax.plot(doy, g["pred"], c=C_PRED, lw=1.4, label="predicted")
    ax.set_ylabel(f"{int(g['year'].iloc[0])}\n[birds/hr]")


def draw_cumulative_passage(ax, year_daily: pd.DataFrame, show_xlabel: bool = True) -> None:
    """Cumulative share of the season's passage -- the phenology view of level 4.

    Normalised to 1, so a curve shifted left or right is a timing error and a curve of the
    wrong shape is a within-season distribution error; magnitude errors, which the
    seasonal-total ratio already reports, are deliberately divided out.

    The year is already labelled on the row's left-hand panel (`draw_doy_year`), so it is
    not repeated here as a title -- with one row per test year, a title on every row
    collided with the `set_ylabel` below it once more than a handful of years are stacked
    on one page.
    """
    if year_daily.empty:
        return _empty(ax)

    g = year_daily.sort_values("doy")
    for column, colour, label in (("obs", C_OBS, "observed"), ("pred", C_PRED, "predicted")):
        total = g[column].sum()
        if total > 0:
            ax.plot(g["doy"], np.cumsum(g[column]) / total, c=colour, lw=1.4, label=label)

    ax.axhline(0.5, c="grey", lw=0.7, ls=":")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative\nshare", fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Day of year")
    else:
        ax.set_xticklabels([])


# --------------------------------------------------------------------------------------
# Level 3 -- intra-day shape
# --------------------------------------------------------------------------------------


def draw_diurnal_profile(ax, profiles: pd.DataFrame, phenology=None) -> None:
    """Mean shape of the day, observed vs predicted, over the hourly-resolution dates.

    Each day is normalised to sum 1 before averaging, so a handful of huge days cannot
    define the mean shape on their own.
    """
    if profiles.empty:
        return _empty(ax, "no hourly-resolution dates")

    def _mean_shape(key):
        stack = []
        for _, row in profiles.iterrows():
            p = np.where(row["covered"], row[key], np.nan)
            total = np.nansum(p)
            if total > 0:
                stack.append(p / total)
        if not stack:
            return np.full(24, np.nan)
        # Hours nobody ever watched are NaN in every row; averaging them is a NaN, not an
        # error, so the "mean of empty slice" warning is noise here rather than a signal.
        stacked = np.vstack(stack)
        mean = np.full(24, np.nan)
        has_data = ~np.isnan(stacked).all(axis=0)
        mean[has_data] = np.nanmean(stacked[:, has_data], axis=0)
        return mean

    hours = np.arange(24)
    ax.plot(hours, _mean_shape("obs_profile"), c=C_OBS, lw=1.6, label="observed")
    ax.plot(hours, _mean_shape("pred_profile"), c=C_PRED, lw=1.6, label="predicted")

    if phenology is not None:
        phen = phenology.hourly_rate(profiles["doy"].to_numpy()).mean(axis=0)
        if phen.sum() > 0:
            ax.plot(hours, phen / phen.sum(), c=C_PHEN, lw=1.2, label="phenology")

    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Share of the day")
    ax.set_title(f"Mean diurnal shape ({len(profiles)} hourly-resolution dates)", fontsize=9)
    ax.legend(fontsize=7)


def draw_peak_hour_scatter(ax, profiles: pd.DataFrame) -> None:
    """Predicted vs observed peak hour, the headline shape metric, day by day."""
    if profiles.empty:
        return _empty(ax, "no hourly-resolution dates")

    obs_peak, pred_peak = [], []
    for _, row in profiles.iterrows():
        hours = np.flatnonzero(row["covered"])
        obs = row["obs_profile"][row["covered"]]
        if obs.sum() <= 0:
            continue
        obs_peak.append(hours[int(np.argmax(obs))])
        pred_peak.append(hours[int(np.argmax(row["pred_profile"][row["covered"]]))])

    if not obs_peak:
        return _empty(ax, "no non-zero days")

    # Jitter: peak hours are integers, so without it a hundred identical days render as
    # one indistinguishable point and the panel says nothing about how common that day is.
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.2, 0.2, size=(2, len(obs_peak)))
    ax.scatter(
        np.asarray(obs_peak) + jitter[0],
        np.asarray(pred_peak) + jitter[1],
        s=10,
        c=C_PRED,
        alpha=0.45,
        rasterized=True,
        linewidths=0,
    )
    lim = [min(obs_peak + pred_peak) - 1, max(obs_peak + pred_peak) + 1]
    ax.plot(lim, lim, "--", c="black", lw=1)
    ax.set_xlabel("Observed peak hour (UTC)")
    ax.set_ylabel("Predicted peak hour (UTC)")
    ax.set_title("Peak hour", fontsize=9)


def draw_sample_day(ax, date, obs_rows, masks, pred_profile, label: Optional[str] = None) -> None:
    """One day: the predicted hourly curve against each survey row's observed flat rate.

    The yellow shading is survey coverage (alpha = fraction of the hour watched) and each
    red segment is one survey row drawn at its observed rate across the hours it covers --
    which is all the observation constrains. A prediction is only wrong where it is red.
    """
    hours = np.arange(24)
    ax.plot(hours + 0.5, pred_profile, c=C_PRED, lw=1.4)

    ymax = max(float(np.max(pred_profile)), float(np.max(obs_rows)) if len(obs_rows) else 0.0)
    ymax = ymax * 1.15 + 1e-6

    # One span per covered hour rather than a single bar: the alpha *is* the information
    # (fraction of that hour watched), and a single artist can only carry one alpha.
    coverage = np.clip(masks.sum(axis=1), 0, 1)
    for h, cov in enumerate(coverage):
        if cov > 0:
            ax.axvspan(h, h + 1, color="gold", alpha=0.45 * float(cov), lw=0)

    for row_idx, obs in enumerate(obs_rows):
        m = masks[:, row_idx]
        covered = np.flatnonzero(m > 0)
        if len(covered):
            ax.plot([covered[0], covered[-1] + 1], [obs, obs], c="tab:red", lw=1.6)

    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.text(
        0.02,
        0.93,
        label or pd.Timestamp(date).strftime("%Y-%m-%d"),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
    )
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def format_cell(value) -> str:
    """Format one table cell for print.

    matplotlib's default float repr turns a year into `2.02e+03` and a row count into
    `1.56e+03`, which is worse than useless in a report meant to be read at a glance.
    Whole numbers print as whole numbers; everything else gets three significant digits.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        if value == int(value) and abs(value) < 1e6:
            return f"{int(value):,}"
        return f"{value:,.3g}" if abs(value) >= 0.01 else f"{value:.1e}"
    return str(value)


def draw_table(ax, df: pd.DataFrame, title: str = "", fontsize: int = 8) -> None:
    """Render a DataFrame as a table panel filling the axes it is given.

    `bbox=[0, 0, 1, 1]` rather than `loc=`: the caller sizes the gridspec row to the
    number of table rows, and the table then fills it exactly instead of floating in the
    middle of a mostly-empty panel.

    Column widths are set from content length rather than left equal: with 8-12 metric
    columns in the width of a page, equal widths make every cell too narrow for its text,
    and matplotlib's table does not wrap -- it overlaps into the neighbouring cell
    instead, silently, which is worse than a table that just looks cramped.
    """
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, loc="left", pad=4)
    if df.empty:
        return _empty(ax)

    columns = [str(c) for c in df.columns]
    cells = [[format_cell(v) for v in row] for row in df.to_numpy()]

    # Longest line of any cell in the column (header or body), in characters -- the same
    # quantity `auto_set_column_width` would use, computed explicitly so it can be turned
    # into a width fraction that actually sums to 1 across every column.
    widths = [
        max([len(line) for line in col.split("\n")] + [len(str(r[i])) for r in cells])
        for i, col in enumerate(columns)
    ]
    widths = np.asarray(widths, dtype=float) + 1.5  # padding
    widths /= widths.sum()

    table = ax.table(
        cellText=cells,
        colLabels=columns,
        colWidths=widths.tolist(),
        cellLoc="right",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#cccccc")
        cell.PAD = 0.03
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold", fontsize=fontsize - 0.5)
