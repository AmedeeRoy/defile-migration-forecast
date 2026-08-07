"""Within-season history features: what has already been observed earlier in the same
season, as model inputs.

This is the "own-site lag" experiment from DEVELOPMENT.md's Trektellen roadmap, built
directly from `all_count_processed.csv` rather than a separate live feed -- for training,
"what was observed earlier this season" is just this species' own earlier rows in the same
year, which the historical count data already has. At forecast time the same feature is
computed the same way, using whatever of the current season has actually been observed so
far (a live daily-cron run and a training row are the same computation, just evaluated at
a different point in the season).

Two kinds of feature, each as a `(value, hours)` pair:

- **Lag windows** (`history_windows`, e.g. `[1, 3, 7]`): the birds/observer-hour rate over
  the trailing N calendar days strictly before a row's date, within the same year.
- **Season-to-date cumulative** (`history_cumulative=True`): the birds/observer-hour rate
  from the start of that year's fitted season through the day before a row's date.

`value` is `log1p(rate)`; `hours` is `log1p(hours observed in that window)`, kept as a
separate channel rather than folded into `value` -- the day before the season starts and a
day after a genuinely quiet week both have `rate == 0`, but only one of them is backed by
real observation. Zero-filling `value` alone (the DEVELOPMENT.md caution about the
Trektellen version of this feature) would make those indistinguishable; `hours` gives the
network the coverage signal to tell them apart and discount the estimate when it's low.

Missing calendar days (no session logged, e.g. bad weather) are true zeros here, not an
ambiguous absence: this is Defile's own historical record, where "no row for that date"
means the site genuinely wasn't watched that day, not "watched but not yet reported" (the
live-feed case the DEVELOPMENT.md caution is actually about).
"""

import os

import numpy as np
import pandas as pd


def _history_column_names(windows, cumulative):
    """The `(value, hours)` column name pairs `add_history_features` adds, in order."""
    names = [(f"history_lag{n}_value", f"history_lag{n}_hours") for n in windows]
    if cumulative:
        names.append(("history_cum_value", "history_cum_hours"))
    return names


def n_history_channels(windows, cumulative):
    """Total number of history channels (value + hours per window/cumulative)."""
    return 2 * (len(windows) + int(bool(cumulative)))


def add_history_features(count: pd.DataFrame, windows=(), cumulative=False) -> pd.DataFrame:
    """Adds within-season history columns to `count` (one row per observation period, of
    one species -- i.e. `DefileDataModule.read_counts`'s frame before `.dropna()` collapses
    it further).

    Requires `count_raw`, `duration`, `date`, `year` columns already present. Returns
    `count` unchanged (same row count and order) plus one `(value, hours)` column pair per
    requested window/cumulative flag; a no-op (returns `count` as-is) if neither is
    requested, so this is a strict no-op default matching current behaviour.

    Parameters
    ----------
    count : pd.DataFrame
        One row per observation period (possibly several per date), already filtered to a
        single species and to the trained season/years.
    windows : sequence of int
        Trailing-day lag windows, e.g. `(1, 3, 7)`.
    cumulative : bool
        Whether to also add the season-to-date cumulative rate.
    """
    if not windows and not cumulative:
        return count

    # Collapse to one row per calendar date (periods within a date have already been
    # deduplicated by the (date, start, end) groupby in read_counts, so this sum is exactly
    # that date's total, not an inflation) then reindex onto a gapless daily calendar per
    # year -- a day with no session becomes a true (0 count, 0 hours) row rather than a
    # silently skipped position, which would otherwise let a 3-day window's shift/rolling
    # reach further back in time than 3 calendar days across a gap.
    daily = count.groupby(["year", "date"], as_index=False).agg(
        count_raw=("count_raw", "sum"), duration=("duration", "sum")
    )

    frames = []
    for year, g in daily.groupby("year"):
        g = g.set_index("date").sort_index()
        full_range = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_range).fillna(0.0)
        g.index.name = "date"
        g["year"] = year
        frames.append(g)
    daily_full = pd.concat(frames).reset_index().sort_values(["year", "date"]).reset_index(drop=True)

    by_year_count = daily_full.groupby("year")["count_raw"]
    by_year_hours = daily_full.groupby("year")["duration"]

    def _add(value_col, hours_col, trailing_count, trailing_hours):
        trailing_count = trailing_count.fillna(0.0)
        trailing_hours = trailing_hours.fillna(0.0)
        rate = np.where(trailing_hours > 0, trailing_count / trailing_hours.replace(0, np.nan), 0.0)
        daily_full[value_col] = np.log1p(rate)
        daily_full[hours_col] = np.log1p(trailing_hours)

    for n in windows:
        _add(
            f"history_lag{n}_value",
            f"history_lag{n}_hours",
            by_year_count.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).sum()),
            by_year_hours.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).sum()),
        )

    if cumulative:
        _add(
            "history_cum_value",
            "history_cum_hours",
            by_year_count.transform(lambda s: s.cumsum().shift(1)),
            by_year_hours.transform(lambda s: s.cumsum().shift(1)),
        )

    history_cols = [c for pair in _history_column_names(windows, cumulative) for c in pair]
    return count.merge(daily_full[["year", "date"] + history_cols], on=["year", "date"], how="left")


def history_as_of_date(data_dir, species, doy, as_of_date, windows=(), cumulative=False) -> np.ndarray:
    """History channels "as of" `as_of_date`, for the forecast path.

    A forecast is for one or more *future* dates, so unlike training (where every row has
    its own, evolving history) the history features here are computed once, from real
    observations strictly before `as_of_date`, and meant to be applied identically to every
    date in the forecast horizon: nothing new is actually observed between issuing the
    forecast and the days it covers, so the "as of now" history doesn't change over that
    horizon either.

    Falls back to all-zero (no coverage) rather than raising if `species` has no rows yet
    in `as_of_date`'s season -- e.g. very early in the season, or if
    `all_count_processed.csv` hasn't been refreshed yet. This mirrors the fallback
    philosophy the rest of this project uses for the daily forecast job (`src.phenology`,
    `src.trend`): a missing input degrades gracefully rather than failing the run.

    Caveat this does not solve: it reads `all_count_processed.csv`, which reflects
    whatever the file was last refreshed with, not necessarily this morning's live counts.
    Genuinely live within-season history needs the file (or a live feed) refreshed daily
    during the season -- see DEVELOPMENT.md "Trektellen counts as model input".
    """
    if not windows and not cumulative:
        return np.zeros(0, dtype=np.float32)

    as_of_date = pd.Timestamp(as_of_date)
    year = as_of_date.year

    path = os.path.join(data_dir, "count", "all_count_processed.csv")
    all_count = pd.read_csv(path, parse_dates=["date", "start", "end"])
    all_count = all_count[
        (all_count["species"] == species)
        & (all_count["date"].dt.year == year)
        & (all_count["date"].dt.day_of_year.between(*doy))
        & (all_count["date"] < as_of_date)
    ]

    history_cols = [c for pair in _history_column_names(windows, cumulative) for c in pair]
    if all_count.empty:
        return np.zeros(len(history_cols), dtype=np.float32)

    periods = all_count.groupby(["date", "start", "end"], as_index=False)["count"].sum()
    periods["duration"] = (periods["end"] - periods["start"]).dt.total_seconds() / 3600
    periods = periods[periods["duration"] > 0]
    periods["count_raw"] = periods["count"]
    periods["year"] = year

    # A placeholder row for `as_of_date` itself: contributes nothing (count_raw=duration=0)
    # to any window, but gives `add_history_features` a row to compute and return the
    # "as of today" history for -- every window/cumulative sum excludes the row's own date
    # by construction (see that function's shift/cumsum-then-shift logic).
    placeholder = pd.DataFrame(
        {"date": [as_of_date], "count_raw": [0.0], "duration": [0.0], "year": [year]}
    )
    combined = pd.concat([periods[["date", "count_raw", "duration", "year"]], placeholder], ignore_index=True)

    out = add_history_features(combined, windows=windows, cumulative=cumulative)
    row = out[out["date"] == as_of_date].iloc[0]
    return row[history_cols].to_numpy(dtype=np.float32)
