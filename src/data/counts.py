"""Count data: the defile-dataset tables -> the model's survey-period counts.

The counts come from the separate **defile-dataset** repo, which reads the raw files, applies the
data corrections (night periods, overlapping counts, timestamps; each flagged, nothing removed) and
documents every column. Its two tables, `surveys.csv` and `observations.csv`, are copied into
`data/count/dataset/` (`scripts/build_counts.py --dataset <dir>`).

This module holds only the choices the *model* needs, turning those tables into survey periods of
(mostly) one hour, written to `data/count/all_count_processed.csv`. Every rule that removes, moves
or adds rows is one named step recorded in a `ProcessingLog`, and `check_model_counts` verifies the
result, so `scripts/build_counts.py` can report what each rule did. `data/count/readme.md` lists
the rules.

Output schema (`OUTPUT_COLUMNS`), one row per species per survey period: `species` (English name,
empty when the taxon has none), `date` (local survey date), `count` (birds), `start`/`end` (UTC).
Effort is implicit: a survey period is any (start, end) appearing on at least one row, and a
species absent from a period counted zero there (`DefileDataModule.read_counts`). That is why hours
with birds of no species need a `No species` row with count 0 -- without it the hour would not
exist as effort at all.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The site's local time: the dataset's `date` columns are local calendar dates.
TIMEZONE = "Europe/Paris"

DATASET_DIR = os.path.join("count", "dataset")  # under the data dir
DATASET_FILES = ("surveys.csv", "observations.csv", "metadata.json")

# Effort placeholder for an hour surveyed with no bird recorded.
NO_SPECIES = "No species"

# Dataset flags (defile-dataset `build.py`) whose observations the model cannot use.
EXCLUDED_FLAGS = {
    "no_time": "Record without a start/end time: it belongs to no survey period.",
    "no_survey": "Entry whose Trektellen count is missing from the header export.",
    "duplicate_survey": "Entry of a count overlapping a longer one: the same birds, counted "
    "twice; the longer count is kept.",
    "time_outside_survey": "Timestamp more than 10 min outside its count period.",
}

# A period is split into clock hours only if it is longer than this ...
SPLIT_MIN_DURATION = pd.Timedelta(hours=2)
# ... and (Trektellen) fewer than this share of its sightings with migrating birds lack a
# timestamp.
SPLIT_MAX_UNTIMED_SHARE = 0.5
# Periods shorter than this are dropped, with their birds (Trektellen only).
MIN_PERIOD_DURATION = pd.Timedelta(minutes=10)

# Checks only (nothing is removed by these): output periods shorter than this give extreme
# hourly rates; survey windows longer than this (a July dawn-to-dusk day is ~15 h 20) are
# probably a mis-entered start or end.
SHORT_PERIOD_WARNING = pd.Timedelta(minutes=15)
LONG_PERIOD_WARNING = pd.Timedelta(hours=16)

OUTPUT_COLUMNS = ["species", "date", "count", "start", "end"]
SOURCES = ("historical", "trektellen")


def fmt_duration(td: pd.Timedelta) -> str:
    """'10 min', '2 h' -- for rule descriptions."""
    minutes = td.total_seconds() / 60
    return f"{minutes / 60:g} h" if minutes >= 60 else f"{minutes:g} min"


# ---------------------------------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------------------------------


def read_dataset(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`(surveys, observations, metadata)` from `<data_dir>/count/dataset/`."""
    folder = os.path.join(data_dir, DATASET_DIR)
    surveys = pd.read_csv(os.path.join(folder, "surveys.csv"), low_memory=False)
    for c in ("start", "end", "start_original", "end_original", "day_start", "day_end"):
        surveys[c] = pd.to_datetime(surveys[c], utc=True)
    surveys["date"] = pd.to_datetime(surveys["date"])
    observations = pd.read_csv(os.path.join(folder, "observations.csv"), low_memory=False)
    for c in ("datetime", "datetime_original"):
        observations[c] = pd.to_datetime(observations[c], utc=True, format="ISO8601")
    observations["date"] = pd.to_datetime(observations["date"])
    observations["flags"] = observations["flags"].fillna("")
    with open(os.path.join(folder, "metadata.json")) as f:
        metadata = json.load(f)
    return surveys, observations, metadata


# ---------------------------------------------------------------------------------------
# Model counts
# ---------------------------------------------------------------------------------------


@dataclass
class Step:
    """One processing rule and the rows it touched."""

    source: str
    name: str
    action: str  # "removed" | "modified" | "added" | "merged"
    rule: str
    rows: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def birds(self) -> float:
        return float(self.rows["count"].sum()) if "count" in self.rows else 0.0


@dataclass
class ProcessingLog:
    steps: list[Step] = field(default_factory=list)

    def record(self, source, name, action, rule, rows) -> None:
        self.steps.append(Step(source, name, action, rule, rows.copy()))

    def removed_birds(self, source: str) -> pd.Series:
        """Birds dropped by `removed` steps, by year."""
        rows = [s.rows for s in self.steps if s.source == source and s.action == "removed"]
        rows = [r for r in rows if len(r)]
        if not rows:
            return pd.Series(dtype=float)
        df = pd.concat(rows)
        return df.groupby(df["date"].dt.year)["count"].sum()


@dataclass
class ModelCounts:
    counts: pd.DataFrame  # OUTPUT_COLUMNS, the content of all_count_processed.csv
    log: ProcessingLog
    # Survey windows split into clock hours, per source: columns date, start, end (UTC).
    split_windows: dict[str, pd.DataFrame]
    # Birds in the dataset (before any step), by year, per source.
    raw_birds: dict[str, pd.Series]


def overlaps_any(start, end, ref_start, ref_end) -> np.ndarray:
    """For each [start, end), whether it overlaps any [ref_start, ref_end).

    Sort-based (O(n log n)): the intervals starting before `end` overlap iff the latest end among
    them is after `start`.
    """
    start, end = np.asarray(start, "datetime64[ns]"), np.asarray(end, "datetime64[ns]")
    rs, re_ = np.asarray(ref_start, "datetime64[ns]"), np.asarray(ref_end, "datetime64[ns]")
    order = np.argsort(rs)
    rs, max_end = rs[order], np.maximum.accumulate(re_[order])
    k = np.searchsorted(rs, end, side="left")
    out = np.zeros(len(start), dtype=bool)
    has = k > 0
    out[has] = max_end[k[has] - 1] > start[has]
    return out


def hourly_slots(windows: pd.DataFrame) -> pd.DataFrame:
    """Cut each [start, end) window into clock hours (UTC), keeping partial first/last hours.

    Returns one row per slot: the window's other columns plus the slot's `start`/`end`.
    """
    rows = []
    for w in windows.itertuples(index=False):
        w = w._asdict()
        edges = [w["start"], *pd.date_range(w["start"].ceil("h"), w["end"], freq="h"), w["end"]]
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                rows.append({**w, "start": a, "end": b})
    return pd.DataFrame(rows, columns=windows.columns).drop_duplicates(ignore_index=True)


def _drop_flagged(df: pd.DataFrame, src: str, log: ProcessingLog) -> pd.DataFrame:
    flags = df["flags"].str.split(";")
    for flag, rule in EXCLUDED_FLAGS.items():
        hit = flags.apply(lambda f: flag in f)
        if hit.any():
            log.record(src, f"Flagged {flag}", "removed", rule, df[hit])
            df, flags = df[~hit], flags[~hit]
    return df


def _with_survey(obs: pd.DataFrame, surveys: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = obs.merge(surveys[["survey_id", *columns]], on="survey_id", how="left")
    assert len(out) == len(obs)
    return out


def historical_model_counts(obs: pd.DataFrame, surveys: pd.DataFrame, log: ProcessingLog):
    """Historical observations -> model counts.

    Returns (counts, split windows).
    """
    src = "historical"
    df = _drop_flagged(obs, src, log)
    df = _with_survey(df, surveys, ["start", "end", "day_start", "day_end"])

    # Grouped by the original (French) name: two names mapping to one English name would
    # stay two rows, which the "One row per species and period" check would catch.
    keys = ["taxon_name_original", "english_name", "date", "start", "end", "day_start", "day_end"]
    merged = df.groupby(keys, as_index=False, dropna=False).agg(count=("count", "sum"))
    log.record(
        src,
        "Same species, same period",
        "merged",
        "Several records of one species in one period are summed into one row. "
        f"{len(df) - len(merged)} rows merged; no bird lost.",
        df[df.duplicated(keys, keep=False)],
    )
    df = merged

    # Zero-fill: on a day recorded hour by hour, an hour with no bird has no row and so
    # would not exist as effort. A day counts as hour-by-hour when one of its periods lies
    # strictly inside the day window (starts after it opens and ends before it closes).
    periods = df[["date", "start", "end", "day_start", "day_end"]].drop_duplicates()
    inner = (
        (periods["start"] != periods["day_start"])
        & (periods["end"] != periods["day_end"])
        & ((periods["day_end"] - periods["day_start"]) > SPLIT_MIN_DURATION)
    )
    windows = (
        periods.loc[inner, ["date", "day_start", "day_end"]]
        .drop_duplicates()
        .rename(columns={"day_start": "start", "day_end": "end"})
        .reset_index(drop=True)
    )
    slots = hourly_slots(windows)
    empty = ~overlaps_any(slots["start"], slots["end"], df["start"], df["end"])
    zeros = slots[empty].assign(english_name=NO_SPECIES, count=0)
    log.record(
        src,
        "Zero-fill empty hours",
        "added",
        f"On days recorded hour by hour (a period strictly inside a day window longer than "
        f"{fmt_duration(SPLIT_MIN_DURATION)}), each clock hour of the day window overlapping "
        f"no record gets a '{NO_SPECIES}' row with count 0, so it exists as survey effort.",
        zeros,
    )
    if len(zeros):
        df = pd.concat([df, zeros], ignore_index=True)
    return df.rename(columns={"english_name": "species"})[OUTPUT_COLUMNS], windows


def trektellen_model_counts(obs: pd.DataFrame, surveys: pd.DataFrame, log: ProcessingLog):
    """Trektellen observations -> model counts.

    Returns (counts, split windows).
    """
    src = "trektellen"
    df = _drop_flagged(obs, src, log)
    df = _with_survey(df, surveys, ["start", "end"])

    # Which periods to split into hours: long enough, and mostly timestamped. The share is over
    # entries with migrating birds: untimed entries of local birds only (count 0) say nothing
    # about whether the count was timed, and would otherwise un-split a timed count.
    stats = (
        df[df["count"] > 0]
        .groupby(["survey_id", "date", "start", "end"])
        .agg(n=("datetime", "size"), untimed=("datetime", lambda x: x.isna().sum()))
        .reset_index()
    )
    to_split = stats[
        ((stats["end"] - stats["start"]) > SPLIT_MIN_DURATION)
        & (stats["untimed"] / stats["n"] < SPLIT_MAX_UNTIMED_SHARE)
    ]
    split = df["survey_id"].isin(to_split["survey_id"])
    windows = to_split[["date", "start", "end"]].drop_duplicates().reset_index(drop=True)

    untimed = split & df["datetime"].isna()
    log.record(
        src,
        "Untimed sighting in split period",
        "removed",
        "In a period split into hours, a sighting without timestamp cannot be placed in an "
        "hour. Mostly local birds, or totals entered at the end of the day.",
        df[untimed],
    )
    df = df[~untimed]
    split = split[~untimed]

    hour = df.loc[split, "datetime"].dt.floor("h")
    new_start = hour.where(hour > df.loc[split, "start"], df.loc[split, "start"])
    new_end = (hour + pd.Timedelta(hours=1)).where(
        hour + pd.Timedelta(hours=1) < df.loc[split, "end"], df.loc[split, "end"]
    )
    log.record(
        src,
        "Assign to clock hour",
        "modified",
        "Each sighting of a split period gets the clock hour of its timestamp as period, "
        "clipped to the count period.",
        df[split],
    )
    df.loc[split, "start"] = new_start
    df.loc[split, "end"] = new_end

    # Zero-fill: an hour of a split period with no sighting of any species gets no row, and
    # would not exist as effort.
    slots = hourly_slots(windows)
    slots = slots[(slots["end"] - slots["start"]) >= MIN_PERIOD_DURATION]
    empty = ~overlaps_any(slots["start"], slots["end"], df["start"], df["end"])
    zeros = slots[empty].assign(english_name=NO_SPECIES, count=0)
    log.record(
        src,
        "Zero-fill empty hours",
        "added",
        f"Each clock hour (at least {fmt_duration(MIN_PERIOD_DURATION)}) of a split period "
        f"with no sighting gets a '{NO_SPECIES}' row with count 0, so it exists as effort.",
        zeros,
    )
    if len(zeros):
        df = pd.concat([df, zeros], ignore_index=True)

    short = (df["end"] - df["start"]) < MIN_PERIOD_DURATION
    log.record(
        src,
        f"Period shorter than {fmt_duration(MIN_PERIOD_DURATION)}",
        "removed",
        f"Periods shorter than {fmt_duration(MIN_PERIOD_DURATION)}, with their birds: too "
        "short for a reliable hourly rate. Mostly the partial last hour of a split period.",
        df[short],
    )
    df = df[~short].rename(columns={"english_name": "species"})

    mapped = df[df["species"].notna()]
    merged = mapped.groupby(["species", "date", "start", "end"], as_index=False)["count"].sum()
    log.record(
        src,
        "Same species, same period",
        "merged",
        "Sightings of one species in one period are summed into one row. "
        f"{len(mapped) - len(merged)} rows merged; no bird lost. Sightings of taxa with no "
        "English name in the dataset's taxonomy are kept as separate rows with no species "
        "name (they still mark the period as surveyed).",
        mapped[mapped.duplicated(["species", "date", "start", "end"], keep=False)],
    )
    out = pd.concat([merged, df[df["species"].isna()]], ignore_index=True)
    return out[OUTPUT_COLUMNS], windows


def build_model_counts(surveys: pd.DataFrame, observations: pd.DataFrame) -> ModelCounts:
    log = ProcessingLog()
    parts, windows, raw = [], {}, {}
    for src, fn in (
        ("historical", historical_model_counts),
        ("trektellen", trektellen_model_counts),
    ):
        obs = observations[observations["source"] == src]
        counts, windows[src] = fn(obs, surveys[surveys["source"] == src], log)
        parts.append(counts)
        raw[src] = obs.groupby(obs["date"].dt.year)["count"].sum()
    out = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["start", "end", "species"], kind="stable", na_position="last")
        .reset_index(drop=True)
    )
    out["count"] = out["count"].astype(int)
    return ModelCounts(counts=out, log=log, split_windows=windows, raw_birds=raw)


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def source_of(dates: pd.Series, trektellen_from: int) -> pd.Series:
    return np.where(dates.dt.year >= trektellen_from, "trektellen", "historical")


def check_model_counts(mc: ModelCounts) -> list[Check]:
    c = mc.counts
    t_from = int(mc.raw_birds["trektellen"].index.min())
    periods = c[["date", "start", "end"]].drop_duplicates().sort_values(["start", "end"])
    periods["duration_min"] = (periods["end"] - periods["start"]).dt.total_seconds() / 60
    periods["source"] = source_of(periods["date"], t_from)
    checks = []

    # Every bird either reaches the output or is dropped by a named step.
    rows = []
    for src in SOURCES:
        out_birds = c.loc[source_of(c["date"], t_from) == src]
        out_birds = out_birds.groupby(out_birds["date"].dt.year)["count"].sum()
        df = pd.DataFrame(
            {"raw": mc.raw_birds[src], "removed": mc.log.removed_birds(src), "output": out_birds}
        ).fillna(0)
        df["unexplained"] = df["raw"] - df["removed"] - df["output"]
        rows.append(df.assign(source=src).rename_axis("year").reset_index())
    rec = pd.concat(rows, ignore_index=True)
    bad = rec[rec["unexplained"].abs() > 0]
    checks.append(
        Check(
            "Birds accounted for",
            "pass" if bad.empty else "fail",
            "Per source and year: birds in the dataset = birds dropped by a named step + birds "
            f"in the output. {len(bad)} year(s) do not balance.",
            bad,
        )
    )

    p = periods.reset_index(drop=True)
    idx = np.flatnonzero(p["start"].values[1:] < p["end"].cummax().values[:-1])
    ov = p.loc[np.unique(np.r_[idx, idx + 1])]
    checks.append(
        Check(
            "No overlapping periods",
            "pass" if ov.empty else "fail",
            f"{len(idx)} survey period(s) overlap an earlier one (a period must be counted once).",
            ov,
        )
    )

    bad = p[p["duration_min"] <= 0]
    checks.append(
        Check(
            "Positive durations",
            "pass" if bad.empty else "fail",
            f"{len(bad)} period(s) with end <= start.",
            bad,
        )
    )

    short_min = SHORT_PERIOD_WARNING.total_seconds() / 60
    bad = p[(p["duration_min"] > 0) & (p["duration_min"] < short_min)]
    checks.append(
        Check(
            f"Periods shorter than {short_min:.0f} min",
            "pass" if bad.empty else "warn",
            f"{len(bad)} period(s); a few birds over minutes give extreme hourly rates. "
            f"By source: {bad['source'].value_counts().to_dict()}.",
            bad,
        )
    )

    # Before splitting too: a whole-day period split into hours would not show up otherwise.
    windows = pd.concat(
        [w.assign(source=s, split=True) for s, w in mc.split_windows.items()] + [p],
        ignore_index=True,
    )
    windows["duration_min"] = (windows["end"] - windows["start"]).dt.total_seconds() / 60
    long_h = LONG_PERIOD_WARNING.total_seconds() / 3600
    bad = windows[windows["duration_min"] > long_h * 60].drop_duplicates(["start", "end"])
    checks.append(
        Check(
            f"Periods longer than {long_h:.0f} h",
            "pass" if bad.empty else "warn",
            f"{len(bad)} period(s), before or after splitting into hours; likely a "
            "mis-entered start or end.",
            bad,
        )
    )

    dup = c[c["species"].notna() & c.duplicated(["species", "start", "end"], keep=False)]
    checks.append(
        Check(
            "One row per species and period",
            "pass" if dup.empty else "fail",
            f"{len(dup)} duplicated row(s).",
            dup,
        )
    )

    unm = c[c["species"].isna()]
    checks.append(
        Check(
            "Species with an English name",
            "pass" if unm.empty else "warn",
            f"{len(unm)} row(s), {unm['count'].sum():.0f} birds, have no species name (taxon "
            "not in the dataset's taxonomy). They are never counted for any species but still "
            "mark their period as surveyed.",
            unm,
        )
    )

    # In windows split into hours, every clock hour should exist as a period.
    rows = []
    for src, windows in mc.split_windows.items():
        slots = hourly_slots(windows)
        slots = slots[(slots["end"] - slots["start"]) >= MIN_PERIOD_DURATION]
        covered = overlaps_any(slots["start"], slots["end"], periods["start"], periods["end"])
        rows.append(slots[~covered].assign(source=src))
    missing = pd.concat(rows, ignore_index=True)
    checks.append(
        Check(
            "Surveyed hours with no row",
            "pass" if missing.empty else "fail",
            f"Clock hours (>= {fmt_duration(MIN_PERIOD_DURATION)}) inside a window split into "
            "hours that no output period covers: surveyed, but missing as effort, so their "
            f"zero counts are lost. By source: {missing['source'].value_counts().to_dict()}.",
            missing,
        )
    )
    return checks
