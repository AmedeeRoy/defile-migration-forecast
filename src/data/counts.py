"""Count data: the defile-dataset release tables -> the model's survey-period counts.

The counts come from the separate **defile-dataset** repo, which reads the raw files, applies the
data corrections and documents every column. Its release tables, `count.csv`, `survey.csv` and
`taxonomy.csv` (with `datapackage.json`, and the build's `metadata.json`), are copied into
`data/count/dataset/` (`scripts/build_counts.py --dataset <dir>`).

This module holds only the choices the *model* needs, turning those tables into survey periods of
(mostly) one hour, written to `data/count/all_count_processed.csv`. Every rule that removes, moves
or adds rows is one named step recorded in a `ProcessingLog`, and `check_model_counts` verifies the
result, so `scripts/build_counts.py` can report what each rule did. `data/count/readme.md` lists
the rules.

Output schema (`OUTPUT_COLUMNS`), one row per species per survey period: `species` (the dataset's
English name), `date` (local survey date), `count` (birds), `start`/`end` (UTC). Effort is
implicit: a survey period is any (start, end) appearing on at least one row, and a species absent
from a period counted zero there (`DefileDataModule.read_counts`). That is why a surveyed hour with
no bird needs a `No species` row with count 0 -- without it the hour would not exist as effort at
all.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from suncalc import get_position

from src.data.weather import get_lat_lon

# The site's local time: the dataset's dates, and the model's `date`, are local calendar dates.
TIMEZONE = "Europe/Paris"

DATASET_DIR = os.path.join("count", "dataset")  # under the data dir
# Release tables, in the release's `dataset/` folder; `metadata.json` sits one level up.
DATASET_FILES = ("count.csv", "survey.csv", "taxonomy.csv", "datapackage.json")
METADATA_FILE = "metadata.json"
# Recorded times of the entries the release keeps at day level because they fall outside their
# survey (`entry_times`): extracted by `scripts/build_counts.py` from the dataset's internal
# observation table, which `source_count_id` joins. Optional: without it those entries are dropped.
ENTRY_TIMES_FILE = "entry_times.csv"

# Effort placeholder for a period surveyed with no bird recorded.
NO_SPECIES = "No species"

# The main migration direction (`count.csv` `count_category`); `reverse` and `local` are other
# quantities, not modelled.
MAIN_CATEGORY = "normal"
# Presence without a number (`count_estimation`); its `count` is empty.
PRESENCE_ONLY = "x"
# The dataset keeps an entry timed outside its survey at day level, like an untimed one; only its
# `remark_processing` tells the two apart.
OUTSIDE_SURVEY_REMARK = "Entry time outside the native survey"

# Survey coverage (`survey.csv` `survey_coverage`) the model cannot use. Only `complete` surveys
# give a rate over their interval.
EXCLUDED_COVERAGE = {
    "none": "Survey with no counting (rain, low cloud, a closure): not effort. It has no counts.",
    "partial": "Survey counted over only part of its interval, with unknown gap times: neither "
    "its rate over the interval nor its empty hours can be trusted. Dropped with its birds.",
    "unknown": "Survey whose coverage cannot be established: dropped with its birds.",
}

# A period is split into clock hours only if it is longer than this ...
SPLIT_MIN_DURATION = pd.Timedelta(hours=2)
# ... and (Trektellen) fewer than this share of its sightings with migrating birds lack a
# timestamp.
SPLIT_MAX_UNTIMED_SHARE = 0.5
# Trektellen: a count reaching further than this into the night (before civil dawn, after civil
# dusk) is clipped to dawn/dusk. Counts closed the same evening end at most ~30 min after dusk;
# every one beyond 45 min was closed days to months later (defile-dataset, until 2026-10).
NIGHT_TOLERANCE = pd.Timedelta(minutes=45)
# Civil twilight: the sun's altitude (deg) at dawn and dusk, the model's night threshold too
# (`night_mask_by_doy_hour`).
NIGHT_SUN_ALTITUDE = -6.0
# An entry timed less than this far outside its survey (clock rounding, a late entry) is moved just
# inside it, by TIMESTAMP_NUDGE from the edge; further out, it belongs to no counted period.
# The rule defile-dataset applied until 2026-10 (`time_adjusted`); a model choice, made here.
TIMESTAMP_TOLERANCE = pd.Timedelta(minutes=10)
TIMESTAMP_NUDGE = pd.Timedelta(minutes=1)
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


def local_date(t: pd.Series) -> pd.Series:
    """Local calendar date (a naive midnight timestamp) of UTC times."""
    return t.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()


def parse_surveys(survey: pd.DataFrame) -> pd.DataFrame:
    """`survey.csv` plus `start`/`end` (UTC), local `date` and `source`.

    `source` is `trektellen` or `historical` (notebook, spreadsheet, Naturalist, and the curated
    non-counting periods).
    """
    s = survey.copy()
    bounds = s["datetime"].str.split("/", expand=True)
    s["start"] = pd.to_datetime(bounds[0], utc=True)
    s["end"] = pd.to_datetime(bounds[1], utc=True)
    s["date"] = local_date(s["start"])
    s["source"] = np.where(s["recording_era"] == "trektellen", "trektellen", "historical")
    return s


def parse_counts(
    count: pd.DataFrame,
    surveys: pd.DataFrame,
    taxonomy: pd.DataFrame,
    entry_times: pd.DataFrame | None = None,
):
    """`count.csv` plus the model's `species`, the `source` and `date` of its survey, and its own
    timestamp `datetime` (UTC; NaT when untimed).

    A count's `datetime` is either empty (it inherits its survey's interval), a local date (untimed,
    or timed outside its survey: the dataset keeps it at day level), or a UTC time. The dataset also
    allows a UTC interval, which no release has used; it is refused rather than guessed at.
    `entry_times` (`source_count_id`, `datetime`) gives back the recorded time of entries timed
    outside their survey, so `fit_times_to_survey` can apply the model's tolerance to them.
    """
    c = count.copy()
    raw = c["datetime"].fillna("")
    interval = raw.str.contains("/")
    if interval.any():
        raise ValueError(
            f"{interval.sum()} count(s) with their own interval (e.g. "
            f"{c.loc[interval, 'count_id'].iloc[0]}): not handled by the model's processing."
        )
    date_only = raw.str.len() == 10
    timed = raw.str.contains("T")
    c["datetime"] = pd.to_datetime(raw.where(timed), utc=True, format="ISO8601")
    c = c.merge(surveys[["survey_id", "source", "date"]], on="survey_id", how="left")
    assert c["source"].notna().all(), "count without a released survey"
    c.loc[timed.values, "date"] = local_date(c.loc[timed.values, "datetime"])
    c.loc[date_only.values, "date"] = pd.to_datetime(raw[date_only]).values
    if entry_times is not None and len(entry_times):
        recorded = c["source_count_id"].map(
            entry_times.set_index("source_count_id")["datetime"].pipe(pd.to_datetime, utc=True)
        )
        back = outside_survey(c) & recorded.notna()
        c.loc[back, "datetime"] = recorded[back]
    c["count"] = c["count"].astype(float)  # presence-only counts are empty
    names = taxonomy.set_index("taxon_id")["english_name"]
    c["species"] = c["taxon_id"].map(names)
    return c


def outside_survey(counts: pd.DataFrame) -> pd.Series:
    """Entries the dataset marks as timed outside their survey."""
    return counts["remark_processing"].fillna("").str.startswith(OUTSIDE_SURVEY_REMARK)


def entry_times(observations: pd.DataFrame, count: pd.DataFrame) -> pd.DataFrame:
    """Recorded time (UTC) of each entry timed outside its survey, from the dataset's internal
    observation table (`observation_id`, `datetime_original`), joined on `source_count_id`."""
    ids = count.loc[outside_survey(count), "source_count_id"].unique()
    o = observations[observations["observation_id"].isin(ids)]
    return pd.DataFrame(
        {"source_count_id": o["observation_id"], "datetime": o["datetime_original"]}
    ).dropna()


def read_dataset(data_dir: str):
    """`(surveys, counts, taxonomy, metadata)` from `<data_dir>/count/dataset/`.

    `metadata` is the release's build `metadata.json`, empty if it was not copied.
    """
    folder = os.path.join(data_dir, DATASET_DIR)
    surveys = parse_surveys(pd.read_csv(os.path.join(folder, "survey.csv"), low_memory=False))
    taxonomy = pd.read_csv(os.path.join(folder, "taxonomy.csv"))
    count = pd.read_csv(os.path.join(folder, "count.csv"), low_memory=False)
    path = os.path.join(folder, ENTRY_TIMES_FILE)
    times = pd.read_csv(path) if os.path.exists(path) else None
    counts = parse_counts(count, surveys, taxonomy, times)
    path = os.path.join(folder, METADATA_FILE)
    metadata = json.load(open(path)) if os.path.exists(path) else {}
    return surveys, counts, taxonomy, metadata


def trektellen_species_ids(taxonomy: pd.DataFrame) -> dict[str, int]:
    """English name -> Trektellen species id, from the dataset's taxonomy.

    Where a taxon has several ids (a few unidentified groups), the lowest.
    """
    t = taxonomy.dropna(subset=["trektellen_species_id"])
    ids = t["trektellen_species_id"].astype(str).str.split(",").str[0].astype(int)
    return dict(zip(t["english_name"], ids))


# ---------------------------------------------------------------------------------------
# Model counts
# ---------------------------------------------------------------------------------------


@dataclass
class Step:
    """One processing rule and the rows it touched."""

    source: str
    name: str
    action: str  # "removed" | "modified" | "added" | "merged" | "not used"
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
    # Trektellen survey windows split into clock hours: columns date, start, end (UTC).
    split_windows: pd.DataFrame
    # Main-direction birds in the dataset (before any step), by year, per source.
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


def civil_twilight(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Civil dawn and dusk at Defile on each local date, UTC: the first minute of the day with the
    sun at or above -6 deg, and the minute after the last (as defile-dataset computed them)."""
    lat, lon = get_lat_lon("Defile")
    days = pd.DatetimeIndex(pd.to_datetime(dates.unique()))
    midnight = days.tz_localize(TIMEZONE).tz_convert("UTC")
    minutes = np.arange(24 * 60).astype("timedelta64[m]")
    grid = pd.DatetimeIndex((midnight.values[:, None] + minutes[None, :]).ravel())
    altitude = np.degrees(np.asarray(get_position(grid, lon[0], lat[0])["altitude"]))
    day = altitude.reshape(len(days), -1) >= NIGHT_SUN_ALTITUDE
    first = day.argmax(axis=1)
    last = day.shape[1] - 1 - day[:, ::-1].argmax(axis=1)
    dawn = dict(zip(days, midnight + pd.to_timedelta(first, unit="min")))
    dusk = dict(zip(days, midnight + pd.to_timedelta(last + 1, unit="min")))
    return dates.map(dawn), dates.map(dusk)


def clip_to_twilight(counts: pd.DataFrame, surveys: pd.DataFrame, src: str, log: ProcessingLog):
    """Trektellen counts reaching well into the night start at civil dawn / end at civil dusk.

    Entries timed outside the clipped interval are then handled by `fit_times_to_survey`.

    A count closed hours after dark, sometimes the next morning, is an entry error: zero-filled,
    its night hours would be effort nobody watched. This is the rule the dataset applied until its
    2026-10 release, which keeps the recorded times pending correction.

    Returns (counts, surveys).
    """
    dawn, dusk = civil_twilight(surveys["date"])
    early = surveys["start"] < dawn - NIGHT_TOLERANCE
    late = surveys["end"] > dusk + NIGHT_TOLERANCE
    log.record(
        src,
        "Survey clipped to twilight",
        "modified",
        f"A survey starting more than {fmt_duration(NIGHT_TOLERANCE)} before civil dawn, or "
        f"ending more than {fmt_duration(NIGHT_TOLERANCE)} after civil dusk (sun at -6 deg), "
        "starts at dawn / ends at dusk: a count left open in the dark is an entry error.",
        surveys.loc[early | late, ["survey_id", "date", "start", "end"]].assign(
            dawn=dawn[early | late], dusk=dusk[early | late]
        ),
    )
    surveys = surveys.assign(
        start=surveys["start"].where(~early, dawn), end=surveys["end"].where(~late, dusk)
    )
    return counts, surveys


def fit_times_to_survey(counts: pd.DataFrame, surveys: pd.DataFrame, src: str, log: ProcessingLog):
    """Entries timed less than `TIMESTAMP_TOLERANCE` outside their survey move just inside it;
    those further out are dropped.

    The survey's end is exclusive, so an entry at its very minute is outside by zero and moves in.
    """
    bounds = surveys.set_index("survey_id")
    t = counts["datetime"]
    start, end = counts["survey_id"].map(bounds["start"]), counts["survey_id"].map(bounds["end"])
    early = t.notna() & (t < start) & ((start - t) < TIMESTAMP_TOLERANCE)
    late = t.notna() & (t >= end) & ((t - end) < TIMESTAMP_TOLERANCE)
    log.record(
        src,
        "Timed just outside its survey",
        "modified",
        f"An entry timed less than {fmt_duration(TIMESTAMP_TOLERANCE)} before its survey starts or "
        f"after it ends (clock rounding, a late entry) is moved {fmt_duration(TIMESTAMP_NUDGE)} "
        "inside it.",
        counts[early | late],
    )
    t = t.where(~early, start + TIMESTAMP_NUDGE).where(~late, end - TIMESTAMP_NUDGE)
    counts = counts.assign(datetime=t)
    outside = t.notna() & ((t < start) | (t >= end))
    log.record(
        src,
        "Timed outside its survey",
        "removed",
        f"An entry timed {fmt_duration(TIMESTAMP_TOLERANCE)} or more outside its survey "
        "(after twilight clipping) belongs to no counted period.",
        counts[outside],
    )
    return counts[~outside]


def _with_survey(counts: pd.DataFrame, surveys: pd.DataFrame) -> pd.DataFrame:
    out = counts.merge(surveys[["survey_id", "start", "end"]], on="survey_id", how="left")
    assert len(out) == len(counts)
    return out


def usable_surveys(counts: pd.DataFrame, surveys: pd.DataFrame, src: str, log: ProcessingLog):
    """Drop the surveys the model cannot use (`EXCLUDED_COVERAGE`), with their counts, then the
    presence-only counts and those timed outside their survey.

    Returns (counts, surveys).
    """
    for coverage, rule in EXCLUDED_COVERAGE.items():
        hit = surveys["survey_coverage"] == coverage
        in_hit = counts["survey_id"].isin(surveys.loc[hit, "survey_id"])
        if coverage == "none":
            # No counts by construction; the surveys themselves are what is not used.
            assert not in_hit.any(), "counts in a survey with coverage 'none'"
            rows = surveys.loc[hit, ["survey_id", "date", "start", "end", "survey_coverage"]]
            log.record(src, "Survey not counted", "not used", rule, rows)
        else:
            log.record(src, f"Survey coverage {coverage}", "removed", rule, counts[in_hit])
        surveys, counts = surveys[~hit], counts[~in_hit]
    presence = counts["count_estimation"] == PRESENCE_ONLY
    log.record(
        src,
        "Presence only",
        "removed",
        "A record of presence without a number is never turned into a count.",
        counts[presence],
    )
    counts = counts[~presence]
    outside = outside_survey(counts) & counts["datetime"].isna()
    log.record(
        src,
        "Timed outside its survey, time unknown",
        "removed",
        "An entry the dataset marks as timed outside its survey, whose recorded time is not in "
        f"`{ENTRY_TIMES_FILE}`: the tolerance cannot be applied, and it belongs to no counted "
        "period.",
        counts[outside],
    )
    return counts[~outside], surveys


def historical_model_counts(counts: pd.DataFrame, surveys: pd.DataFrame, log: ProcessingLog):
    """Historical counts -> model counts: one row per taxon and survey period.

    Historical counts carry no timing of their own: each takes its survey's interval. Days recorded
    hour by hour are one survey per hour in the dataset, so they need no splitting here.
    """
    src = "historical"
    df = _with_survey(counts, surveys)
    # Grouped by taxon id: two ids sharing an English name would stay two rows, which the "One row
    # per species and period" check would catch.
    keys = ["taxon_id", "species", "date", "start", "end"]
    merged = df.groupby(keys, as_index=False, dropna=False).agg(count=("count", "sum"))
    log.record(
        src,
        "Same species, same period",
        "merged",
        "Several records of one taxon in one period (age/sex subgroups, repeated entries) are "
        f"summed into one row. {len(df) - len(merged)} rows merged; no bird lost.",
        df[df.duplicated(keys, keep=False)],
    )
    return merged[OUTPUT_COLUMNS]


def trektellen_model_counts(counts: pd.DataFrame, surveys: pd.DataFrame, log: ProcessingLog):
    """Trektellen counts -> model counts.

    Returns (counts, split windows).
    """
    src = "trektellen"
    df = _with_survey(counts, surveys)

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
        "In a period split into hours, a sighting without a timestamp cannot be placed in an "
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
    zeros = slots[empty].assign(species=NO_SPECIES, count=0)
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
    df = df[~short]

    keys = ["species", "date", "start", "end"]
    merged = df.groupby(keys, as_index=False, dropna=False)["count"].sum()
    log.record(
        src,
        "Same species, same period",
        "merged",
        f"Sightings of one species in one period are summed into one row. "
        f"{len(df) - len(merged)} rows merged; no bird lost.",
        df[df.duplicated(keys, keep=False)],
    )
    return merged[OUTPUT_COLUMNS], windows


def zero_fill_empty_surveys(
    counts: pd.DataFrame, surveys: pd.DataFrame, src: str, log: ProcessingLog
) -> pd.DataFrame:
    """A counted survey with no record at all is a period counted with nothing seen.

    Only `complete` surveys reach this point (`usable_surveys`): days without counting are `none`
    in the dataset, so an empty survey here is a real zero, even alone on its day (an explicit 'no
    species' entry).
    """
    long_enough = (surveys["end"] - surveys["start"]) >= MIN_PERIOD_DURATION
    empty = surveys[long_enough]
    free = ~overlaps_any(empty["start"], empty["end"], counts["start"], counts["end"])
    zeros = empty[free][["date", "start", "end"]].assign(species=NO_SPECIES, count=0)
    log.record(
        src,
        "Zero-fill empty surveys",
        "added",
        f"A counted survey (at least {fmt_duration(MIN_PERIOD_DURATION)}) that no record "
        f"overlaps gets a '{NO_SPECIES}' row with count 0, so it exists as effort.",
        zeros,
    )
    return pd.concat([counts, zeros[OUTPUT_COLUMNS]], ignore_index=True) if len(zeros) else counts


def build_model_counts(surveys: pd.DataFrame, counts: pd.DataFrame) -> ModelCounts:
    log = ProcessingLog()
    counts = counts[counts["count_category"] == MAIN_CATEGORY]
    parts, raw, windows = [], {}, None
    for src in SOURCES:
        c, s = counts[counts["source"] == src], surveys[surveys["source"] == src]
        raw[src] = c.groupby(c["date"].dt.year)["count"].sum()
        c, s = usable_surveys(c, s, src, log)
        if src == "trektellen":
            c, s = clip_to_twilight(c, s, src, log)
            c = fit_times_to_survey(c, s, src, log)
            out, windows = trektellen_model_counts(c, s, log)
        else:
            out = historical_model_counts(c, s, log)
        parts.append(zero_fill_empty_surveys(out, s, src, log))
    out = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["start", "end", "species"], kind="stable")
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
        [mc.split_windows.assign(source="trektellen", split=True), p], ignore_index=True
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

    dup = c[c.duplicated(["species", "start", "end"], keep=False)]
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
            "pass" if unm.empty else "fail",
            f"{len(unm)} row(s), {unm['count'].sum():.0f} birds, have no species name (taxon "
            "without an English name in the dataset's taxonomy).",
            unm,
        )
    )

    # In windows split into hours, every clock hour should exist as a period.
    slots = hourly_slots(mc.split_windows)
    slots = slots[(slots["end"] - slots["start"]) >= MIN_PERIOD_DURATION]
    covered = overlaps_any(slots["start"], slots["end"], periods["start"], periods["end"])
    missing = slots[~covered]
    checks.append(
        Check(
            "Surveyed hours with no row",
            "pass" if missing.empty else "fail",
            f"{len(missing)} clock hour(s) (>= {fmt_duration(MIN_PERIOD_DURATION)}) inside a "
            "Trektellen window split into hours that no output period covers: surveyed, but "
            "missing as effort, so their zero counts are lost.",
            missing,
        )
    )

    # A day whose only effort is empty surveys: the dataset calls it counted ('complete'), but it
    # is worth a look, since a day of no counting entered as one empty survey would look the same.
    zeros = pd.concat(
        [s.rows for s in mc.log.steps if s.name == "Zero-fill empty surveys"], ignore_index=True
    )
    birds_days = c.loc[c["species"] != NO_SPECIES, "date"]
    alone = zeros[~zeros["date"].isin(birds_days)]
    checks.append(
        Check(
            "Empty surveys alone on their day",
            "pass" if alone.empty else "warn",
            f"{len(alone)} empty survey(s), zero-filled as counted, on a day with no bird record: "
            "real zeros only if the day was counted (an explicit 'no species' entry), else "
            "the dataset should mark them coverage 'none'.",
            alone,
        )
    )
    return checks
