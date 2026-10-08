"""Explore export: the defile-dataset release tables -> the JSON files defileViz's Explore page
reads.

Unlike `src/data/counts.py`, nothing here is a model choice: every value is a plain aggregation
of the release tables (`count.csv`, `survey.csv`, `taxonomy.csv`, `report_text.csv`), following
the dataset README's daily-total rules (a count's own timing, else its survey's; the local
Europe/Paris day). No count is moved, dropped, imputed or redistributed, so raw daily and annual
totals reconcile with `count.csv` (`tests/test_explore.py`). Entries the release keeps at day
level (untimed, or timed outside their survey) stay untimed here.

Files, written by `scripts/build_explore.py` (one `build_*` function each):

- `manifest.json`: when and from which dataset build it was made, the date range, partial years,
  the default window and the tier thresholds.
- `taxa.json`: one entry per taxon, with names (French from the eBird taxonomy), rank,
  occurrence tier, totals, and `start_year`, the first year its counts are comparable from.
- `effort.json`: one entry per local day with any survey: counted hours (union of `complete`
  survey intervals), first/last counted time, hours counted per local clock hour, and the hours of
  `partial`, `unknown` and `none` (not counted) surveys.
- `species/<taxon_id>.json`: per day, the main-direction count with its qualifiers, the reverse
  and local counts, the share of birds timed to the hour, the day's coverage `c` and the
  effort-adjusted count; per local hour, the birds timed to it; per year, totals and the
  effort-adjusted index; the time-of-day profile used; the species' report texts.
- `reports.json`: the report texts that are not about one species (site, monitoring, weather,
  results, outreach).

Tables are columnar (`{"date": [...], "count": [...]}`) to keep the files small.
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from src.data.counts import (
    DATASET_DIR,
    MAIN_CATEGORY,
    METADATA_FILE,
    PRESENCE_ONLY,
    REPORT_FILE,
    TIMEZONE,
    parse_counts,
    parse_surveys,
)
from src.metrics import ERA_EDGES

# Bump when a field's meaning changes, so defileViz can refuse an export it does not understand.
DEFINITIONS_VERSION = 1

# Default comparison window (month, day), inclusive. A window for annual totals, not a filter:
# every day is exported.
WINDOW = ((7, 18), (11, 18))

# Occurrence tiers. "full": a dashboard; "short": records, annual totals and report links;
# "rare": the rare-records table only. On days with migrating birds (`tier_of`): a seasonal curve
# needs days, not birds (one large flock is still a rare record). Provisional, to tune on the page.
TIER_FULL_MIN_DAYS = 50
TIER_FULL_MIN_YEARS = 5
TIER_RARE_MAX_DAYS = 10

# Start year: the first year a taxon's counts are comparable from (`start_year`); earlier years
# are exported raw but get no adjusted value, and defileViz crops its annual series there.
# Systematic daily counting began in 1993. Its targets were always raptors, herons and egrets,
# pigeons and corvids (and other large birds, counted individually); passerines were hardly
# recorded before 2007, when the number of passerine taxa doubles and their birds rise 15-fold
# (defile-dataset docs/sampling-history.md). Targets start in 1993 even when rare then (Peregrine
# recovered); any other taxon starts in 1993 only if recorded in at least `START_MIN_SHARE` as
# large a share of the 1993-2006 years as of the later ones, else in 2007.
SYSTEMATIC_FROM = ERA_EDGES[0]
ALL_GROUPS_FROM = 2007
TARGET_ORDERS = {
    "Accipitriformes",
    "Falconiformes",
    "Pelecaniformes",
    "Ciconiiformes",
    "Columbiformes",
}
TARGET_FAMILIES = {"Corvidae"}
START_MIN_SHARE = 0.75

# French names: the eBird taxonomy in French (France), joined on `ebird_code`; no API key needed.
# Downloaded once into the data dir by `scripts/build_explore.py` (`--refresh-names` to update).
EBIRD_TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&locale=fr_FR"
EBIRD_TAXONOMY_FILE = os.path.join("count", "ebird_taxonomy_fr_FR.csv")
# Taxa whose `ebird_code` is no longer in the eBird taxonomy.
FRENCH_NAME_OVERRIDES = {"avibase-81B32602": "Corneille mantelée"}  # hoocro4, Corvus c. cornix

# A count is "timed" when its own time, or its survey, places it within one local clock hour.
TIMED_MAX_DURATION = pd.Timedelta(hours=1)

COVERAGE_STATES = ("complete", "partial", "unknown", "none")
REPORT_SPECIES = "species"  # report_text category whose key is a taxon_id

HOURS = 24
DECIMALS = 3  # hours and fractions


# --- local time -------------------------------------------------------------


def to_local(t: pd.Series) -> pd.Series:
    """UTC times -> naive local wall-clock times."""
    return t.dt.tz_convert(TIMEZONE).dt.tz_localize(None)


def split_at_midnight(start: pd.Series, end: pd.Series) -> pd.DataFrame:
    """Local intervals -> pieces within one local day (`index` points back to the input row)."""
    pieces = []
    for i, s, e in zip(start.index, start, end):
        while s < e:
            cut = min(e, s.normalize() + pd.Timedelta(days=1))
            pieces.append((i, s, cut))
            s = cut
    return pd.DataFrame(pieces, columns=["index", "start", "end"]).set_index("index")


def merge_intervals(start: np.ndarray, end: np.ndarray) -> list[tuple]:
    """Union of intervals, as sorted non-overlapping (start, end) pairs."""
    out = []
    for s, e in sorted(zip(start, end)):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def hours_per_clock_hour(intervals: list[tuple]) -> np.ndarray:
    """Hours covered in each of the 24 local clock hours of a day by (start, end) intervals."""
    cover = np.zeros(HOURS)
    for s, e in intervals:
        day = s.normalize()
        a, b = (s - day) / pd.Timedelta(hours=1), (e - day) / pd.Timedelta(hours=1)
        for h in range(int(np.floor(a)), min(int(np.ceil(b)), HOURS)):
            cover[h] += max(0.0, min(b, h + 1) - max(a, h))
    return cover


# --- effort -----------------------------------------------------------------


def build_effort(surveys: pd.DataFrame) -> pd.DataFrame:
    """One row per local day with any survey.

    `hours` is the union of `complete` survey intervals (never a sum of nested or overlapping
    ones); `first`/`last` its local bounds; `periods` the number of complete surveys; `hourly` the
    hours counted in each local clock hour; `partial_hours`, `unknown_hours`, `none_hours` the
    other surveys' hours. `state`: `counted` (some complete hours), else `uncertain` (only partial
    or unknown), else `not_counted` (only `none`: rain, closures).
    """
    s = surveys.assign(lstart=to_local(surveys["start"]), lend=to_local(surveys["end"]))
    pieces = split_at_midnight(s["lstart"], s["lend"])
    pieces = pieces.join(s[["survey_coverage", "recording_era"]])
    pieces["date"] = pieces["start"].dt.normalize()

    rows = []
    for date, day in pieces.groupby("date"):
        row = {"date": date}
        complete = day[day["survey_coverage"] == "complete"]
        union = merge_intervals(complete["start"].tolist(), complete["end"].tolist())
        hourly = hours_per_clock_hour(union)
        row["hours"] = hourly.sum()
        row["first"] = union[0][0] if union else None
        row["last"] = union[-1][1] if union else None
        row["periods"] = len(complete)
        row["hourly"] = hourly
        for state in COVERAGE_STATES[1:]:
            part = day[day["survey_coverage"] == state]
            union_s = merge_intervals(part["start"].tolist(), part["end"].tolist())
            row[f"{state}_hours"] = sum((e - b) / pd.Timedelta(hours=1) for b, e in union_s)
        eras = day.loc[day["recording_era"] != "curated", "recording_era"]
        row["era"] = eras.mode().iloc[0] if len(eras) else "curated"
        rows.append(row)
    e = pd.DataFrame(rows)
    e["state"] = np.select(
        [e["hours"] > 0, (e["partial_hours"] + e["unknown_hours"]) > 0],
        ["counted", "uncertain"],
        "not_counted",
    )
    return e


def annual_effort(effort: pd.DataFrame) -> pd.DataFrame:
    """Per year: days by `state`, counted hours, and the same in the default window."""
    e = effort.assign(year=effort["date"].dt.year, win=in_window(effort["date"]))
    counted = e["state"] == "counted"
    return (
        e.assign(
            days=counted,
            uncertain_days=e["state"] == "uncertain",
            not_counted_days=e["state"] == "not_counted",
            window_days=counted & e["win"],
            window_hours=e["hours"].where(e["win"], 0),
        )
        .groupby("year")[
            ["days", "hours", "window_days", "window_hours", "uncertain_days", "not_counted_days"]
        ]
        .sum()
        .reset_index()
    )


# --- counts -----------------------------------------------------------------


def timed_hour(counts: pd.DataFrame, surveys: pd.DataFrame) -> pd.Series:
    """Local clock hour (0-23) each count is timed to, or NaN.

    Timed: the count's own time (a point), or, without one, a survey of at most an hour within
    one local clock hour (historical hour-by-hour records). Day-level entries are untimed.
    """
    sv = surveys.set_index("survey_id")
    start = to_local(counts["survey_id"].map(sv["start"]))
    end = to_local(counts["survey_id"].map(sv["end"]))
    one_hour = ((end - start) <= TIMED_MAX_DURATION) & (
        start.dt.floor("h") == (end - pd.Timedelta(seconds=1)).dt.floor("h")
    )
    own = counts["datetime"].notna()
    inherits = counts["raw_datetime"].isna()
    hour = pd.Series(np.nan, index=counts.index)
    hour[own] = to_local(counts.loc[own, "datetime"]).dt.hour
    hour[inherits & one_hour] = start[inherits & one_hour].dt.hour
    return hour


def daily_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """One row per taxon and local day with any count row.

    `count`: main-direction birds (NaN if only presence-only records); `reverse`, `local`: NaN when
    no row (unknown, not zero); `qualifiers`: the day's distinct `count_estimation` codes on main-
    direction rows, sorted (e.g. `"~>"`); `timed`: the share of main-direction birds timed to the
    hour (NaN without birds).
    """
    normal = counts[counts["count_category"] == MAIN_CATEGORY]
    key = ["taxon_id", "date"]
    d = normal.groupby(key).agg(
        count=("count", lambda x: x.sum(min_count=1)),
        timed_birds=("timed_count", "sum"),
        qualifiers=("count_estimation", lambda x: "".join(sorted(set(x.dropna())))),
    )
    for cat in ("reverse", "local"):
        rows = counts[counts["count_category"] == cat]
        d = d.join(rows.groupby(key)["count"].sum().rename(cat), how="outer")
    d["qualifiers"] = d["qualifiers"].fillna("")
    d["timed"] = (d["timed_birds"] / d["count"]).where(d["count"] > 0)
    return d.drop(columns="timed_birds").reset_index()


def hourly_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Main-direction birds per taxon, local day and local clock hour, from timed counts only."""
    timed = counts[(counts["count_category"] == MAIN_CATEGORY) & counts["hour"].notna()]
    h = timed.groupby(["taxon_id", "date", "hour"])["count"].sum().reset_index()
    h["hour"] = h["hour"].astype(int)
    return h


def in_window(dates: pd.Series) -> pd.Series:
    """Whether local dates fall in the default comparison window `WINDOW`."""
    md = dates.dt.month * 100 + dates.dt.day
    (m0, d0), (m1, d1) = WINDOW
    return (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)


def annual(days: pd.DataFrame) -> pd.DataFrame:
    """Per year: total birds, birds in the window, days with a record, and the top day."""
    d = days.assign(year=days["date"].dt.year, win=in_window(days["date"]))
    d["win_count"] = d["count"].where(d["win"])
    g = d.groupby("year")
    a = g.agg(
        total=("count", "sum"),
        window=("win_count", "sum"),
        days=("date", "size"),
    )
    top = d.loc[d["count"].fillna(-1).groupby(d["year"]).idxmax()]
    a["max"] = top.set_index("year")["count"]
    a["max_date"] = top.set_index("year")["date"]
    return a.reset_index()


# --- taxa -------------------------------------------------------------------


def tier_of(years: int, days: int) -> str:
    """Occurrence tier from the years and days with migrating birds (thresholds above)."""
    if days <= TIER_RARE_MAX_DAYS:
        return "rare"
    if days >= TIER_FULL_MIN_DAYS and years >= TIER_FULL_MIN_YEARS:
        return "full"
    return "short"


def start_year(order: str, family: str, years: set[int], last_year: int) -> int:
    """First comparable year of a taxon (see `SYSTEMATIC_FROM`), from its order, family and the
    years with migrating birds; `last_year` is the last year of the data."""
    if order in TARGET_ORDERS or family in TARGET_FAMILIES:
        return SYSTEMATIC_FROM
    before = sum(SYSTEMATIC_FROM <= y < ALL_GROUPS_FROM for y in years)
    after = sum(ALL_GROUPS_FROM <= y <= last_year for y in years)
    share_before = before / (ALL_GROUPS_FROM - SYSTEMATIC_FROM)
    share_after = after / (last_year - ALL_GROUPS_FROM + 1)
    systematic = before > 0 and share_before >= START_MIN_SHARE * share_after
    return SYSTEMATIC_FROM if systematic else ALL_GROUPS_FROM


def french_names(taxonomy: pd.DataFrame, ebird: pd.DataFrame) -> pd.Series:
    """French name of each taxon (eBird `COMMON_NAME` by `ebird_code`), first letter capitalised
    ("labbe sp." -> "Labbe sp."), with `FRENCH_NAME_OVERRIDES`."""
    names = taxonomy["ebird_code"].map(ebird.set_index("SPECIES_CODE")["COMMON_NAME"])
    names = names.fillna(taxonomy["taxon_id"].map(FRENCH_NAME_OVERRIDES))
    return names.str[:1].str.upper() + names.str[1:]


def build_taxa(
    taxonomy: pd.DataFrame, days: pd.DataFrame, ebird: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Taxonomy plus French name and occurrence: years and days with migrating birds (or a
    presence-only record), first/last year, total birds, tier and `start_year`."""
    d = days[(days["count"] > 0) | days["qualifiers"].str.contains(PRESENCE_ONLY)]
    occ = d.groupby("taxon_id").agg(
        years=("date", lambda x: x.dt.year.nunique()),
        first_year=("date", lambda x: x.dt.year.min()),
        last_year=("date", lambda x: x.dt.year.max()),
        days=("date", "size"),
        birds=("count", "sum"),
    )
    t = taxonomy[
        ["taxon_id", "english_name", "scientific_name", "taxon_rank", "order", "family"]
    ].join(occ, on="taxon_id")
    if ebird is not None:
        t.insert(1, "french_name", french_names(taxonomy, ebird))
    t["tier"] = [tier_of(y, n) if pd.notna(n) else "rare" for y, n in zip(t["years"], t["days"])]
    recorded = d.groupby("taxon_id")["date"].agg(lambda x: set(x.dt.year))
    last = int(days["date"].max().year)
    t["start_year"] = [
        start_year(o, f, recorded.get(i, set()), last)
        for i, o, f in zip(t["taxon_id"], t["order"], t["family"])
    ]
    return t.sort_values(["order", "family", "scientific_name"], na_position="last")


# --- assembly ---------------------------------------------------------------


def release_counts(count: pd.DataFrame, surveys: pd.DataFrame, taxonomy: pd.DataFrame):
    """`count.csv` parsed (`counts.parse_counts`, without entry times), plus its own
    `raw_datetime`, the local clock `hour` it is timed to (`timed_hour`) and its `timed_count`."""
    c = parse_counts(count, surveys, taxonomy)
    c["raw_datetime"] = count["datetime"].to_numpy()
    c["hour"] = timed_hour(c, surveys)
    c["timed_count"] = c["count"].where(c["hour"].notna(), 0).fillna(0)
    return c


def read_release(data_dir: str):
    """`(surveys, counts, taxonomy, reports, metadata)` from `<data_dir>/count/dataset/`.

    The release tables only: `entry_times.csv`, the model's extra, is not read. `counts` is
    `release_counts`.
    """
    folder = os.path.join(data_dir, DATASET_DIR)
    surveys = parse_surveys(pd.read_csv(os.path.join(folder, "survey.csv"), low_memory=False))
    taxonomy = pd.read_csv(os.path.join(folder, "taxonomy.csv"))
    count = pd.read_csv(os.path.join(folder, "count.csv"), low_memory=False)
    c = release_counts(count, surveys, taxonomy)
    reports = pd.read_csv(os.path.join(folder, REPORT_FILE), encoding="utf-8-sig")
    path = os.path.join(folder, METADATA_FILE)
    metadata = json.load(open(path)) if os.path.exists(path) else {}
    return surveys, c, taxonomy, reports, metadata


def partial_years(days: pd.DataFrame) -> list[int]:
    """The last year, if its last record is before the window's end (a season in progress)."""
    last = days["date"].max()
    (m1, d1) = WINDOW[1]
    return [int(last.year)] if (last.month, last.day) < (m1, d1) else []


def manifest(metadata: dict, days: pd.DataFrame, effort: pd.DataFrame, git_sha: str) -> dict:
    """When and from what the export was built, and the definitions it uses."""
    return {
        "definitions_version": DEFINITIONS_VERSION,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "dataset_built_at": metadata.get("built_at"),
        "dataset_git_sha": metadata.get("git_sha"),
        "first_date": effort["date"].min(),
        "last_date": max(effort["date"].max(), days["date"].max()),
        "partial_years": partial_years(days),
        "window": [f"{m:02d}-{d:02d}" for m, d in WINDOW],
        "tiers": {
            "full_min_days": TIER_FULL_MIN_DAYS,
            "full_min_years": TIER_FULL_MIN_YEARS,
            "rare_max_days": TIER_RARE_MAX_DAYS,
        },
        "start_years": [SYSTEMATIC_FROM, ALL_GROUPS_FROM],
        "timezone": TIMEZONE,
    }


# --- JSON -------------------------------------------------------------------


def _value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if v == v.normalize() else v.strftime("%H:%M")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = round(float(v), DECIMALS)
        return int(f) if f.is_integer() else f
    if isinstance(v, np.ndarray):
        return [_value(x) for x in v]
    return v


def columns(df: pd.DataFrame) -> dict:
    """A frame as `{column: [values]}`, with dates as `YYYY-MM-DD`, times as `HH:MM`, NaN as
    null."""
    return {c: [_value(v) for v in df[c]] for c in df.columns}


def records(df: pd.DataFrame) -> list[dict]:
    """A frame as a list of `{column: value}`, same conversions as `columns`."""
    return [{c: _value(v) for c, v in row.items()} for row in df.to_dict("records")]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_value)
