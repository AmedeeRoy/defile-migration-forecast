"""Tests for the model's count processing (src/data/counts.py), on small synthetic dataset tables.

Reading and correcting the raw files is tested in defile-dataset. Here, the input is the shape of
its two tables (`surveys`, `observations`), and the real data is checked by `check_model_counts` on
every `scripts/build_counts.py` run.
"""

import numpy as np
import pandas as pd

from src.data import counts as C

DAY = "2023-08-01"


def utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local).tz_localize(C.TIMEZONE).tz_convert("UTC")


def _taxon(o: pd.DataFrame) -> pd.DataFrame:
    """The dataset's taxon columns, from a `species` column (the model's name)."""
    o["taxon_name_original"] = o["ebird_english_name"] = o["species"]
    o["english_name"] = None
    o["taxon_kind"] = "bird"
    return o


def _trektellen(entries: list[tuple], surveys: list[tuple]):
    """entries: (survey id, local time or None, english name, count, flags);
    surveys: (survey id, local start, local end)."""
    o = pd.DataFrame(entries, columns=["survey_id", "t", "species", "count", "flags"])
    o["source"], o["date"] = "trektellen", pd.Timestamp(DAY)
    o = _taxon(o)
    local = pd.to_datetime(DAY + " " + o["t"].fillna("00:00")).where(o["t"].notna())
    o["datetime"] = local.dt.tz_localize(C.TIMEZONE).dt.tz_convert("UTC")
    s = pd.DataFrame(surveys, columns=["survey_id", "start", "end"])
    s["start"] = s["start"].map(lambda t: utc(f"{DAY} {t}"))
    s["end"] = s["end"].map(lambda t: utc(f"{DAY} {t}"))
    s["source"], s["flags"], s["duplicate_of"] = "trektellen", "", None
    s["date"] = pd.Timestamp(DAY)
    return s, o.drop(columns="t")


def _historical(rows: list[tuple], day=("08:00", "12:00")):
    """Rows: (English name, local start, local end, count), all on one day."""
    o = pd.DataFrame(rows, columns=["species", "start", "end", "count"])
    o = _taxon(o)
    o["source"], o["date"], o["flags"] = "historical", pd.Timestamp("2015-09-15"), ""
    o["survey_id"] = "H" + o["start"] + o["end"]
    s = o[["survey_id", "start", "end"]].drop_duplicates().copy()
    for c in ("start", "end"):
        s[c] = s[c].map(lambda t: utc(f"2015-09-15 {t}"))
    s["day_start"], s["day_end"] = utc(f"2015-09-15 {day[0]}"), utc(f"2015-09-15 {day[1]}")
    s["source"], s["flags"], s["duplicate_of"] = "historical", "", None
    s["date"] = pd.Timestamp("2015-09-15")
    return s, o.drop(columns=["start", "end"])


def test_hourly_slots_keeps_partial_first_and_last_hours():
    w = pd.DataFrame({"start": [utc(f"{DAY} 06:40")], "end": [utc(f"{DAY} 09:20")]})
    slots = C.hourly_slots(w)
    assert list(slots["start"].dt.tz_convert(C.TIMEZONE).dt.strftime("%H:%M")) == [
        "06:40",
        "07:00",
        "08:00",
        "09:00",
    ]
    assert (slots["end"] - slots["start"]).sum() == w["end"].iloc[0] - w["start"].iloc[0]


def test_overlaps_any_matches_brute_force():
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp(DAY, tz="UTC")

    def intervals(n):
        a = t0 + pd.to_timedelta(rng.integers(0, 600, n), unit="min")
        return a, a + pd.to_timedelta(rng.integers(1, 120, n), unit="min")

    s, e = intervals(200)
    rs, re_ = intervals(50)
    expected = [((rs < b) & (re_ > a)).any() for a, b in zip(s, e)]
    assert list(C.overlaps_any(s, e, rs, re_)) == expected


def test_trektellen_split_into_hours_zero_fill_and_rules():
    surveys, obs = _trektellen(
        [
            ("T10", "06:30", "Red Kite", 5, ""),  # hour 06
            ("T10", "06:45", "Red Kite", 3, ""),  # hour 06, summed with the one above
            ("T10", "05:00", "Red Kite", 7, "time_outside_survey"),  # flagged: dropped
            ("T10", None, "Red Kite", 9, ""),  # untimed in a split period: dropped
            ("T10", "09:03", "Red Kite", 6, ""),  # 09:00-09:05 fragment < 10 min: dropped
            ("T10", "08:15", None, 1, ""),  # taxon without English name: kept, unnamed
        ],
        [("T10", "06:00", "09:05")],
    )
    log = C.ProcessingLog()
    out, windows = C.trektellen_model_counts(obs, surveys, log)

    named = out[out["species"].notna()]
    by_hour = named.groupby([named["start"].dt.tz_convert(C.TIMEZONE).dt.hour, "species"])
    assert by_hour["count"].sum().to_dict() == {(6, "Red Kite"): 8, (7, C.NO_SPECIES): 0}
    assert out["species"].isna().sum() == 1  # the unnamed one, 08:00-09:00
    assert len(windows) == 1

    removed = {s.name: s.birds for s in log.steps if s.action == "removed"}
    assert removed == {
        "Flagged time_outside_survey": 7,
        "Untimed sighting in split period": 9,
        f"Period shorter than {C.fmt_duration(C.MIN_PERIOD_DURATION)}": 6,
    }
    # Every bird is either in the output or removed by a named step.
    assert out["count"].sum() + sum(removed.values()) == obs["count"].sum()


def test_trektellen_mostly_untimed_period_is_not_split():
    surveys, obs = _trektellen(
        [("T10", "07:30", "Red Kite", 1, ""), ("T10", None, "Red Kite", 4, "")] * 2,
        [("T10", "06:00", "12:00")],
    )
    out, windows = C.trektellen_model_counts(obs, surveys, C.ProcessingLog())
    assert windows.empty
    assert out["count"].sum() == 10
    assert (out["end"] - out["start"]).iloc[0] == pd.Timedelta(hours=6)


def test_historical_zero_fills_empty_hours_of_an_hourly_day():
    surveys, obs = _historical(
        [
            ("Red Kite", "08:00", "09:00", 3),
            ("Red Kite", "08:00", "09:00", 2),  # same period: summed
            ("Red Kite", "10:00", "11:00", 1),
        ]
    )
    out, windows = C.historical_model_counts(obs, surveys, C.ProcessingLog())
    by_hour = out.set_index(out["start"].dt.tz_convert(C.TIMEZONE).dt.hour)
    assert by_hour.loc[8, "count"] == 5
    assert by_hour.loc[9, "species"] == C.NO_SPECIES and by_hour.loc[9, "count"] == 0
    assert by_hour.loc[11, "species"] == C.NO_SPECIES
    assert len(windows) == 1


def test_historical_whole_day_record_is_not_zero_filled():
    surveys, obs = _historical([("Red Kite", "08:00", "12:00", 3)])
    out, windows = C.historical_model_counts(obs, surveys, C.ProcessingLog())
    assert windows.empty
    assert list(out["species"]) == ["Red Kite"]


def test_model_species_prefers_ebird_names():
    obs = pd.DataFrame(
        {
            "ebird_english_name": ["Eurasian Kestrel", None, None, None],
            "english_name": ["Common Kestrel", "Carrion Crow (cornix)", None, None],
            "taxon_kind": ["bird", "bird", "no_species", "non_bird"],
        }
    )
    assert C.model_species(obs).tolist() == [
        "Eurasian Kestrel",
        "Carrion Crow (cornix)",
        C.NO_SPECIES,
        C.NON_BIRD,
    ]


def test_checks_pass_on_clean_input():
    hs, ho = _historical([("Red Kite", "08:00", "09:00", 3), ("Red Kite", "10:00", "11:00", 1)])
    ts, to = _trektellen(
        [("T10", "06:30", "Red Kite", 5, ""), ("T10", "08:30", "Red Kite", 1, "")],
        [("T10", "06:00", "09:00")],
    )
    obs = pd.concat([ho, to]).drop(columns="species")
    mc = C.build_model_counts(pd.concat([hs, ts]), obs)
    status = {c.name: c.status for c in C.check_model_counts(mc)}
    assert status["Birds accounted for"] == "pass"
    assert status["No overlapping periods"] == "pass"
    assert status["Surveyed hours with no row"] == "pass"


def test_empty_survey_is_a_zero_only_on_a_counted_day():
    surveys, obs = _trektellen(
        [("T10", "06:30", "Red Kite", 5, "")],
        [("T10", "06:00", "07:00"), ("T11", "07:00", "08:00"), ("T12", "06:00", "18:00")],
    )
    surveys.loc[surveys["survey_id"].isin(["T11", "T12"]), "flags"] = C.FLAG_NO_ENTRIES
    alone = surveys["survey_id"] == "T12"  # the next day, alone on it
    surveys.loc[alone, "date"] = pd.Timestamp("2023-08-02")
    surveys.loc[alone, ["start", "end"]] += pd.Timedelta(days=1)
    log = C.ProcessingLog()
    counts, _ = C.trektellen_model_counts(obs, surveys, log)
    out = C.zero_fill_empty_surveys(counts, surveys, "trektellen", log)
    zeros = out[out["species"] == C.NO_SPECIES]
    assert zeros["start"].tolist() == [utc(f"{DAY} 07:00")]  # T11; not the empty day T12
    not_used = next(s for s in log.steps if s.name == "Empty survey on a day not counted")
    assert not_used.rows["survey_id"].tolist() == ["T12"]
