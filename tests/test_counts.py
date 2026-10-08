"""Tests for the model's count processing (src/data/counts.py), on small synthetic release tables.

Reading and correcting the raw files is tested in defile-dataset. Here, the input is the shape of
its release tables (`survey.csv`, `count.csv`, `taxonomy.csv`), and the real data is checked by
`check_model_counts` on every `scripts/build_counts.py` run (and by the last test, when a release
is present in `data/count/dataset/`).
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data import counts as C

DAY = "2023-08-01"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local).tz_localize(C.TIMEZONE).tz_convert("UTC")


def iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _surveys(rows: list[tuple], era: str = "trektellen", day: str = DAY) -> pd.DataFrame:
    """Rows: (survey id, local start, local end[, coverage])."""
    out = []
    for r in rows:
        sid, a, b, coverage = (*r, "complete")[:4]
        out.append(
            {
                "survey_id": sid,
                "datetime": f"{iso(utc(f'{day} {a}'))}/{iso(utc(f'{day} {b}'))}",
                "recording_era": era,
                "survey_coverage": coverage,
            }
        )
    return C.parse_surveys(pd.DataFrame(out))


def _counts(rows: list[tuple], surveys: pd.DataFrame, day: str = DAY) -> pd.DataFrame:
    """Rows: (survey id, local time / "date" / None, species, count[, category, estimation,
    remark_processing]).

    A time is a timed entry, "date" a day-level one, None inherits the survey's interval.
    """
    out = []
    for i, r in enumerate(rows):
        sid, t, species, count, category, estimation, remark = (*r, "normal", None, None)[:7]
        dt = None if t is None else day if t == "date" else iso(utc(f"{day} {t}"))
        out.append(
            {
                "count_id": f"c{i}",
                "survey_id": sid,
                "taxon_id": f"avibase-{species}",
                "datetime": dt,
                "count": count,
                "count_category": category,
                "count_estimation": estimation,
                "remark_processing": remark,
            }
        )
    count = pd.DataFrame(out)
    names = sorted({r[2] for r in rows})
    taxonomy = pd.DataFrame({"taxon_id": [f"avibase-{n}" for n in names], "english_name": names})
    return C.parse_counts(count, surveys, taxonomy)


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


def test_parse_counts_timing_kinds():
    surveys = _surveys([("T10", "06:00", "09:00")])
    c = _counts(
        [
            ("T10", "06:30", "Red Kite", 5),
            ("T10", "date", "Red Kite", 2),
            ("T10", None, "Kite", 1),
        ],
        surveys,
    )
    assert c["datetime"].iloc[0] == utc(f"{DAY} 06:30")
    assert c["datetime"].iloc[1:].isna().all()  # day-level and inherited are both untimed
    assert (c["date"] == pd.Timestamp(DAY)).all()
    assert c["species"].tolist() == ["Red Kite", "Red Kite", "Kite"]
    assert (c["source"] == "trektellen").all()


def test_parse_counts_refuses_a_count_interval():
    surveys = _surveys([("T10", "06:00", "09:00")])
    count = pd.DataFrame(
        {
            "count_id": ["c0"],
            "survey_id": ["T10"],
            "taxon_id": ["avibase-x"],
            "datetime": ["2023-08-01T05:00:00Z/2023-08-01T06:00:00Z"],
            "count": [1],
        }
    )
    with pytest.raises(ValueError, match="own interval"):
        C.parse_counts(count, surveys, pd.DataFrame({"taxon_id": [], "english_name": []}))


def test_trektellen_split_into_hours_zero_fill_and_rules():
    surveys = _surveys([("T10", "06:00", "09:05")])
    counts = _counts(
        [
            ("T10", "06:30", "Red Kite", 5),  # hour 06
            ("T10", "06:45", "Red Kite", 3),  # hour 06, summed with the one above
            ("T10", "date", "Red Kite", 7, "normal", None, C.OUTSIDE_SURVEY_REMARK + "; ..."),
            ("T10", "date", "Red Kite", 9),  # untimed in a split period: dropped
            ("T10", "09:03", "Red Kite", 6),  # 09:00-09:05 fragment < 10 min: dropped
            ("T10", "08:15", "Kite", 1),  # another taxon, 08:00-09:00
        ],
        surveys,
    )
    log = C.ProcessingLog()
    counts, surveys = C.usable_surveys(counts, surveys, "trektellen", log)
    out, windows = C.trektellen_model_counts(counts, surveys, log)

    by_hour = out.groupby([out["start"].dt.tz_convert(C.TIMEZONE).dt.hour, "species"])
    assert by_hour["count"].sum().to_dict() == {
        (6, "Red Kite"): 8,
        (7, C.NO_SPECIES): 0,
        (8, "Kite"): 1,
    }
    assert len(windows) == 1

    removed = {s.name: s.birds for s in log.steps if s.action == "removed" and s.birds}
    assert removed == {
        "Timed outside its survey": 7,
        "Untimed sighting in split period": 9,
        f"Period shorter than {C.fmt_duration(C.MIN_PERIOD_DURATION)}": 6,
    }
    assert out["count"].sum() + sum(removed.values()) == 31  # every bird accounted for


def test_trektellen_mostly_untimed_period_is_not_split():
    surveys = _surveys([("T10", "06:00", "12:00")])
    counts = _counts(
        [("T10", "07:30", "Red Kite", 1), ("T10", "date", "Red Kite", 4)] * 2, surveys
    )
    out, windows = C.trektellen_model_counts(counts, surveys, C.ProcessingLog())
    assert windows.empty
    assert out["count"].sum() == 10
    assert (out["end"] - out["start"]).iloc[0] == pd.Timedelta(hours=6)


def test_count_left_open_overnight_ends_at_dusk():
    # Closed at 03:00 the next day, birds only in the morning: clipped to civil dusk.
    s = pd.DataFrame(
        {
            "survey_id": ["T10"],
            "datetime": [f"{iso(utc(f'{DAY} 06:00'))}/{iso(utc('2023-08-02 03:00'))}"],
            "recording_era": ["trektellen"],
            "survey_coverage": ["complete"],
        }
    )
    surveys = C.parse_surveys(s)
    counts = _counts([("T10", "07:30", "Red Kite", 4), ("T10", "08:30", "Red Kite", 1)], surveys)
    log = C.ProcessingLog()
    counts, surveys = C.clip_to_twilight(counts, surveys, "trektellen", log)
    _, dusk = C.civil_twilight(surveys["date"])
    assert surveys["end"].iloc[0] == dusk.iloc[0]
    local_dusk = dusk.iloc[0].tz_convert(C.TIMEZONE)
    assert (local_dusk.hour, local_dusk.minute) > (21, 0) and local_dusk.hour < 22  # Aug 1
    out, _ = C.trektellen_model_counts(counts, surveys, log)
    assert out["end"].max() == dusk.iloc[0]  # no zero hour after dusk


def test_count_ending_shortly_after_dusk_is_kept():
    _, dusk = C.civil_twilight(pd.Series([pd.Timestamp(DAY)]))
    end = (dusk.iloc[0] + pd.Timedelta(minutes=30)).tz_convert(C.TIMEZONE).strftime("%H:%M")
    surveys = _surveys([("T10", "06:00", end)])
    counts = _counts([("T10", "08:00", "Red Kite", 1)], surveys)
    _, clipped = C.clip_to_twilight(counts, surveys, "trektellen", C.ProcessingLog())
    assert clipped["end"].iloc[0] == surveys["end"].iloc[0]


def test_entry_at_the_survey_end_moves_into_it():
    surveys = _surveys([("T10", "06:00", "09:00")])
    counts = _counts([("T10", "09:00", "Red Kite", 3), ("T10", "07:10", "Red Kite", 1)], surveys)
    counts, _ = C.clip_to_twilight(counts, surveys, "trektellen", C.ProcessingLog())
    assert counts["datetime"].iloc[0] == utc(f"{DAY} 08:59")
    out, _ = C.trektellen_model_counts(counts, surveys, C.ProcessingLog())
    assert out.loc[out["start"] == utc(f"{DAY} 08:00"), "count"].sum() == 3


def test_historical_sums_subgroups_and_zero_fills_empty_surveys():
    surveys = _surveys(
        [("H8", "08:00", "09:00"), ("H9", "09:00", "10:00"), ("H10", "10:00", "11:00")],
        era="spreadsheet",
    )
    counts = _counts(
        [
            ("H8", None, "Red Kite", 3),  # adults
            ("H8", None, "Red Kite", 2),  # juveniles of the same entry: summed
            ("H10", None, "Red Kite", 1),
        ],
        surveys,
    )
    log = C.ProcessingLog()
    out = C.historical_model_counts(counts, surveys, log)
    out = C.zero_fill_empty_surveys(out, surveys, "historical", log)
    by_hour = out.set_index(out["start"].dt.tz_convert(C.TIMEZONE).dt.hour)
    assert by_hour.loc[8, "count"] == 5
    assert by_hour.loc[9, "species"] == C.NO_SPECIES and by_hour.loc[9, "count"] == 0
    assert by_hour.loc[10, "count"] == 1


def test_coverage_none_partial_and_presence_only():
    surveys = _surveys(
        [
            ("T10", "06:00", "07:00"),
            ("T11", "07:00", "08:00", "partial"),
            ("T12", "08:00", "09:00", "none"),
        ]
    )
    counts = _counts(
        [
            ("T10", None, "Red Kite", 5),
            ("T10", None, "Kite", None, "normal", C.PRESENCE_ONLY),
            ("T11", None, "Red Kite", 9),
        ],
        surveys,
    )
    log = C.ProcessingLog()
    counts, kept = C.usable_surveys(counts, surveys, "trektellen", log)
    assert kept["survey_id"].tolist() == ["T10"]
    assert counts["count"].tolist() == [5]
    steps = {s.name: (s.n_rows, s.birds) for s in log.steps}
    assert steps["Survey not counted"][0] == 1
    assert steps["Survey coverage partial"] == (1, 9)
    assert steps["Presence only"][0] == 1


def test_build_keeps_the_main_direction_only_and_checks_pass():
    old = "2015-09-15"
    hs = _surveys([("H8", "08:00", "09:00"), ("H10", "10:00", "11:00")], "notebook", old)
    hc = _counts([("H8", None, "Red Kite", 3), ("H10", None, "Red Kite", 1)], hs, old)
    ts = _surveys([("T10", "06:00", "09:00")])
    tc = _counts(
        [
            ("T10", "06:30", "Red Kite", 5),
            ("T10", "06:30", "Red Kite", 40, "reverse"),
            ("T10", "06:30", "Red Kite", 7, "local"),
            ("T10", "08:30", "Red Kite", 1),
        ],
        ts,
    )
    mc = C.build_model_counts(pd.concat([hs, ts]), pd.concat([hc, tc]))
    assert mc.counts["count"].sum() == 3 + 1 + 5 + 1
    status = {c.name: c.status for c in C.check_model_counts(mc)}
    assert status["Birds accounted for"] == "pass"
    assert status["No overlapping periods"] == "pass"
    assert status["Surveyed hours with no row"] == "pass"
    assert status["One row per species and period"] == "pass"


def test_trektellen_species_ids_take_the_first_of_several():
    taxonomy = pd.DataFrame(
        {
            "english_name": ["Red Kite", "raptor sp.", "Kite"],
            "trektellen_species_id": ["101", "997,1150", None],
        }
    )
    assert C.trektellen_species_ids(taxonomy) == {"Red Kite": 101, "raptor sp.": 997}


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, C.DATASET_DIR, "count.csv")),
    reason="no defile-dataset release in data/count/dataset/",
)
def test_real_release_reconciles():
    """On the copied release: every main-direction bird of `count.csv` is in the output or dropped
    by a named step, per source and year, and no check fails."""
    surveys, counts, _, _ = C.read_dataset(DATA_DIR)
    mc = C.build_model_counts(surveys, counts)
    raw = counts.loc[counts["count_category"] == C.MAIN_CATEGORY, "count"].sum()
    removed = sum(s.birds for s in mc.log.steps if s.action == "removed")
    assert mc.counts["count"].sum() + removed == raw
    failed = [c.name for c in C.check_model_counts(mc) if c.status == "fail"]
    assert not failed
