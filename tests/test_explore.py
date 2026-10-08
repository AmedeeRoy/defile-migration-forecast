"""Tests for the Explore export (src/data/explore.py), on small synthetic release tables, and on
the real release when it is present in `data/count/dataset/`: the export must reconcile with
`count.csv`, and with the dataset's own daily totals when its interim build sits next to this
repo."""

import os

import numpy as np
import pandas as pd
import pytest

from src.data import counts as C
from src.data import explore as E

DAY = "2023-08-01"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_DAILY = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "defile-dataset",
    "interim",
    "derived",
    "daily_counts.csv",
)


def utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local).tz_localize(C.TIMEZONE).tz_convert("UTC")


def iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _surveys(rows: list[tuple]) -> pd.DataFrame:
    """Rows: (survey id, local start, local end[, coverage[, era]]), as 'YYYY-MM-DD HH:MM'."""
    out = []
    for r in rows:
        sid, a, b, coverage, era = (*r, "complete", "trektellen")[:5]
        out.append(
            {
                "survey_id": sid,
                "datetime": f"{iso(utc(a))}/{iso(utc(b))}",
                "recording_era": era,
                "survey_coverage": coverage,
            }
        )
    return C.parse_surveys(pd.DataFrame(out))


def _counts(rows: list[tuple], surveys: pd.DataFrame) -> pd.DataFrame:
    """Rows: (survey id, own timing, taxon, count[, category, estimation]).

    Own timing: a local 'YYYY-MM-DD HH:MM', a 'YYYY-MM-DD' (day level), or None (inherits).
    """
    out = []
    for i, r in enumerate(rows):
        sid, t, taxon, count, category, estimation = (*r, "normal", None)[:6]
        own = None if t is None else t if len(t) == 10 else iso(utc(t))
        out.append(
            {
                "count_id": f"c{i}",
                "source_count_id": f"c{i}",
                "survey_id": sid,
                "taxon_id": f"avibase-{taxon}",
                "datetime": own,
                "count": count,
                "count_category": category,
                "count_estimation": estimation,
            }
        )
    count = pd.DataFrame(out)
    taxa = sorted({r[2] for r in rows})
    taxonomy = pd.DataFrame({"taxon_id": [f"avibase-{t}" for t in taxa], "english_name": taxa})
    return E.release_counts(count, surveys, taxonomy)


def test_effort_is_the_union_of_complete_intervals_by_local_hour():
    s = _surveys(
        [
            ("a", f"{DAY} 08:30", f"{DAY} 10:00"),
            ("b", f"{DAY} 10:00", f"{DAY} 10:45"),
            ("c", f"{DAY} 12:00", f"{DAY} 13:00", "partial"),
            ("d", "2023-08-02 08:00", "2023-08-02 12:00", "none"),
            ("e", "2023-08-03 09:00", "2023-08-03 10:00", "unknown"),
        ]
    )
    e = E.build_effort(s).set_index(E.build_effort(s)["date"].dt.strftime("%Y-%m-%d"))
    d = e.loc[DAY]
    assert d["hours"] == pytest.approx(2.25)
    assert (d["first"].strftime("%H:%M"), d["last"].strftime("%H:%M")) == ("08:30", "10:45")
    assert list(d["hourly"][8:12]) == pytest.approx([0.5, 1, 0.75, 0])
    assert d["partial_hours"] == 1 and d["periods"] == 2
    assert e["state"].to_dict() == {
        DAY: "counted",
        "2023-08-02": "not_counted",
        "2023-08-03": "uncertain",
    }


def test_effort_splits_a_survey_at_local_midnight():
    s = _surveys([("a", f"{DAY} 22:00", "2023-08-02 02:00")])
    e = E.build_effort(s)
    assert list(e["hours"]) == [2, 2]
    assert e["hourly"].iloc[1][:2].tolist() == [1, 1]


def test_timed_to_the_hour_own_time_or_a_one_hour_survey_only():
    s = _surveys(
        [
            ("long", f"{DAY} 08:00", f"{DAY} 12:00"),
            ("hour", f"{DAY} 13:00", f"{DAY} 14:00", "complete", "spreadsheet"),
            ("straddle", f"{DAY} 14:30", f"{DAY} 15:30", "complete", "spreadsheet"),
        ]
    )
    c = _counts(
        [
            ("long", f"{DAY} 09:40", "kite", 5),  # own time
            ("long", None, "kite", 7),  # inherits a 4 h survey
            ("long", DAY, "kite", 11),  # day level
            ("hour", None, "kite", 13),  # inherits a clock hour
            ("straddle", None, "kite", 17),  # inherits an hour across two clock hours
        ],
        s,
    )
    assert c["hour"].iloc[[0, 3]].tolist() == [9, 13]
    assert c["hour"].iloc[[1, 2, 4]].isna().all()
    d = E.daily_counts(c)
    assert d["timed"].iloc[0] == pytest.approx((5 + 13) / 53)
    h = E.hourly_counts(c)
    assert dict(zip(h["hour"], h["count"])) == {9: 5, 13: 13}


def test_daily_counts_keep_unknown_apart_from_zero_and_keep_qualifiers():
    s = _surveys([("a", f"{DAY} 08:00", f"{DAY} 12:00")])
    c = _counts(
        [
            ("a", None, "kite", 10, "normal", "~"),
            ("a", None, "kite", 20, "normal", ">"),
            ("a", None, "kite", 0, "reverse"),
            ("a", None, "dove", None, "normal", "x"),
        ],
        s,
    )
    d = E.daily_counts(c).set_index("taxon_id")
    kite, dove = d.loc["avibase-kite"], d.loc["avibase-dove"]
    assert kite["count"] == 30 and kite["qualifiers"] == ">~"
    assert kite["reverse"] == 0 and np.isnan(kite["local"])  # explicit zero vs no row
    assert np.isnan(dove["count"]) and dove["qualifiers"] == "x"
    t = E.build_taxa(
        pd.DataFrame({"taxon_id": ["avibase-kite", "avibase-dove", "avibase-none"]}).assign(
            english_name="", scientific_name="", taxon_rank="", order="", family=""
        ),
        E.daily_counts(c),
    ).set_index("taxon_id")
    assert t.loc["avibase-dove", "days"] == 1  # presence-only is occurrence
    assert t.loc["avibase-none", "tier"] == "rare"


def test_tiers():
    assert E.tier_of(E.TIER_FULL_MIN_YEARS, E.TIER_FULL_MIN_BIRDS, 50) == "full"
    assert E.tier_of(E.TIER_FULL_MIN_YEARS - 1, 1e6, 50) == "short"
    assert E.tier_of(40, 1e6, E.TIER_RARE_MAX_DAYS) == "rare"


def test_annual_window_and_top_day():
    days = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-07-17", "2020-08-01", "2020-11-18", "2021-09-01"]),
            "count": [100.0, 5.0, 7.0, np.nan],
        }
    )
    a = E.annual(days).set_index("year")
    assert a.loc[2020, "total"] == 112 and a.loc[2020, "window"] == 12
    assert a.loc[2020, "max_date"] == pd.Timestamp("2020-07-17")
    assert a.loc[2021, "total"] == 0 and a.loc[2021, "days"] == 1


def test_partial_year_is_a_season_in_progress():
    days = pd.DataFrame({"date": pd.to_datetime(["2025-11-30", "2026-10-01"])})
    assert E.partial_years(days) == [2026]
    assert E.partial_years(days.iloc[:1]) == []


def test_json_values():
    assert E.dumps({"d": pd.Timestamp("2020-01-02"), "t": pd.Timestamp("2020-01-02 08:30")}) == (
        '{"d":"2020-01-02","t":"08:30"}'
    )
    assert E.columns(pd.DataFrame({"x": [1.0, np.nan, 0.12345]})) == {"x": [1, None, 0.123]}


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, C.DATASET_DIR, C.REPORT_FILE)),
    reason="no release copied into data/count/dataset/",
)
def test_real_release_reconciles():
    surveys, counts, *_ = E.read_release(DATA_DIR)
    d = E.daily_counts(counts)
    raw = pd.read_csv(os.path.join(DATA_DIR, C.DATASET_DIR, "count.csv"), low_memory=False)
    expected = raw.groupby("count_category")["count"].sum()
    assert d["count"].sum() == expected["normal"]
    assert d["reverse"].sum() == expected.get("reverse", 0)
    assert d["local"].sum() == expected.get("local", 0)
    normal = counts["count_category"] == C.MAIN_CATEGORY
    assert E.hourly_counts(counts)["count"].sum() == counts.loc[normal, "timed_count"].sum()
    if os.path.exists(DATASET_DAILY):
        ds = pd.read_csv(DATASET_DAILY, parse_dates=["date"])
        m = ds.merge(d, on=["date", "taxon_id"], how="outer", indicator=True)
        assert (m["_merge"] == "both").all()
        for a, b in [
            ("count_x", "count_y"),
            ("count_reverse", "reverse"),
            ("count_local", "local"),
        ]:
            assert np.allclose(m[a].fillna(-1), m[b].fillna(-1)), a
