"""Tests for the Explore export (src/explore/), on small synthetic release tables, and on the real
release when it is present in `data/count/dataset/`: the export must reconcile with `count.csv`,
and with the dataset's own daily totals when its interim build sits next to this repo."""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import counts as C
from src.explore import export as E
from src.explore import profile as P
from src.explore import trend as T

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
    assert E.tier_of(E.TIER_FULL_MIN_YEARS, E.TIER_FULL_MIN_DAYS) == "full"
    assert E.tier_of(E.TIER_FULL_MIN_YEARS - 1, 1000) == "short"
    assert E.tier_of(40, E.TIER_RARE_MAX_DAYS) == "rare"


def test_french_names_by_ebird_code_capitalised_with_overrides():
    taxonomy = pd.DataFrame(
        {
            "taxon_id": ["avibase-a", "avibase-b", "avibase-81B32602"],
            "ebird_code": ["eurspa1", "jaeger", "hoocro4"],
        }
    )
    ebird = pd.DataFrame(
        {"SPECIES_CODE": ["eurspa1", "jaeger"], "COMMON_NAME": ["Épervier d'Europe", "labbe sp."]}
    )
    assert E.french_names(taxonomy, ebird).tolist() == [
        "Épervier d'Europe",
        "Labbe sp.",
        "Corneille mantelée",
    ]


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


def _effort(rows: list[tuple]) -> pd.DataFrame:
    """Rows: (date, {local hour: fraction counted}[, state])."""
    out = []
    for r in rows:
        date, cover, state = (*r, "counted")[:3]
        hourly = np.zeros(E.HOURS)
        for h, f in cover.items():
            hourly[h] = f
        out.append(
            {"date": pd.Timestamp(date), "hourly": hourly, "hours": hourly.sum(), "state": state}
        )
    return pd.DataFrame(out)


def test_profile_samples_are_hour_rates_over_the_day_rate_with_zero_hours():
    effort = _effort([(DAY, {8: 1, 9: 1, 10: 1, 11: 0.25})])
    days = pd.DataFrame(
        {"taxon_id": ["k"], "date": [pd.Timestamp(DAY)], "count": [30.0], "timed": [1.0]}
    )
    hourly = pd.DataFrame(
        {
            "taxon_id": ["k", "k"],
            "date": [pd.Timestamp(DAY)] * 2,
            "hour": [8, 10],
            "count": [10, 20],
        }
    )
    s = P.profile_samples(days, hourly, effort)
    # hour 11 is counted for less than PROFILE_MIN_HOUR_EFFORT: not a sample
    assert dict(zip(s["hour"], s["ratio"])) == pytest.approx({8: 1, 9: 0, 10: 2})
    assert (s["weight"] == 30**P.RATIO_WEIGHT_POWER).all()
    assert P.profile_samples(days.assign(timed=0.5), hourly, effort).empty


def test_coverage_index_and_adjusted_days():
    n = P.PROFILE_DOY[1] - P.PROFILE_DOY[0] + 1
    profile = np.zeros((n, E.HOURS))
    profile[:, [10, 11]] = 0.5
    effort = _effort(
        [
            ("2020-08-01", {10: 1, 11: 1}),  # c = 1
            ("2020-08-02", {10: 1}),  # c = 0.5, no record: zero birds
            ("2020-08-03", {9: 1}),  # c = 0
            ("2020-08-04", {10: 1, 11: 1}, "uncertain"),  # not a counted day
            ("1980-08-01", {10: 0.4}),  # c = 0.2
        ]
    )
    c = P.coverage(effort, profile)
    assert c.tolist() == pytest.approx([1, 0.5, 0, 1, 0.2])
    days = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-08-01", "2020-08-03", "2020-08-04", "1980-08-01"]),
            "count": [40.0, 5.0, 9.0, 10.0],
        }
    )
    a = P.annual_index(days, effort, c).set_index("year")
    assert a.loc[2020, "birds"] == 45 and a.loc[2020, "days"] == 3
    assert a.loc[2020, "index"] == pytest.approx(45 / 1.5)
    assert np.isnan(a.loc[1980, "index"])  # mean c below COVERAGE_MIN
    d = P.adjust_days(days, effort, c).set_index("date")["adjusted"]
    assert d["2020-08-01"] == 40 and np.isnan(d["2020-08-03"]) and np.isnan(d["2020-08-04"])


def test_profile_table_steps_through_the_season():
    n = P.PROFILE_DOY[1] - P.PROFILE_DOY[0] + 1
    t = P.profile_table(np.full((n, E.HOURS), 1 / E.HOURS))
    assert t["doy"][0] == P.PROFILE_DOY[0] and t["doy"][1] - t["doy"][0] == P.PROFILE_EXPORT_STEP
    assert len(t["p"][0]) == E.HOURS


def test_start_year_targets_from_1993_others_when_recorded_systematically():
    every = dict.fromkeys(range(1993, 2026), 100)
    late = dict.fromkeys(range(E.ALL_GROUPS_FROM, 2026), 100)
    assert E.start_year("Falconiformes", "Falconidae", late, 2025) == E.SYSTEMATIC_FROM
    assert E.start_year("Passeriformes", "Corvidae", late, 2025) == E.SYSTEMATIC_FROM
    assert E.start_year("Passeriformes", "Fringillidae", late, 2025) == E.ALL_GROUPS_FROM
    assert E.start_year("Passeriformes", "Alaudidae", every, 2025) == E.SYSTEMATIC_FROM
    assert E.start_year("Charadriiformes", "Scolopacidae", {}, 2025) == E.ALL_GROUPS_FROM
    # a few birds in the years it was not counted are not a series
    trickle = every | dict.fromkeys(range(1993, E.ALL_GROUPS_FROM), 1)
    assert E.start_year("Passeriformes", "Hirundinidae", trickle, 2025) == E.ALL_GROUPS_FROM


def test_combined_series_sum_their_members():
    taxonomy = pd.DataFrame(
        {
            "taxon_id": ["a", "b", "c", "x"],
            "scientific_name": ["Columba palumbus", "Columba oenas", "Columba sp.", "Other"],
            "english_name": ["A", "B", "C", "X"],
            "taxon_rank": ["species", "species", "spuh", "species"],
            "order": ["Columbiformes"] * 4,
            "family": ["Columbidae"] * 4,
        }
    )
    taxonomy = pd.concat(
        [
            taxonomy,
            pd.DataFrame(
                {
                    "taxon_id": [f"h{i}" for i in range(5)],
                    "scientific_name": E.COMBINED["combined-hirundinidae"]["members"],
                    "english_name": list("HIJKL"),
                    "taxon_rank": "species",
                    "order": "Passeriformes",
                    "family": "Hirundinidae",
                }
            ),
        ],
        ignore_index=True,
    )
    day = pd.Timestamp("2020-09-01")
    days = pd.DataFrame(
        {
            "taxon_id": ["a", "c", "x"],
            "date": [day] * 3,
            "count": [10.0, 30.0, 99.0],
            "reverse": [np.nan, 2.0, np.nan],
            "local": [np.nan] * 3,
            "qualifiers": [">", "~", ""],
            "timed": [1.0, 0.0, 1.0],
        }
    )
    hourly = pd.DataFrame(
        {"taxon_id": ["a", "x"], "date": [day] * 2, "hour": [9, 9], "count": [10, 99]}
    )
    t, d, h = E.add_combined(taxonomy, days, hourly)
    row = d[d["taxon_id"] == "combined-columba"].iloc[0]
    assert row["count"] == 40 and row["reverse"] == 2 and np.isnan(row["local"])
    assert row["qualifiers"] == ">~" and row["timed"] == pytest.approx(0.25)
    assert h.loc[h["taxon_id"] == "combined-columba", "count"].tolist() == [10]
    members = t.set_index("taxon_id").loc["combined-columba", "members"]
    assert members == ["a", "b", "c"]
    assert len(d) == len(days) + 1  # no swallow records, no combined swallow days


def test_nothing_adjusted_before_start_year():
    n = P.PROFILE_DOY[1] - P.PROFILE_DOY[0] + 1
    profile = np.full((n, E.HOURS), 1 / E.HOURS)
    effort = _effort(
        [
            ("2000-08-01", {h: 1 for h in range(24)}),
            ("2010-08-01", {12: 1} | {h: 1 for h in range(24)}),
        ]
    )
    days = pd.DataFrame(
        {"date": pd.to_datetime(["2000-08-01", "2010-08-01"]), "count": [5.0, 7.0]}
    )
    c = P.coverage(effort, profile)
    assert P.adjust_days(days, effort, c, 2007)["adjusted"].isna().tolist() == [True, False]
    a = P.annual_index(days, effort, c, 2007).set_index("year")["index"]
    assert np.isnan(a[2000]) and a[2010] == 7


ROOT = Path(__file__).resolve().parents[1]
EXPLORE_USERS = {
    "scripts/build_explore.py",
    "scripts/analyse_explore_effort.py",
    "scripts/benchmark_trend.py",
}


def test_forecast_code_does_not_import_explore():
    """The boundary in `src/explore/__init__.py`: only the explore scripts (and this test) import
    `src.explore`; the forecast reaches its results through files, never imports."""
    offenders = []
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py")]:
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("src/explore/") or rel in EXPLORE_USERS:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(n == "src.explore" or n.startswith("src.explore.") for n in names):
                offenders.append(rel)
    assert not offenders, f"forecast code importing src.explore: {offenders}"


# --- trend ------------------------------------------------------------------


def _synthetic_frame(trend: float, seed: int = 0) -> pd.DataFrame:
    """Window days of 2005-2016 with a log-linear trend, a Gaussian season and half the days half
    counted."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", "2016-12-31")
    dates = dates[E.in_window(pd.Series(dates)).to_numpy()]
    f = pd.DataFrame({"date": dates, "year": dates.year, "doy": dates.dayofyear})
    mu = np.exp(3 + trend * (f["year"] - 2005) - ((f["doy"] - 260) / 20) ** 2 / 2)
    f["c"] = np.where(rng.random(len(f)) < 0.5, 1.0, 0.5)
    f["y"] = rng.negative_binomial(2, 2 / (2 + mu * f["c"])).astype(float)
    return f


def test_model_frame_fills_every_window_day():
    effort = _effort([("2020-08-01", {h: 1 for h in range(24)}), ("2020-08-02", {})])
    effort.loc[1, "state"] = "not_counted"
    days = pd.DataFrame(
        {"date": pd.to_datetime(["2020-08-01", "2020-07-01"]), "count": [5.0, 99.0]}
    )
    c = pd.Series([0.8, 0.0], index=effort.index)
    f = T.model_frame(days, effort, c, 2020, 2020).set_index("date")
    assert len(f) == E.in_window(pd.Series(pd.date_range("2020-01-01", "2020-12-31"))).sum()
    assert f.loc["2020-08-01", "c"] == 0.8 and f.loc["2020-08-01", "y"] == 5
    assert f.loc["2020-08-02", "c"] == 0 and f.loc["2020-08-02", "y"] == 0
    assert f["y"].sum() == 5  # outside the window: not in the frame


def test_fill_keeps_the_counted_and_draws_the_rest():
    f = _synthetic_frame(0.0)
    m = T.fit("season", f, np.arange(2005, 2017), (int(f["doy"].min()), int(f["doy"].max())))
    full = f.assign(c=1.0)
    assert (T.fill_draws(m, full, 1.0, n=20) == full["y"].to_numpy()).all()
    draws = T.fill_draws(m, f, 1.0, n=200)
    assert (draws >= f["y"].to_numpy()).all()
    assert (draws[:, (f["c"] < 1).to_numpy()] > f["y"].to_numpy()[(f["c"] < 1).to_numpy()]).any()


@pytest.mark.parametrize("variant", ["gam", "gp"])
def test_trend_recovers_a_doubling(variant):
    f = _synthetic_frame(np.log(2) / 11, seed=1)  # x2 over 2005-2016
    m = T.fit(variant, f, np.arange(2005, 2017), (int(f["doy"].min()), int(f["doy"].max())))
    a = T.annual_totals(m, f, n=200).set_index("year")
    assert a.loc[2016, "smooth"] / a.loc[2005, "smooth"] == pytest.approx(2, rel=0.3)
    assert ((a["q2.5"] <= a["total"]) & (a["total"] <= a["q97.5"])).all()
