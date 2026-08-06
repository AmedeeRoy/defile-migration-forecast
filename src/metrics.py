"""Evaluation metrics for the Défilé forecast.

Implements the four-level metric set described in `DEVELOPMENT.md` Phase 1. The guiding
constraints there, restated because they drive every design choice in this module:

- **Never pool species or eras.** Merlin is 95% zero survey rows, Common Buzzard 65%; a
  single pooled number hides which problem the model actually failed at. Every metric is
  reported overall *and* broken down by era.
- **Always report a skill score against a naive baseline.** A raw MAE of 3 birds/hr means
  nothing on its own. Two baselines are computed for every level: day-of-year
  **phenology** (from `data/count/species_doy_statistics.json`) and **persistence**
  (yesterday's observed rate). If the model does not beat phenology, the weather
  features are contributing nothing, and no raw metric value alone would show that.
- **One headline metric per level**, with the supporting diagnostics computed but not
  headlined, so a run can be judged at a glance and still be debugged when the headline
  looks wrong.

Everything here is evaluation-only: a second scoring pass over predictions that already
exist. Nothing in this module feeds training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.phenology import Phenology

# --------------------------------------------------------------------------------------
# Eras
# --------------------------------------------------------------------------------------

# Boundaries match `year_period` in src/data/defile_datamodule.py, which is also what the
# "period" train/val/test split balances across -- so an era here is exactly one of the
# strata the split was built from. See data/count/readme.md for the protocol history that
# motivates the cuts: sporadic pigeon-focused coverage before 1993, daily volunteer
# monitoring from 1993, and the shift to hourly recording forms from 2014.
ERA_EDGES: Tuple[int, int] = (1993, 2014)
ERA_LABELS: Tuple[str, str, str] = ("pre-1993", "1993-2013", "2014+")


def era_of(year: "int | np.ndarray") -> "str | np.ndarray":
    """Map a year (or array of years) to its era label."""
    lo, hi = ERA_EDGES
    return np.where(year < lo, ERA_LABELS[0], np.where(year < hi, ERA_LABELS[1], ERA_LABELS[2]))


# --------------------------------------------------------------------------------------
# Small metric primitives
# --------------------------------------------------------------------------------------


def _skill(score: float, baseline: float, lower_is_better: bool = True) -> float:
    """Skill score of `score` against `baseline`, on the conventional `1 - s/s_ref` form.

    1 means perfect, 0 means "no better than the baseline", negative means worse. For
    higher-is-better scores (CSI) the complement is used instead, so the sign convention
    is the same at every level of the table.
    """
    if not np.isfinite(score) or not np.isfinite(baseline):
        return np.nan
    if lower_is_better:
        return np.nan if baseline == 0 else 1.0 - score / baseline
    return np.nan if baseline == 1 else (score - baseline) / (1.0 - baseline)


def _mae(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs))) if len(obs) else np.nan


def _bias(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(pred - obs)) if len(obs) else np.nan


def contingency(obs_event: np.ndarray, pred_event: np.ndarray) -> Dict[str, float]:
    """Full 2x2 contingency table plus CSI, POD and FAR.

    CSI (critical success index, `H / (H + M + F)`) is the headline: it ignores correct
    rejections, which dominate for a species whose survey rows are 61-95% zero and would
    make accuracy-style scores look excellent for a model that never predicts an event.
    The raw counts are kept because CSI alone cannot distinguish "misses everything" from
    "cries wolf constantly", and that is exactly what you need to know when it looks wrong.
    """
    hits = float(np.sum(obs_event & pred_event))
    misses = float(np.sum(obs_event & ~pred_event))
    false_alarms = float(np.sum(~obs_event & pred_event))
    correct_neg = float(np.sum(~obs_event & ~pred_event))

    denom = hits + misses + false_alarms
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "csi": hits / denom if denom else np.nan,
        "pod": hits / (hits + misses) if (hits + misses) else np.nan,
        "far": false_alarms / (hits + false_alarms) if (hits + false_alarms) else np.nan,
    }


def earth_movers_distance(obs_profile: np.ndarray, pred_profile: np.ndarray) -> float:
    """1-D Wasserstein distance between two 24-hour profiles, in hours.

    Both profiles are normalised to sum to 1 first, so this measures *shape* disagreement
    only, independent of the day's magnitude. On a regular unit grid the Wasserstein
    distance reduces to the L1 distance between the two CDFs, so no optimisation is needed.
    """
    o, p = np.asarray(obs_profile, float), np.asarray(pred_profile, float)
    if o.sum() <= 0 or p.sum() <= 0:
        return np.nan
    return float(np.sum(np.abs(np.cumsum(o / o.sum()) - np.cumsum(p / p.sum()))))


def passage_dates(doy: np.ndarray, rate: np.ndarray, levels=(10, 50, 90)) -> Dict[int, float]:
    """Day-of-year at which each cumulative-passage level is reached.

    Interpolated between days rather than snapped to one, so a half-day phenology shift is
    visible instead of being quantised away.
    """
    order = np.argsort(doy)
    d, r = np.asarray(doy, float)[order], np.asarray(rate, float)[order]
    total = r.sum()
    if total <= 0 or len(d) < 2:
        return {lvl: np.nan for lvl in levels}
    cdf = np.cumsum(r) / total
    return {lvl: float(np.interp(lvl / 100.0, cdf, d)) for lvl in levels}


# --------------------------------------------------------------------------------------
# Evaluation frame
# --------------------------------------------------------------------------------------


def build_frame(
    count: pd.DataFrame,
    mask: np.ndarray,
    pred_hourly: np.ndarray,
    phenology: Optional[Phenology] = None,
) -> pd.DataFrame:
    """Assemble the tidy per-survey-row frame every metric below is computed from.

    :param count: The split's count rows, in dataloader order (`DefileDataset.count`).
    :param mask: `(24, n_rows)` fraction of each hour covered by each survey row, in the
        same order (`DefileDataset.mask`).
    :param pred_hourly: `(n_rows, 24)` predicted log1p(birds/hr) per hour.
    :param phenology: Optional baseline; adds a `phen` column.
    :return: One row per survey period, with observed and predicted rates in birds/hr.
    """
    mask = np.asarray(mask, dtype=float)
    if mask.shape[0] != 24:
        raise ValueError(f"mask must be (24, n_rows), got {mask.shape}")
    coverage = mask.sum(axis=0)  # hours covered by each survey

    pred_hourly = np.asarray(pred_hourly, dtype=float)
    if pred_hourly.shape != (len(count), 24):
        raise ValueError(
            f"pred_hourly must be (n_rows, 24) = {(len(count), 24)}, got {pred_hourly.shape}"
        )

    # Same aggregation the loss uses (src.models.criterion.applyMask): back to count space
    # with expm1 *before* averaging over the covered hours, never after. Averaging in log
    # space would target a systematically smaller value on peaked days and would not be
    # the quantity the model was trained to get right.
    with np.errstate(invalid="ignore", divide="ignore"):
        pred_rate = np.sum(np.expm1(pred_hourly) * mask.T, axis=1) / coverage

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(count["date"].values),
            "obs": count["count"].to_numpy(dtype=float),  # birds/hr over the survey
            "pred": pred_rate,
            "coverage": coverage,
            "start_hour": np.argmax(mask > 0, axis=0).astype(float),
        }
    )
    frame["doy"] = frame["date"].dt.dayofyear
    frame["year"] = frame["date"].dt.year
    frame["era"] = era_of(frame["year"].to_numpy())

    if phenology is not None:
        frame["phen"] = phenology.daily_rate(frame["doy"].to_numpy())

    return frame


def daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse survey rows to one row per date, coverage-weighted.

    A date can carry anything from one 9-hour survey (pre-2014) to fifteen 1-hour rows
    (2022 on). Weighting by covered hours makes the daily rate mean the same thing in both
    eras -- an unweighted mean would let a single unrepresentative 20-minute row outvote a
    full day of watching.
    """
    w = frame["coverage"].to_numpy()
    out = (
        frame.assign(_ow=frame["obs"] * w, _pw=frame["pred"] * w, _w=w)
        .groupby("date", as_index=False)
        .agg(
            obs=("_ow", "sum"),
            pred=("_pw", "sum"),
            coverage=("_w", "sum"),
            n_rows=("obs", "size"),
            doy=("doy", "first"),
            year=("year", "first"),
            era=("era", "first"),
            **({"phen": ("phen", "first")} if "phen" in frame else {}),
        )
    )
    out["obs"] /= out["coverage"]
    out["pred"] /= out["coverage"]

    # Persistence baseline: yesterday's observed daily rate. NaN when the previous
    # calendar day is absent from this split (season edges, unwatched days), which drops
    # that row from the persistence comparison only -- never from the model's own score.
    out = out.sort_values("date").reset_index(drop=True)
    prev = out.set_index("date")["obs"]
    out["persistence"] = prev.reindex(out["date"] - pd.Timedelta(days=1)).to_numpy()
    return out


def hourly_profiles(
    frame: pd.DataFrame,
    mask: np.ndarray,
    pred_hourly: np.ndarray,
    min_rows: int = 6,
    max_span: float = 1.5,
) -> pd.DataFrame:
    """Reconstruct true hour-by-hour profiles for the dates that actually have them.

    A date recorded as several ~1-hour survey rows *is* an hour-by-hour count. Those dates
    (common since 2014, near-universal since 2021) are the only ones where intra-day shape
    can be scored at all; longer single-block surveys carry no shape information and are
    excluded rather than compared against a flat line.

    :param min_rows: Minimum number of survey rows on a date for it to qualify.
    :param max_span: Maximum hours covered by any one of those rows.
    :return: One row per qualifying date, carrying the 24-element observed and predicted
        profiles and the hours that were actually covered.
    """
    mask = np.asarray(mask, dtype=float)
    pred_count = np.expm1(np.asarray(pred_hourly, dtype=float))  # (n, 24) birds/hr

    records: List[dict] = []
    for date, idx in frame.groupby("date").indices.items():
        idx = np.asarray(idx)
        if len(idx) < min_rows or frame["coverage"].to_numpy()[idx].max() > max_span:
            continue

        m = mask[:, idx]  # (24, k)
        weight = m.sum(axis=1)  # hours covered across the day
        covered = weight > 0
        if covered.sum() < min_rows:
            continue

        obs_profile = np.zeros(24)
        obs_profile[covered] = (m[covered] @ frame["obs"].to_numpy()[idx]) / weight[covered]

        records.append(
            {
                "date": date,
                "doy": frame["doy"].to_numpy()[idx][0],
                "year": frame["year"].to_numpy()[idx][0],
                "era": frame["era"].to_numpy()[idx][0],
                "covered": covered,
                "obs_profile": obs_profile,
                # All rows of a date share the same prediction (one forward pass per
                # date's weather), so the first row's profile is the day's profile.
                "pred_profile": pred_count[idx[0]],
            }
        )

    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# The four levels
# --------------------------------------------------------------------------------------


def row_level(frame: pd.DataFrame, daily: pd.DataFrame) -> Dict[str, float]:
    """Level 1 -- MAE and Bias per survey row, birds/hr, with both skill scores."""
    obs, pred = frame["obs"].to_numpy(), frame["pred"].to_numpy()
    out = {"n_rows": float(len(frame)), "mae": _mae(obs, pred), "bias": _bias(obs, pred)}

    if "phen" in frame:
        out["mae_phen"] = _mae(obs, frame["phen"].to_numpy())
        out["mae_skill_phen"] = _skill(out["mae"], out["mae_phen"])

    # Persistence lives on the daily frame (it is yesterday's *daily* rate); scoring it on
    # the same daily aggregate keeps the comparison like-for-like.
    ok = daily["persistence"].notna().to_numpy()
    if ok.any():
        d_obs = daily["obs"].to_numpy()[ok]
        out["mae_persistence"] = _mae(d_obs, daily["persistence"].to_numpy()[ok])
        out["mae_daily"] = _mae(d_obs, daily["pred"].to_numpy()[ok])
        out["mae_skill_persistence"] = _skill(out["mae_daily"], out["mae_persistence"])

    return out


def event_level(daily: pd.DataFrame, phenology: Phenology, level: float = 90) -> Dict[str, float]:
    """Level 2 -- CSI against a per-doy phenological p90 threshold, plus the full table."""
    if "phen" not in daily or daily.empty:
        return {}

    threshold = phenology.quantile(daily["doy"].to_numpy(), level)
    # A p90 threshold that is exactly zero (deep off-season, where 90% of days saw
    # nothing) would score every non-zero prediction as an event; require a real count.
    valid = threshold > 0
    if not valid.any():
        return {}

    thr = threshold[valid]
    obs_event = daily["obs"].to_numpy()[valid] >= thr
    out = {
        f"event_{k}": v
        for k, v in contingency(obs_event, daily["pred"].to_numpy()[valid] >= thr).items()
    }
    out["event_n_days"] = float(valid.sum())
    out["event_base_rate"] = float(obs_event.mean())

    # Phenology can never exceed its own p90 by construction, so it predicts no events
    # and scores CSI 0 -- a degenerate baseline. Persistence is the informative one here.
    persistence = daily["persistence"].to_numpy()[valid]
    ok = np.isfinite(persistence)
    if ok.any():
        base = contingency(obs_event[ok], persistence[ok] >= thr[ok])
        out["event_csi_persistence"] = base["csi"]
        out["event_csi_skill_persistence"] = _skill(
            out["event_csi"], base["csi"], lower_is_better=False
        )

    return out


def shape_level(profiles: pd.DataFrame) -> Dict[str, float]:
    """Level 3 -- peak-hour error in hours (headline) and EMD (computed, not headlined)."""
    if profiles.empty:
        return {}

    peak_err, emd = [], []
    for _, row in profiles.iterrows():
        covered = row["covered"]
        obs, pred = row["obs_profile"][covered], row["pred_profile"][covered]
        if obs.sum() <= 0:
            continue  # a day with no birds at all has no peak to get right
        hours = np.flatnonzero(covered)
        peak_err.append(abs(hours[int(np.argmax(pred))] - hours[int(np.argmax(obs))]))
        emd.append(earth_movers_distance(obs, pred))

    if not peak_err:
        return {}
    return {
        "shape_n_days": float(len(peak_err)),
        "shape_peak_hour_mae": float(np.mean(peak_err)),
        "shape_peak_hour_within_1h": float(np.mean(np.asarray(peak_err) <= 1)),
        "shape_emd": float(np.nanmean(emd)),
    }


def season_level(daily: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Level 4 -- median-passage-date error (days) and seasonal total ratio, per year.

    Returned per year as well as aggregated: with only ~3 test years under the current
    random-period split, a single averaged phenology number is not something to lean on.
    Making this trustworthy needs leave-one-year-out or rolling-origin CV (DEVELOPMENT.md
    Phase 1), which is a change to the eval harness, not to this function.
    """
    rows = []
    for year, g in daily.groupby("year"):
        obs_dates = passage_dates(g["doy"].to_numpy(), g["obs"].to_numpy())
        pred_dates = passage_dates(g["doy"].to_numpy(), g["pred"].to_numpy())
        obs_total, pred_total = g["obs"].sum(), g["pred"].sum()
        rows.append(
            {
                "year": int(year),
                "era": g["era"].iloc[0],
                "n_days": len(g),
                "obs_median_doy": obs_dates[50],
                "pred_median_doy": pred_dates[50],
                "median_error_days": pred_dates[50] - obs_dates[50],
                "obs_p10_doy": obs_dates[10],
                "pred_p10_doy": pred_dates[10],
                "obs_p90_doy": obs_dates[90],
                "pred_p90_doy": pred_dates[90],
                "total_ratio": pred_total / obs_total if obs_total > 0 else np.nan,
            }
        )

    per_year = pd.DataFrame(rows)
    if per_year.empty:
        return {}, per_year

    return {
        "season_n_years": float(len(per_year)),
        "season_median_date_mae": float(np.nanmean(np.abs(per_year["median_error_days"]))),
        "season_total_ratio": float(np.nanmean(per_year["total_ratio"])),
    }, per_year


# --------------------------------------------------------------------------------------
# Top-level report
# --------------------------------------------------------------------------------------


@dataclass
class MetricReport:
    """Everything computed for one species in one run.

    `scalars` is the flat form for Lightning/CSV logging; the frames are what the PDF
    report renders and what a Phase 2 experiment comparison would read back.
    """

    species: str
    scalars: Dict[str, float]
    by_era: pd.DataFrame
    per_year: pd.DataFrame
    frame: pd.DataFrame
    daily: pd.DataFrame
    profiles: pd.DataFrame

    def logged(self, prefix: str = "test") -> Dict[str, float]:
        """The subset worth pushing to the logger, namespaced under `prefix`."""
        return {f"{prefix}/{k}": float(v) for k, v in self.scalars.items() if np.isfinite(v)}


def _levels(
    frame: pd.DataFrame, daily: pd.DataFrame, profiles: pd.DataFrame, phen: Phenology
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(row_level(frame, daily))
    out.update(event_level(daily, phen))
    out.update(shape_level(profiles))
    season, _ = season_level(daily)
    out.update(season)
    return out


def evaluate(
    count: pd.DataFrame,
    mask: np.ndarray,
    pred_hourly: np.ndarray,
    species: str,
    data_dir: str,
) -> MetricReport:
    """Score one species' test predictions at all four levels, overall and per era."""
    phen = Phenology.load(data_dir, species)
    frame = build_frame(count, mask, pred_hourly, phen)
    daily = daily_frame(frame)
    profiles = hourly_profiles(frame, mask, pred_hourly)

    scalars = _levels(frame, daily, profiles, phen)
    _, per_year = season_level(daily)

    era_rows = []
    for era in ERA_LABELS:
        f = frame[frame["era"] == era]
        if f.empty:
            continue
        d = daily[daily["era"] == era]
        p = profiles[profiles["era"] == era] if not profiles.empty else profiles
        era_rows.append({"era": era, **_levels(f, d, p, phen)})

    return MetricReport(
        species=species,
        scalars=scalars,
        by_era=pd.DataFrame(era_rows),
        per_year=per_year,
        frame=frame,
        daily=daily,
        profiles=profiles,
    )


def validation_skill(
    count: pd.DataFrame, mask: np.ndarray, pred_hourly: np.ndarray, phenology: Phenology
) -> float:
    """Row-level MAE skill against phenology, cheap enough to log every val epoch.

    This is the one number worth watching during training: it answers "are the weather
    features earning their place *yet*", which `val/loss` alone cannot.
    """
    frame = build_frame(count, mask, pred_hourly, phenology)
    return _skill(
        _mae(frame["obs"].to_numpy(), frame["pred"].to_numpy()),
        _mae(frame["obs"].to_numpy(), frame["phen"].to_numpy()),
    )
