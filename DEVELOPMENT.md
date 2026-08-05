# Défilé migration forecast — development roadmap

This tracks what's actually left to do. Resolved items are removed once merged and
retrained against, not archived here — git history and the PR list are the record of what
was fixed and why. If something below looks wrong or already done, say so before it gets
deleted rather than after.

## What we are building

An autonomous system that, every morning without human intervention, publishes a forecast
of the **hourly passage rate of migrating raptors at Défilé de l'Écluse**, per species, for
today and the next few days. The forecast is consumed by the
[defileViz](https://github.com/Rafnuss/defileViz) web app.

Three things worth keeping in view when touching model architecture: the unit of
prediction is an **hourly rate**, not a daily total, so the shape of the day matters as
much as the daily sum. The system must run **unattended** — a silently wrong or stale
forecast is as serious as a crash, and neither is currently detected. And the features
available at 03:00 UTC on the day of the run are the only features that exist; anything
the model learns to depend on that isn't available then is a liability.

## Status

Phases 0 and 1 are merged: `centralize-weather` (one `get_weather()` for training and
forecasting), the season guard (4.11), and the uncertainty-channel drop (4.8). 4.9
(normalisation leakage) was fixed and then reverted — Raphaël's call, fitting on the full
dataset isn't a real problem for this use case, so it's closed as won't-fix rather than
carried as an open item. None of the `prod/models/` checkpoints have been retrained against
any of this yet (Phase 2, below).

## Open defects

**4.19d The model trains on 25 km wind and is served ~2 km wind, and at Défilé those are
substantially different fields.** Root cause now confirmed, not just suspected: the
forecast path (`source="forecast"` in `src/data/weather.py`) doesn't set `models=`, so
Open-Meteo's `best_match` picks a model per location. At Défilé that resolves to **DWD
ICON-D2 at 2 km** — confirmed by comparing the returned grid coordinates (`lat=46.12,
lon=5.92`) against an explicit `models=dwd_icon_d2` request, which snaps to the identical
point. Training uses ERA5 pinned to `models=era5`, a 0.25° (~25-28 km) grid. So the
mismatch is a ~12x difference in grid spacing between the two paths, not a subtlety.

Measured over 61 days at Défilé, switching the forecast path from the default (ICON-D2) to
`ecmwf_ifs025` — a real, working parameter value, confirmed against the live API, that
resolves to the same 0.25° grid as `era5` (`lat=46.0, lon=6.0`):

| vs ERA5 archive | default (ICON-D2, 2 km) | `ecmwf_ifs025` (0.25°) |
|---|---|---|
| `u_component_of_wind_10m` corr | 0.27 | **0.55** |
| `v_component_of_wind_10m` corr | 0.66 | **0.80** |
| `u_component_of_wind_100m` corr | 0.50 | **0.65** |
| `v_component_of_wind_100m` corr | 0.69 | **0.83** |
| `u_wind_10m` std ratio (fcst/ERA5) | 1.73 | **0.95** |
| `temperature_2m` corr / std ratio | 0.97 / 0.94 | 0.97 / **1.01** |

Pinning to `ecmwf_ifs025` roughly doubles wind agreement and brings the variance ratio to
near 1. It doesn't reach 1.0 corr — ERA5 is a reanalysis that assimilates observations,
IFS-025 is a raw forecast, so some residual disagreement is expected even at matching
resolution — but the improvement is large and real, not marginal.

There's also a domain reason to prefer the coarser product on its own merits, independent
of train/serve consistency: birds crossing Défilé integrate wind over a section of the
gorge at least 2-5 km wide. ICON-D2's 2 km detail is finer than what's biologically
relevant to a bird making a crossing decision over that width; a 25 km field is arguably
closer to the right physical scale for the phenomenon, not just a compromise for
consistency's sake.

**Historical Forecast API, checked and corrected:** covers `2016-01-01` through a rolling
window to roughly two weeks ahead of today (`2026-08-20` as of this check), not "since
2024" as assumed going in. It carries the same `models=` catalog, including
`ecmwf_ifs025`. About 10 years of archived past-forecast data at matching lead times — not
long enough to replace the 60-year ERA5 training history, but plenty for the lead-day
skill measurement in Phase 3, and for a fine-tune experiment at matching lead times if that
turns out to be worth doing after the simpler `ecmwf_ifs025` pin is measured in production.

**Recommended next step, not yet implemented:** pin the forecast path to `models=ecmwf_ifs025`
in `src/data/weather.py`. Cheap (one parameter), well-supported by the numbers above, and
the domain argument suggests it's not merely a stopgap. `tests/test_weather.py::test_wind_over_complex_terrain_is_documented_as_divergent`
records the current (default-model) numbers so the effect can't be quietly forgotten;
its tolerances would need loosening once the default model changes.

## Plan

**Phase 0 — done.** `centralize-weather` merged: one `get_weather()` entry point
(`src/data/weather.py`) for both training and forecasting, replacing the GEE CSV export
and the independent Open-Meteo forecast client that used to drift apart from it.

**Phase 1 — done.** `fix/uncertainty-channel` (4.8) and `feat/season-guard` (4.11) merged.
`fix/normalization-leakage` (4.9) was merged, then reverted — see Status above.

**Phase 2 — retrain all 11 species checkpoints**, next. This is the actual gate for
trusting any model-quality number; nothing before this point should be judged on accuracy.
Worth doing once, after the `ecmwf_ifs025` pin below lands too, rather than retraining
twice.

**Phase 3 — two research spikes**, branch per experiment, not urgent to land quickly:
- **Year selection.** `data/count/readme.md` documents real protocol changes: sporadic
  pigeon-focused coverage before 1993, daily volunteer monitoring from 1993, a salaried
  observer 2008–2016, two salaried observers from 2017. Recording granularity changed too
  (daily totals → hourly forms → Naturalist → Trektellen), which already reweights the loss
  toward the hourly-resolution era by accident. Run a **year-subset ladder**
  (1966+/1993+/2008+/2014+/2017+) on a **chronological** holdout (not the current
  random-years split, which flatters the model by letting it interpolate across eras) —
  cheap, and the single most informative experiment available. `year_used: "period"` is
  available and untested; include it in the sweep. Bigger structural idea worth a follow-up:
  predict the *share* of the season's total per day/hour and forecast the annual total
  separately, removing most year-to-year variance from the hard part of the problem.
- **Wind resolution** (4.19d, above) — pin `models=ecmwf_ifs025` on the forecast path, then
  measure the effect on forecast skill (not just feature correlation) before deciding
  whether the lead-day fine-tune on the Historical Forecast API is worth the extra work.

**Phase 4 — Trektellen counts as model input.** Plumbing already exists: a working proxy at
`https://defile.raphaelnussbaumer.com/trektellen/{siteId}/{yyyymmdd}` and a 47 MB NW-Europe
export in `data/Trektellen_raptor_2015_2024/`. Three things matter more than the
implementation: missing days must be represented with an observed-flag channel, not
zero-filled (zero means "watched and saw nothing" — filling absence with it teaches exactly
backwards, worst on bad-weather days when observers stay home); a lagged count's usefulness
decays with lead time, so either train per-lead heads or accept near-term-only value;
and the biggest upside is upstream sites (spatial early warning of a wave in transit), not
autoregression on Défilé itself, so scope the first experiment as own-site lag first
(`feat/trektellen-defile-lag`), upstream sites as the follow-up
(`feat/trektellen-upstream-sites`) once that shows value. A third-party API on the forecast
path needs a defined fallback (observed-flag channel, carry on) rather than an exception
that kills the daily run.

**Phase 5 — operational hardening**, independent small PRs, can run anytime in parallel:
- Freshness check (assert the published file's date matches today) + failure notification.
  Right now if `predict.py` fails, the app keeps serving stale files with no indication.
- Reproducible daily job: `conda env create || conda env update` on every run is slow and
  can drift; a lockfile or prebuilt container image would fix that.
- Transform-in-checkpoint: `data/transform_data.pickle` is a single global file that must
  correspond to the promoted checkpoints, with nothing enforcing that and retraining
  silently rewriting it. Saving transform parameters inside the checkpoint removes the
  coupling entirely.

## Unverified

The count timezone handling looks correct —
`notebooks/processing_count_data.ipynb` uses `tz_localize("Europe/Paris", ...)` then
`tz_convert`, and stored survey start hours peak at 06–07 UTC (08–09 local), consistent
with post-sunrise starts — but this has not been explicitly confirmed. Worth doing once: a
silent one- or two-hour shift between the count mask and the weather time axis would
distort the whole diurnal curve and be nearly invisible in aggregate metrics.
