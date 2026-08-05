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

`centralize-weather` is rebased onto current `main`, tested, and not yet merged — see
Phase 0 below. Everything else identified in the last full audit (August 2026) has merged
or is resolved on that branch, **except** the four items in the next section. None of the
`prod/models/` checkpoints have been retrained against any of it yet.

## Open defects

**4.8 The uncertainty output channel is untrained.** `configs/model/unet.yaml` sets
`nb_output_features: 2`, and `ProbaRMSE.alpha` is 1 everywhere (`model.criterion.rmse.alpha`
is still commented out of the `hparams_search` sweep). With `alpha = 1` the loss is
`1·rmse_term + 0·nll_term`, so channel 1 never receives gradient and stays essentially
random, even though `save_test` exports `pred_count_up`/`pred_count_down` from it. The
app's uncertainty bands come from climatological quantiles, not the model, so nothing
user-facing is wrong today — but the exported bands should not be trusted. Fix: drop to one
output channel, or tune `alpha` in (0, 1).

**4.9 Normalisation statistics are fitted on train + val + test.** `DataTransformer` is
built from the full filtered weather arrays in `defile_datamodule.py` before the split is
used. Mild leakage, worse for being min/max normalisation (outlier-sensitive). Fix: fit on
training years only.

**4.11 There is no season guard.** The cron runs every day of the year, but training is
restricted to `doy` 196–335 (mid-July to end of November). For roughly seven months a year
the job would publish confident extrapolations from a model that has never seen that part
of the year. Fix: skip or clearly flag out-of-season runs, and have the app render them as
unavailable rather than as a forecast.

**4.19d The model trains on 25 km wind and would be served ~2 km wind, and at Défilé those
are substantially different fields.** Unifying the weather provider (below) does *not* fix
this. The ERA5 archive is a 0.25° (~25 km) grid; the Open-Meteo forecast endpoint serves
high-resolution NWP. Measured over 61 days:

| | Défilé (gorge) | Frankfurt (flat) |
|---|---|---|
| `u_component_of_wind_10m` corr | **0.27** | 0.90 |
| `v_component_of_wind_10m` corr | 0.66 | 0.81 |
| `u_component_of_wind_100m` corr | 0.50 | 0.92 |
| `temperature_2m` corr | 0.97 | 0.97 |
| `surface_pressure` corr | 0.98 | 0.99 |
| `u_wind_10m` std ratio (fcst/ERA5) | **1.73** | 0.71 |

Temperature and pressure agree everywhere, so this is resolution, not a mapping error.
ERA5's cell cannot resolve the gorge that makes Défilé a bottleneck, while the forecast
model can, and produces 73% more variance in the along-valley component. The model would
learn wind–passage relationships from a field without the channelling effect, then get
served one with it at inference time. Wind direction is one of the strongest drivers of
raptor passage, so this is a strong candidate for why forecast skill might disappoint even
after every other fix lands.

Two candidate mitigations, neither implemented: pin the forecast endpoint to a coarse
global model (`models=ecmwf_ifs025`) so it matches ERA5's scale — cheap to try, trades
forecast sharpness for train/serve consistency, should be measured not assumed; or train on
Open-Meteo's Historical Forecast API (past forecasts at known lead times) so both sides of
the contract are the same product at the same resolution — the more principled fix, more
work. `tests/test_weather.py::test_wind_over_complex_terrain_is_documented_as_divergent`
records the effect so it can't be quietly forgotten.

## Plan

**Phase 0 — land the structural piece.** Merge `centralize-weather`: one `get_weather()`
entry point (`src/data/weather.py`) for both training and forecasting, replacing the GEE
CSV export and the independent Open-Meteo forecast client that used to drift apart from it.
Blocks everything below — 4.19d's mitigations need the new weather module to exist, and
retraining needs this merged first.

**Phase 1 — three independent, mechanical fixes**, same size as the batch already merged.
Can run in parallel; none touch the same files as each other or as Phase 0.
- `fix/uncertainty-channel` (4.8)
- `fix/normalization-leakage` (4.9)
- `feat/season-guard` (4.11)

**Phase 2 — retrain all 11 species checkpoints**, once Phase 0 + 1 are merged. This is the
actual gate for trusting any model-quality number; nothing before this point should be
judged on accuracy.

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
- **Wind resolution** (4.19d's two mitigations, above) — measure before committing to either.

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
