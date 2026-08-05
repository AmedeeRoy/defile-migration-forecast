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
forecasting), the season guard (4.11), the uncertainty-channel drop (4.8), and 4.19d (the
forecast path is now pinned to `ecmwf_ifs025`, matching ERA5's 0.25° grid — see below). 4.9
(normalisation leakage) was fixed and then reverted — Raphaël's call, fitting on the full
dataset isn't a real problem for this use case, so it's closed as won't-fix. Training on
Open-Meteo's Historical Forecast API (fine-tuning at matched lead times) is also closed as
won't-fix. None of the `prod/models/` checkpoints have been retrained against any of this
yet (Phase 2, below).

**4.19d, resolved but with a residual worth knowing about.** Root cause was confirmed, not
just suspected: the forecast path didn't set `models=`, so Open-Meteo's `best_match` picked
a model per location, resolving to **DWD ICON-D2 at 2 km** at Défilé — confirmed by matching
returned grid coordinates against an explicit `models=dwd_icon_d2` request. Training uses
ERA5 at 0.25° (~25-28 km) — a ~12x difference in grid spacing. Pinning the forecast path to
`ecmwf_ifs025` (same 0.25° grid as `era5`, confirmed against the live API) measurably closes
most of that gap:

| vs ERA5 archive, at Défilé | default (ICON-D2, 2 km) | `ecmwf_ifs025` (0.25°) |
|---|---|---|
| `u_component_of_wind_10m` corr | 0.27 | **0.55** |
| `v_component_of_wind_10m` corr | 0.66 | **0.80** |
| `u_wind_10m` std ratio (fcst/ERA5) | 1.73 | **0.95** |

There's a domain reason to prefer the coarser product beyond consistency, too: birds
crossing Défilé integrate wind over a section of the gorge at least 2-5 km wide, wider than
ICON-D2's native resolution — 25 km is arguably closer to the biologically relevant scale.

A residual gap remains even at matched resolution — at flat Frankfurt the same comparison
reaches ~0.93 corr, well above Défilé's 0.55 — because ERA5 is a reanalysis that
assimilates observations and IFS-025 is a raw forecast, and terrain still amplifies that
distinction. `tests/test_weather.py::test_wind_over_complex_terrain_is_documented_as_divergent`
tracks it so it isn't forgotten. Not something to fix further; a real property of comparing
these two products in complex terrain.

## Plan

**Phase 0 — done.** `centralize-weather` merged: one `get_weather()` entry point
(`src/data/weather.py`) for both training and forecasting, replacing the GEE CSV export
and the independent Open-Meteo forecast client that used to drift apart from it.

**Phase 1 — done.** `fix/uncertainty-channel` (4.8) and `feat/season-guard` (4.11) merged.
`fix/normalization-leakage` (4.9) was merged, then reverted — see Status above.

**Phase 2 — retrain all 11 species checkpoints, and fix what "model quality" means while
doing it.** Retraining is the actual gate for trusting any model-quality number; nothing
before this point — including the `ecmwf_ifs025` pin's actual effect on forecast skill, not
just feature correlation — should be judged on accuracy. But the two metrics currently
reported (`test/r2_score`, `test/spearman_coeff` in `defile_module.py`) pool every test-set
hour into one number, which can't tell a model that gets the *average* right from one that
gets the *shape* right. Add metrics for the three things that actually matter for this
forecast and are invisible in a pooled r2:
- **Peak reproduction** — error in predicted peak day-of-year and peak magnitude, per year
  per species, not just average error across all days.
- **Intra-day shape** — correlation or RMSE between predicted and true *diurnal* pattern
  within a survey day (normalise out the daily magnitude first, so this isolates shape from
  total count). This is the metric this project's whole hourly-rate framing has been missing.
- **Inter-annual variation** — does the model track which years had more or less passage,
  not just perform well on average across years pooled together.

Package these, plus the existing per-year diagnostic plots (`plt_doy_sum` already plots
true-vs-predicted daily curves per test year; `plt_timeseries`, `plt_true_vs_prediction`,
`plt_counts_distribution` in `src/plots/save_test.py`), into **one consolidated PDF/PNG
report per species per run** rather than the current scattered plot files — this becomes the
tool used to judge every Phase 3 experiment (year ladder, location/variable ablation)
against, so it needs to exist before those comparisons are meaningful, not after.

**Phase 3 — general modelling research, branch per experiment, not urgent to land quickly.**
The main one: **location and variable selection.** `era5_main_variables`,
`era5_hourly_locations`/`variables` and `era5_daily_locations`/`variables` were chosen once,
at the start of the project, with no documented ablation since. Systematically test which
locations and variables actually earn their place — leave-one-out or permutation-importance
ablation on a trained model (the existing saliency/SHAP machinery in
`src/plots/explanations.py` and `configs/model/unet.yaml`'s `compute_saliency` can inform
this), or an Optuna sweep over feature subsets. The goal is a smaller, justified feature set,
not just accuracy: fewer inputs means less surface area for the next drifted-unit or
wrong-convention bug, and a smaller model to retrain each time a fix lands.

Also in scope for this phase:
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

For later, lower priority: **a probability envelope from the Tweedie loss itself**, rather
than a second model-predicted output channel (the approach 4.8 removed for being
untrained). The Tweedie distribution already has a defined variance-mean relationship
(`Var(Y) = φ·μ^p`, `p` already fitted per species in `TweedieLoss`), which may be enough to
derive calibrated uncertainty bands analytically instead of learning a second channel. Would
also replace the climatological-quantile bands the app currently shows, which don't come
from the model at all.

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
