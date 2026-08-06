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

Environment management is `uv` (`pyproject.toml` + `uv.lock`), not conda —
`environment.yaml` is gone. Dependencies are pinned to exact versions, not ranges, since the
first attempt at ranged constraints silently resolved `torch` and `pyarrow` several major
versions ahead of anything this project has been tested against; bump deliberately with
`uv lock --upgrade-package <name>`, not by loosening the pins.

Weather is centralised on Open-Meteo (`src.data.weather.get_weather`) for both training and
forecasting, with the forecast path pinned to `ecmwf_ifs025` to match ERA5's 0.25° training
grid. A residual wind-agreement gap remains between the two in complex terrain even at
matched resolution (Défilé ~0.55 corr vs. flat-terrain ~0.93,
`tests/test_weather.py::test_wind_over_complex_terrain_is_documented_as_divergent`) — not
something to fix further, but worth checking against Phase 1's new skill metrics once
retrained, to see whether it actually moves forecast skill or only feature correlation.

Tweedie alone is the current default and standard loss (`configs/model/unet.yaml`); `ProbaRMSE`
is fixed and still available in `src/models/criterion.py` for the ablation Phase 1 calls for,
but isn't included by default.

**Decided, not revisiting:** fitting normalisation stats on the full dataset rather than
train-only is fine for this use case. Fine-tuning on Open-Meteo's Historical Forecast API is
out of scope.

**Nothing in `prod/models/` has been retrained against any current fix yet** — that's Phase 1.

## Plan

**Phase 1 — retrain all 11 species checkpoints, and fix what "model quality" means while
doing it.** Retraining is the actual gate for trusting any model-quality number; nothing
should be judged on accuracy before this, including whether the `ecmwf_ifs025` pin above
actually helps. The data is highly skewed (61–95% zero survey rows depending on species; the
top 1% of rows hold 27–70% of all birds counted) and the survey unit changed shape over the
project's history (mean survey duration ~9.7 h in 2013 vs. ~1.0 h from 2022 on, rows/year up
~12x over the same span) — this reweights the loss toward the hourly-recording era by
accident (see the year-subset ladder and `out_h` items in Phase 2, both downstream of the
same fact) — so what gets reported needs to account for both, not just retrain against the
two pooled metrics that exist today.

**Fixed, not yet retrained against:** `UNetplus.__init__` (`src/models/components/unet.py`)
used to hard-zero the network's own output at UTC hours 0–4 and 19–23, ahead of and
independent of the real per-sample survey coverage mask used in the loss. Real survey
coverage starts before 05:00 UTC on 145 of 4,900 days (3%, concentrated in July–August dawn
starts under CEST) — on those days the loss's mask correctly said "count this hour" but the
network was architecturally forced to output zero there regardless, biasing the prediction
down on exactly the days this mattered for. A data check confirms the direction: across
every hourly-resolution dawn/dusk survey row in the dataset, the 11 modelled raptor species
account for only 2 individuals total (thermal-soaring raptors genuinely don't move at
dawn/dusk), so the old bug cost little in volume, but it was pure downside on precisely those
rare, real records. Dropped the hardcoded mask; the per-sample mask already applied in
`applyMask`/`src/models/criterion.py` is correct and sufficient on its own. (`convnet.py`/
`transformer.py` carry the same pattern but aren't wired into any config, so they were inert,
not affected.) Consequence to watch for once retrained: hours no survey has ever covered
(deep night) now get no gradient at all, rather than a guaranteed zero — check the mean
diurnal profile panel in the test report (intra-day shape metric, below) stays sane there.

#### Loss function

- **Row weighting — a config flag (`data.loss_weighting`), not a fixed choice**, since the
  right answer isn't obvious: a 6am–7pm survey and a 10am–2pm survey get very different
  weight under raw-duration weighting even though most of the long survey's extra hours may
  be near-zero activity, and the model's active window (the hard `pred_mask` removed above
  only ever spanned it anyway) is still 05–19 UTC conceptually. Three options to test:
  - `"none"` (status quo) — baseline for comparison.
  - `"active_overlap"` — weight = overlap between the survey mask and the model's 05–19 UTC
    active window, so a short midday survey inside it isn't penalised relative to a long
    dawn-to-dusk one, and no row is penalised for hours the model can't predict.
  - `"phenology"` — weight by expected activity from an **hourly** phenology baseline, so a
    midday hour in peak season counts for more than one in the off-peak fringe.
    `species_doy_statistics.json` already carries this (a GAM-fitted `ratio` field, hour
    of day x day of year — see `scripts/build_phenology_stats.py` and
    `src.phenology.Phenology.hourly_rate`), so this is a config flag now, not a data-prep
    task.

#### Reporting and metrics

Report one, at most two, complementary metrics per level — no redundant variants, though
extra diagnostics can be computed without being shown. All of them reported **per species
and per era, never pooled** (Merlin at 95% zero and Common Buzzard at 65% zero are different
problems), and all with a **skill score against day-of-year phenology**
(`1 − score_model / score_phenology`, using `data/count/species_doy_statistics.json`) and
against **persistence** (yesterday's count) as naive baselines — if the model doesn't beat
phenology, the weather features aren't contributing anything, and no raw metric value
alone will show that. This is purely an evaluation-time comparison, not a training input: a
second scoring pass over the same predictions. Cheap enough to also log every validation
epoch (`val/skill_vs_phenology`), not just at final test time. Caveat: the phenology
file has no `year` field — it's pooled across all years including whatever ends up in the
test split, a mild leakage risk on the baseline side; worth rebuilding per-split if a skill
score ever looks suspiciously good, not blocking to start with. (Also already live in
production as defileViz's stand-in uncertainty band, since the model's own uncertainty
channel was dropped as untrained — this reuses something that already has a job in the
running system.)

| level | headline metric(s) | computed, not headlined |
|---|---|---|
| 1. Row | **MAE** + **Bias**, birds/hr | Tweedie deviance itself (it's the loss; redundant as a metric) |
| 2. Day (event) | **CSI** vs. a per-species-doy phenology threshold (e.g. p90) | full hit/miss/false-alarm/correct-rejection counts, for when CSI looks wrong and you need to know why |
| 3. Intra-day shape (hourly-res. dates only) | **Peak-hour error** (hours) | Wasserstein/EMD distance (catches shape distortion even when the peak hour is right; keep computing it, don't headline two shape numbers) |
| 4. Season (phenology) | **Median passage-date error** (days) + **seasonal total ratio** | 10%/90% passage dates (only worth surfacing if the median error is large and you need to know whether early- or late-season passage is driving it) |

Package all of this — the table above, skill scores, and the existing per-year diagnostic
plots (`plt_doy_sum`, `plt_timeseries`, `plt_true_vs_prediction`, `plt_counts_distribution`
in `src/plots/save_test.py`) — into **one consolidated PDF/PNG report per species per run**,
replacing the current scattered plot files. This is the tool every Phase 2 experiment (year
ladder, location/variable ablation) gets judged against, so it needs to exist before those
comparisons are meaningful, not after.

**Inter-annual skill needs more than the current split can give it.** The random-period
split yields ~3 test years, not enough for a year-tracking correlation to mean anything.
Making the season-level metrics above trustworthy across years needs leave-one-year-out or
rolling-origin cross-validation — a real change to the eval harness, not just a metric
addition, and the same underlying limitation the chronological-holdout idea in Phase 2
addresses. Worth knowing before promising that number works from day one.

**Phase 2 — general modelling research, branch per experiment, not urgent to land quickly.**
The main one: **location and variable selection.** `era5_main_variables`,
`era5_hourly_locations`/`variables` and `era5_daily_locations`/`variables` were chosen once,
at the start of the project, with no documented ablation since. Systematically test which
locations and variables actually earn their place — leave-one-out or permutation-importance
ablation on a trained model (the existing saliency/SHAP machinery in
`src/plots/explanations.py` and `configs/model/unet.yaml`'s `compute_saliency` can inform
this), or an Optuna sweep over feature subsets. The goal is a smaller, justified feature set,
not just accuracy: fewer inputs means less surface area for the next drifted-unit or
wrong-convention bug, and a smaller model to retrain each time a fix lands. (The wind
resolution gap noted in Status is a candidate first case: is Défilé's own wind, measurably
noisier against ERA5 than elsewhere, actually earning its place, or would nearby stations
carry the same signal more reliably?)

Also in scope for this phase:
- **Year selection.** `data/count/readme.md` documents real protocol changes: sporadic
  pigeon-focused coverage before 1993, daily volunteer monitoring from 1993, a salaried
  observer 2008–2016, two salaried observers from 2017. Recording granularity changed too
  (daily totals → hourly forms → Naturalist → Trektellen), which is the same accidental
  reweighting Phase 1 notes. Run a **year-subset ladder**
  (1966+/1993+/2008+/2014+/2017+) on a **chronological** holdout (not the current
  random-years split, which flatters the model by letting it interpolate across eras) —
  cheap, and the single most informative experiment available. `year_used: "period"` is
  available and untested; include it in the sweep. Bigger structural idea worth a
  follow-up: predict the *share* of the season's total per day/hour and forecast the
  annual total separately, removing most year-to-year variance from the hard part of the
  problem.
- **Direct shape supervision for `out_h`.** The architecture already splits into two
  pieces multiplied together: `out_h` (24 values, the *relative shape* of the day) and
  `out_d` (one value, the *overall size* of the day) — `out = 8 · out_h · out_d`. Training
  today only ever checks the *combined* prediction against the survey's total count; nothing
  directly checks whether `out_h`'s shape resembles the true shape of the day. But for
  dates recorded as several one-hour blocks (common since 2014, near-universal since 2021),
  the true hourly counts already exist in the data — e.g. 12 one-hour survey rows on one
  date *are* an hour-by-hour count, currently only used as 12 near-duplicate training rows
  (the same reweighting effect again), never as a shape. Idea: on those dates, normalise
  the true hourly counts to sum to 1 (shape only, not magnitude) and add a direct loss term
  comparing that to similarly-normalised `out_h` — teaching the shape sub-network directly
  instead of only through the combined total. Unlike the (already-fixed) `ProbaRMSE`
  inconsistency, this is a genuinely new term, using different, finer-grained data, to
  supervise a sub-output nothing currently teaches directly. Exploratory — worth a small
  experiment before committing to it further.
- **For later, lower priority: a probability envelope from the Tweedie loss itself**,
  rather than a second model-predicted output channel (the approach already dropped for
  being untrained). The Tweedie distribution already has a defined variance-mean
  relationship (`Var(Y) = φ·μ^p`, `p` already fitted per species), which may be enough to
  derive calibrated uncertainty bands analytically instead of learning a second channel.
  Would also replace the climatological-quantile bands the app currently shows, which don't
  come from the model at all.

**Phase 3 — Trektellen counts as model input.** Plumbing already exists: a working proxy at
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

**Phase 4 — operational hardening**, independent small PRs, can run anytime in parallel:
- Freshness check (assert the published file's date matches today) + failure notification.
  Right now if `predict.py` fails, the app keeps serving stale files with no indication.
- Transform-in-checkpoint: `data/transform_data.pickle` is a single global file that must
  correspond to the promoted checkpoints, with nothing enforcing that and retraining
  silently rewriting it. Saving transform parameters inside the checkpoint removes the
  coupling entirely.
