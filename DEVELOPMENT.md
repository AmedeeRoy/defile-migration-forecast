# Défilé migration forecast — development roadmap

This tracks what's actually left to do. Resolved items move to `DECISIONS.md` (a running
log of settled calls and what was tried) rather than being archived here. If something
below looks wrong or already done, say so before it gets deleted rather than after.

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

Environment management is `uv` (`pyproject.toml` + `uv.lock`); dependencies are pinned to
exact versions (bump deliberately with `uv lock --upgrade-package <name>`). Weather is one
Open-Meteo path for training and serving. What was settled and why lives in `DECISIONS.md`
(weather, architecture, loss, evaluation, production guards); this file only lists what is
left.

**Nothing in `prod/models/` has been retrained against any current fix** — the committed
checkpoints date from 2025-09, before the mask removal, the phenology shape prior, the
Tweedie-only loss and the weather migration. The hyperparameters in `configs/experiment/*.yaml`
were tuned before all of these and should be treated as void until retrained. That is
Phase 1, and it is the gate for trusting any model-quality number.

## Plan

**Phase 1 — retrain all 11 species, and fix what "model quality" means while doing it.**
Nothing should be judged on accuracy before this, including whether the `ecmwf_ifs025` pin
actually helps forecast skill or only feature correlation (the residual wind gap at Défilé,
~0.55 vs ~0.93 over flat terrain, is in `DECISIONS.md`). The data is highly skewed (61–95%
zero survey rows depending on species; the top 1% of rows hold 27–70% of all birds counted)
and the survey unit changed shape over the project's history (mean survey duration ~9.7 h in
2013 vs. ~1.0 h from 2022 on, rows/year up ~12x over the same span). That reweights the loss
toward the hourly-recording era by accident, so what gets reported must account for both.

Things to check on the first retrain:

- Hours no survey has ever covered (deep night) get no gradient, only the phenology prior
  anchoring them: confirm the mean diurnal profile panel in the test report stays sane there.
- Whether `ProbaRMSE` (kept in `src/models/criterion.py`, off by default) earns a place
  in an ablation against the Tweedie-only baseline.

#### Loss function

- **Row weighting — proposed, not implemented.** No `data.loss_weighting` flag exists yet.
  The right answer isn't obvious: a 6am–7pm survey and a 10am–2pm survey get very different
  weight under raw-duration weighting even though most of the long survey's extra hours may
  be near-zero activity. Three options to test behind a config flag:
  - `"none"` (status quo) — baseline for comparison.
  - `"active_overlap"` — weight = overlap between the survey mask and the 05–19 UTC active
    window, so a short midday survey isn't penalised relative to a long dawn-to-dusk one.
  - `"phenology"` — weight by expected activity from the **hourly** phenology baseline, so a
    midday hour in peak season counts for more than one in the off-peak fringe.
    `species_doy_statistics.json` already carries the needed hour x day-of-year `ratio`
    (`src.phenology.Phenology.hourly_rate`), so this needs no new data prep.

#### Reporting and metrics

The metric set and consolidated PDF report have landed (see `DECISIONS.md` → Evaluation); what
remains is using them and closing their known gaps:

- **Inter-annual skill needs more than the current split can give it.** The random-period
  split yields ~3 test years, not enough for a year-tracking correlation to mean anything.
  Trustworthy season-level metrics across years need leave-one-year-out or rolling-origin
  cross-validation — a real change to the eval harness, and the same limitation the
  chronological-holdout idea in Phase 2 addresses.
- **The phenology baseline is pooled across all years**, including the test split (the file
  has no `year` field): a mild leakage risk on the baseline side. Rebuild per-split if a
  skill score ever looks suspiciously good.
- Consider logging `val/skill_vs_phenology` every validation epoch, not just at test time.

The metric levels, for reference (all reported per species and per era, with skill scores vs.
phenology and persistence):

| level                                       | headline metric(s)                                              | computed, not headlined                                                        |
| ------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1. Row                                      | **MAE** + **Bias**, birds/hr                                    | Tweedie deviance itself (it's the loss; redundant as a metric)                 |
| 2. Day (event)                              | **CSI** vs. a per-species-doy phenology threshold (e.g. p90)    | full hit/miss/false-alarm/correct-rejection counts                             |
| 3. Intra-day shape (hourly-res. dates only) | **Peak-hour error** (hours)                                     | Wasserstein/EMD distance (keep computing it, don't headline two shape numbers) |
| 4. Season (phenology)                       | **Median passage-date error** (days) + **seasonal total ratio** | 10%/90% passage dates                                                          |

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
- **Validate the `out_h` phenology-prior anchoring beyond Common Buzzard.** The collapse fix
  (`DECISIONS.md` → Model architecture) has only been validated on Common Buzzard, 3 seeds.
  Follow-ups:
  - Check every other species before trusting or promoting it. The tell for a collapse is
    near-identical shape metrics across a seed sweep.
  - Shape accuracy came out slightly, consistently lower than the previous architecture.
    The learning rate and the `out_h`/`out_d` output scale were tuned for the old
    architecture; retune deliberately once retrained more broadly.
  - **Red Kite reproduces a different, still-open collapse under this fix, on every seed
    tested** — the hourly shape goes flat within a day (though it still varies normally
    between days), even though its own climatological prior looks fine. Not present on the
    previous architecture. Best guess: Red Kite has by far the strongest multi-year
    population trend of any species and is trained with year information withheld
    (`year_used: "constant"`), so a large, systematic part of the variation in count is
    unexplainable from the model's inputs, and saturating the hourly output is a shortcut.
    The annual-trend-correction work (a separate branch) only corrects the model's *output*
    after prediction, so it likely doesn't fix this alone. Worth testing whether training
    against trend-adjusted rates removes the unexplainable variance. Until understood, don't
    retrain or promote Red Kite against this architecture.
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
  The publish guard (`DECISIONS.md` → Production safety) now makes a degenerate forecast fail
  the job rather than reach the app, but nobody is told when that happens, and the app keeps
  serving yesterday's file with no indication.
- Transform-in-checkpoint: `data/transform_data.pickle` is a single global file that must
  correspond to the promoted checkpoints, with nothing enforcing that and retraining
  silently rewriting it. Saving transform parameters inside the checkpoint removes the
  coupling entirely.
