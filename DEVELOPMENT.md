# Défilé migration forecast — development roadmap

Only what is still open. Settled calls and rejected attempts go to `DECISIONS.md`, not here.
If something below looks wrong or already done, say so before deleting it.

## What we are building

An autonomous system that, every morning without human intervention, publishes a forecast of
the **hourly passage rate of migrating raptors at Défilé de l'Écluse**, per species, for today
and the next few days, consumed by [defileViz](https://github.com/Rafnuss/defileViz).

Keep in view: the unit is an **hourly rate**, so the shape of the day matters as much as the
daily total. The system runs **unattended**, so a silently wrong or stale forecast is as bad
as a crash. Only features available at 03:00 UTC on the day exist.

## Status

All 11 species were retrained on the current architecture on 2026-10-01 (`trainer=cpu`,
seed 0) and promoted to `prod/models/`. The hyperparameters in `configs/experiment/*.yaml`
were tuned on an older architecture and loss and have not been re-tuned. Single-seed skill
moves by ±0.1 from seed alone, so no model comparison should rest on one seed.

Where the models stand (seed 0, skill vs phenology / season total ratio): Hen Harrier,
Merlin, Common Buzzard, Honey Buzzard and Black Kite are reasonable (skill 0.18–0.35, ratio
0.7–1.5). Hobby, Kestrel, Sparrowhawk and Osprey over-predict the season 1.5–2.8×. Red Kite
over-predicts 13.6×. Marsh Harrier has the lowest skill (0.09).

## Plan

**Phase 1 — make "model quality" trustworthy.**

- **Multi-seed evaluation by default.** Report the mean and spread over ≥3 seeds
  (`hydra=seed_sweep`). The thread count is now pinned (`num_threads`), so runs on different
  machines are comparable.
- **Re-tune the hyperparameters** (learning rate, weight decay, Tweedie `p`, the `8 *` output
  scale) on the current architecture, per species, against multi-seed skill.
- **Season over-prediction** (Hobby, Kestrel, Sparrowhawk, Osprey, Red Kite): find whether it
  is particular eras, years or days.
- **`ProbaRMSE` ablation** against the Tweedie-only baseline.
- **Row weighting**, behind a config flag, not implemented. Raw-duration weighting favours long
  surveys whose extra hours are mostly empty. Options: `"none"` (status quo),
  `"active_overlap"` (overlap with the 05–19 UTC active window), `"phenology"` (expected
  activity from `Phenology.hourly_rate`).
- **The hourly U-Net also applies BatchNorm to ReLU outputs** (`DownConv` after the first
  layer, `UpConv`): the same pattern that saturated the daily branch. No degenerate channels
  were found there yet, so it is unchanged; reorder it if one shows up.

Reporting gaps:

- The random-period split yields ~3 test years per era, too few for year-to-year skill. That
  needs leave-one-year-out or rolling-origin cross-validation.
- The phenology baseline is pooled over all years, test split included: mild leakage on the
  baseline side. Rebuild it per split if a skill score looks suspiciously good.
- Consider logging `val/skill_vs_phenology` every validation epoch.

| level          | headline metric(s)                                     | computed, not headlined     |
| -------------- | ------------------------------------------------------ | --------------------------- |
| 1. Row         | **MAE** + **Bias**, birds/h                            | Tweedie deviance            |
| 2. Day (event) | **CSI** vs a per-species-doy phenology threshold (p90) | hit/miss/false-alarm counts |
| 3. Intra-day   | **Peak-hour error** (hours), hourly-resolution dates   | Wasserstein/EMD distance    |
| 4. Season      | **Median passage-date error** + **season total ratio** | 10%/90% passage dates       |

**Phase 2 — modelling research, one branch per experiment.**

- **Location and variable selection.** The ERA5 locations and variables were chosen once,
  with no ablation since. Test which earn their place (permutation importance, the saliency in
  `src/plots/explanations.py`, or an Optuna sweep over subsets). Defile's own wind, the
  noisiest against ERA5, is a natural first case.
- **Year selection.** Survey protocol changed in 1993, 2008 and 2017, and recording
  granularity several times (`data/count/readme.md`). Run a year-subset ladder
  (1966+/1993+/2008+/2014+/2017+) on a **chronological** holdout, and include
  `year_used: "period"`. Follow-up idea: predict each day's *share* of the season and the
  annual total separately.
- **Red Kite magnitude.** Pre-1993 test years over-predicted 22–37×, recent years
  under-predicted (0.4–0.7×). Likely its strong population trend, trained with the year
  withheld (`year_used: "constant"`). Try trend-adjusted rates, or hourly-era years only.
- **Lower priority: uncertainty from the Tweedie variance** (`Var(Y) = φ·μ^p`) instead of the
  climatological quantile band defileViz shows now.

**Phase 3 — Trektellen counts as model input.** A proxy exists
(`https://defile.raphaelnussbaumer.com/trektellen/{siteId}/{yyyymmdd}`) and a NW-Europe export
in `data/Trektellen_raptor_2015_2024/`. Missing days need an observed-flag channel, not zeros
(zero means "watched, saw nothing"). Value decays with lead time. Start with Défilé's own lag
(`feat/trektellen-defile-lag`), then upstream sites (`feat/trektellen-upstream-sites`). The
API on the forecast path needs a fallback, not an exception that kills the run.

**Phase 4 — operational hardening**, small independent PRs:

- Freshness check (the published file's date is today) and a failure notification: the app
  keeps serving the last file with no sign it is stale.
- Store the normalisation inside the checkpoint: `data/transform_data.pickle` must match the
  promoted checkpoints, and nothing enforces it.
