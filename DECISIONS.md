# Défilé migration forecast — decisions log

Settled calls, and what was tried and rejected, so later work doesn't redo it. Not a design
spec and not tied to files or lines (read the code for that). Open work is in `DEVELOPMENT.md`.

## Weather

**One weather path for training and serving.** Training used to read GEE-exported CSVs and
the daily job Open-Meteo; their unit conversions, wind convention and daily aggregation had
silently drifted apart, in ways no model metric would show. Everything now goes through
`src.data.weather.get_weather`: one variable vocabulary, one set of pinned request units, one
daily-aggregation rule per variable. Training reads a local Parquet cache of the ERA5 archive,
serving calls the forecast API. `tests/test_weather.py` enforces the train/serve contract.

**Forecast pinned to `ecmwf_ifs025`.** Open-Meteo's default blend includes 2 km models that
resolve the gorge at Défilé, which ERA5's 25 km cell cannot. Pinning to the 0.25° IFS raised
the 10 m wind correlation with ERA5 at Défilé from ~0.27 to ~0.55 (~0.93 over flat terrain).
The rest is a real reanalysis-vs-forecast gap in complex terrain, accepted as-is.

**Normalisation is fitted on the full dataset, not train-only.** Fine here; not revisiting.

**Fine-tuning on Open-Meteo's Historical Forecast API is out of scope.**

## Model architecture

**One output channel.** The second (uncertainty) channel fed an NLL term that never received
gradient. Dropped; defileViz shows the phenology quantiles as its uncertainty band instead.

**Tweedie alone is the loss**, with a per-species `p`. `ProbaRMSE` (fixed, still in
`src/models/criterion.py`) stays off until an ablation shows it helps.

**No fixed-UTC dawn/dusk output mask.** The network used to zero hours 0–4 and 19–23 UTC, but
surveys start before 05 UTC on 3% of days (July–August), where the loss then scored an hour
the network could not predict. Removing it left never-surveyed night hours without gradient.

**The hourly output is anchored to a phenology prior.** Without it, the hourly sub-network
collapsed on some seeds to a flat shape, since the loss only sees the combined prediction.
Rejected: a night-output penalty, and direct shape supervision on hourly-resolution dates
(both raised the cost of collapsing without removing it). Also rejected: a softmax over the
day, which pushed all magnitude into the daily branch (`season_total_ratio` ~0.14). Kept: a
per-hour sigmoid logit bias from the prior, so at init the network reproduces the
climatological shape and each hour still carries magnitude. 3 seeds on Common Buzzard: no
collapse, slightly lower shape accuracy.

**Night is defined by the sun, and forbidden (2026-10).** The prior alone did not hold night:
Red Kite, Sparrowhawk and Marsh Harrier put 50%, 38% and 18% of predicted birds there. Now
one rule (sun below −6° for the whole hour) zeroes both the prior and `out_h`. Unlike the old
mask it follows the season, so July's dawn surveys stay predictable. Red Kite went from −0.28
to +0.16 skill vs phenology. The prior's hourly grid (`RATIO_HOURS`) was also widened to every
surveyed hour (04–18 UTC), with days weighted by √count (held-out L1 on hourly profiles
0.277 → 0.213). The 2026-09 Black Kite model's 2 964 birds/h spike (`out_h` ≈ 1 at 18 UTC)
did not recur after this (test max 152/h, observed 376/h).

**In the daily branch, BatchNorm comes before the ReLU (2026-10).** Hen Harrier predicted
25 birds/h on 2019-07-27, a day it was absent (max observed 4/h), with `out_d` at 1.000
against a median of 0.008. Out-of-range weather was ruled out: clipping every input to the
training range left the logit at 336 (from 550). The cause was Conv → ReLU → BN: a channel
that is zero after the ReLU on almost every day has a running variance near 0 (down to
1e-16), so in eval mode BN multiplies it by up to ~470×. On the rare day it fires, the daily
output saturates. Training mode uses batch statistics, so the loss never sees it. 9 of 11
species had 10–28 such channels out of 128. Conv → BN → ReLU has none, with gain ≤ 3.3, and
no day in any species saturates.

Skill vs phenology at seed 0, both arms in one sweep (`logs/bn_baseline*`, `logs/bn_fix*`):
mean 0.155 → 0.202, median 0.198 → 0.168, 5 of 11 better. The changes outside Hen Harrier
(−0.27 → 0.25) are within seed noise: 3 seeds give 0.25–0.45 for Common Buzzard alone. Over
3 seeds × 3 species (Hen Harrier, Black Kite, Common Buzzard) mean skill is equal or better,
with a narrower spread. Rejected without retraining: input clipping (above), bounding the
daily logit (any useful bound also caps Black Kite's real `out_d` ≈ 1 peak days), and weight
decay (already on, and it does not touch BN's running variance).

## Evaluation

**Skill is judged against day-of-year phenology and persistence, per species and per era.**
If the model doesn't beat phenology, the weather isn't contributing. Four levels (row, day,
intra-day shape, season), one or two headline numbers each, in one PDF report per species per
run. The phenology file is built by `scripts/build_phenology_stats.py`, not a notebook (the
notebook drifted from the committed schema and had a one-day off-by-one in the hour grid).

**Metrics are gathered across DDP ranks, and files written by rank 0 only.**

## Training reproducibility

**Training is deterministic for a given seed, config and thread count.** Changing the number
of CPU threads changes the model (Hen Harrier seed 0: skill −1.34 with default threads, −0.27
with 2). Compare runs only within one sweep. `trainer=mps` gives NaN gradients in
deterministic mode; use `trainer=cpu`.

## Production safety

**The daily job refuses to run out of season.** The model has never seen the months outside
`data.doy`.

**No automatic publish guard.** Whether a model is fit to publish is a human review of its
test report before promotion. Each forecast is traceable instead (`species`, `issued_at`,
`weather_model`, `checkpoint_sha256`, `git_sha`), and each species runs in its own process.
