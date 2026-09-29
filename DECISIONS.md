# Défilé migration forecast — decisions log

A running log of settled calls, kept so later work knows what's already been tried,
what worked, and what was rejected and why. Not a design spec, and deliberately not
tied to specific files or lines — those drift; read the code for that. See
`DEVELOPMENT.md` for what's still open.

## Weather

**One weather path for training and serving.** Training used to read GEE-exported CSVs
and the daily job used Open-Meteo, two independent implementations whose unit conversions,
wind convention and daily aggregation had silently drifted apart — in ways no model metric
would have surfaced. Everything now goes through a single Open-Meteo entry point
(`src.data.weather.get_weather`) with one variable vocabulary, one set of pinned request
units and one daily-aggregation rule per variable; training reads a local Parquet cache of
the ERA5 archive, serving calls the forecast API. `tests/test_weather.py` enforces the
conventions and the train/serve contract.

**Forecast path pinned to `ecmwf_ifs025`.** Open-Meteo's default forecast blend uses
higher-resolution models (DWD ICON-D2, 2 km) that resolve the gorge Défilé sits in, which
ERA5's 25 km cell cannot. Before the pin, 10 m wind correlated only ~0.27 between the two
products at Défilé (vs ~0.90 over flat terrain); pinning to the 0.25° IFS closed most of it
(~0.55 at Défilé, ~0.93 flat). What remains is a real reanalysis-vs-forecast difference that
bites harder in complex terrain, and is accepted as-is: not a thing to fix further, only to
re-check once retrained (see `DEVELOPMENT.md`).

**Normalisation statistics are fitted on the full dataset, not train-only.** Fine for this
use case; not revisiting.

**Fine-tuning on Open-Meteo's Historical Forecast API is out of scope.**

## Model architecture

**Single output channel; the uncertainty channel is gone.** The second output channel fed
an NLL term that never received gradient (`alpha=1` zeroed its contribution), so it was
untrained. Dropped rather than fixed; defileViz shows the phenology file's quantile spread
as its uncertainty band instead.

**Tweedie alone is the default loss.** It is already a proper scoring rule for zero-inflated
skewed counts, with a per-species `p`. `ProbaRMSE` used to be added on top but targeted the
same quantity in a way that disagreed with Tweedie on peaked days; it is fixed and still in
`src/models/criterion.py`, but its benefit is untested, so it stays out until an ablation
against this baseline says otherwise (`DEVELOPMENT.md`, Phase 1).

**The hardcoded dawn/dusk output mask was removed.** The network used to hard-zero its own
output at UTC hours 0–4 and 19–23, independent of the per-sample survey coverage mask the
loss applies. Real survey coverage starts before 05:00 UTC on 145 of 4,900 days (3%, mostly
July–August dawn starts under CEST), so on those days the loss said "count this hour" while
the network was forced to output zero — biasing predictions down on exactly those days. The
volume at stake is tiny (across every hourly-resolution dawn/dusk row, the 11 modelled
species total 2 birds — thermal-soaring raptors don't move then), so this was pure downside
on rare, real records. The per-sample mask in the loss is correct and sufficient on its own.
(`convnet.py`/`transformer.py` carry the same pattern but aren't wired into any config.)
Consequence: hours no survey has ever covered (deep night) got no gradient at all, which is
what let the collapse below happen.

**The hourly shape sub-network could collapse to a flat, constant prediction** after that
mask removal — confirmed directly on some seeds after retraining. Root cause: the training
loss only ever checked the *combined* prediction's masked average, never the hourly shape on
its own, so a flat hourly output paired with a correctly-scaled overall magnitude could
satisfy the loss without learning any real diurnal shape.

Two loss-side penalties were tried and rejected in favour of an architectural fix:
penalizing raw night-hour output directly, and directly supervising shape against real
hour-by-hour survey data on the subset of dates that have it. Both raised the cost of
collapsing without removing the collapse basin itself, and the second only ever covered
a small fraction of dates.

**Landed instead: anchor the hourly network's default output to a smooth climatological
shape**, built from the existing day-of-year phenology baseline (with deep night forced
to zero from sun position rather than fitted, since there's essentially no real data to
fit against there), with the network free to override that default wherever real
weather evidence justifies it. At initialization the network reproduces the
climatological shape almost exactly, removing the flat/constant state that training
could fall into when it has little else to go on. Night is discouraged, not forbidden — no
hour is ever architecturally impossible to predict, which is the mistake the hard mask made.

**First attempt at anchoring was wrong and rejected**: normalizing the hourly output
into a probability distribution over the day (a softmax) forced all magnitude information
into the daily sub-network alone, cutting the model's ability to predict large counts by
roughly 7x in testing. Fixed by anchoring each hour's *default level* independently
(a sigmoid logit bias) instead, without forcing the 24 hours to compete for a fixed total —
this keeps both sub-networks contributing to magnitude, matching the intended split between
daily-scale and hourly-scale information.

**Confirmed with a 3-seed comparison against the previous architecture, on Common
Buzzard**: the collapse did not recur on any seed, and shape metrics stayed
consistently tight across seeds instead of varying widely as they did when a collapse
occurred. Overall magnitude was comparable to before; shape accuracy was slightly,
consistently a bit lower — a small tradeoff, accepted for now, worth revisiting (see
`DEVELOPMENT.md`).

## Evaluation

**Skill is judged against day-of-year phenology and persistence, per species and per era,
never pooled.** If the model doesn't beat phenology the weather features aren't contributing,
and no raw metric shows that. Four metric levels (row, day/event, intra-day shape, season)
with one or two headline numbers each; the extra diagnostics are computed but not headlined.
Everything is packaged into one consolidated PDF report per species per run. Era boundaries
and the baseline machinery each have a single owner in the code (see `AGENTS.md`,
"Constants"). The phenology file's generator moved out of a notebook into
`scripts/build_phenology_stats.py` after the notebook drifted from the committed file's
schema, `pygam` turned out never to have been a pinned dependency, and the hour-of-day grid
had a one-day off-by-one.

**Metrics are gathered across DDP ranks before scoring, and files are written by the
coordinating rank only.** Otherwise `trainer=ddp` (selectable, unused today) would silently
score one rank's shard while every rank raced to write the same outputs.

## Training reproducibility

**Training is now fully deterministic.** The same seed and config used to sometimes
produce different outcomes across runs (notably on `trainer=mps`), which made it hard to
tell whether a change actually mattered or was just noise. Verified directly: repeated runs
with the same seed now produce identical results.

## Production safety

**The daily job refuses to run out of season.** The model has never seen the seven months
outside `data.doy`, so a forecast then is ungrounded; the job skips it rather than publish one.

**The daily job refuses to publish a degenerate forecast.** A non-finite forecast, or one
whose peak is below ~1e-6 log1p(birds/h), fails the job instead of being written: to site
visitors it would look like a confident "no birds today", whereas a failed job leaves
yesterday's file in place. Motivated by the stale-checkpoint forecast of 2026-08-06, which
peaked at ~1e-10. Failure is still *silent to a human* — notification is open in
`DEVELOPMENT.md`, Phase 4.
