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

## Counts

**The model reads the defile-dataset release tables (`count`, `survey`, `taxonomy`), not its
interim two-table layout.** One reader for the model and, next, the explore export (#55): the
release's own timing rules (a count's own time, else its survey's interval; Europe/Paris day)
and its per-survey `survey_coverage` replace the old flags. Historical birds are identical
(12 801 055); Trektellen differs only by the named rules below. `tests/test_counts.py` reconciles
the real release when it is present.

**Effort is the survey rows.** Non-bird and explicit "no species" entries no longer exist in the
release, and are not needed: a `complete` survey with no record is a zero row. The old
"counted day" rule (an empty survey alone on its day is a day without counting) is now the
dataset's `survey_coverage = none`; it stays only as a warning check. Its 5 hits on the 2026-10
release are explicit "no species" entries, i.e. real zeros the old rule discarded.

**`partial`/`unknown` surveys are dropped with their birds** (10 surveys, 11 977 birds), not kept
as "the model should learn that rain means few birds". Their gaps are periods nobody counted;
counting them as effort teaches bad weather -> no birds when there was no observer (the same
error as zero-filling a rained-off day). Keeping only their timed hours would keep the hours
with birds and lose the empty ones, biasing rates up. Poor weather is learnt from complete
surveys counted in it.

**Entries timed outside their survey are dropped**, as before (`time_outside_survey`). The
release keeps them at day level, indistinguishable from untimed entries except by
`remark_processing`; folded into an unsplit survey they inflated its rate (56 Black Kites on
2025-08-03), and counted as untimed they un-split a timed count (2025-10-26). The old dataset's
`time_adjusted` entries (164, 4 556 birds, times moved back into the survey) are now in this
group too, "pending correction" in defile-dataset.

**Night hours of a split Trektellen count are not zero-filled.** The release keeps night periods
as recorded (the old dataset clipped them to dusk); 4 counts in 2025 were left open until the
next morning, ~40 h of zeros nobody watched. Night is the sun below -6 deg for the whole hour,
`night_mask_by_doy_hour`, the model's one definition.

**The historical day window is a dataset matter.** The workbook's `startTimeDay`/`endTimeDay`
(declared attendance) is not in the release, so the old historical zero-fill of empty hours on
hour-by-hour days (832 periods, 668 h, 2014-2021) is gone, and no rule here replaces it: whether
an hour without a record inside the declared day was counted is source evidence, needed by the
explore page's effort too. Requested from defile-dataset as empty `complete` survey rows; until
then 2014-2021 lose those zero hours.

**Species are named by the release's English name (AviList first).** Two modelled species
change: Eurasian Kestrel -> Common Kestrel, European Honey-buzzard -> European Honey Buzzard.
Their configs, checkpoint folders, forecast file names and `species_doy_statistics.json` keys
follow; defileViz must follow too. A local eBird-name map was rejected: a second name list kept
in sync by hand.

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
could fall into when it has little else to go on.

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

**Revisited (2026-10): night is now forbidden, by the sun, not by the clock.** The first
retrain of all 11 species showed the prior alone does not hold night: night hours are never
surveyed, so they get no gradient and nothing bounds the logit there. Red Kite, Sparrowhawk
and Marsh Harrier ran away into it (50%, 38% and 18% of predicted test-set birds at night),
which also flattened their daytime shape. Two fixes, each measured on its own commit:

- The prior was zero outside its 6–17 UTC fit grid at every time of year, so 05/18 UTC got
  midnight's −13.8 logit bias although hour 5 is surveyed on 608 days and carries real birds.
  It is now zero *only* where the sun is below −6°; daylight hours outside the grid hold the
  nearest fitted value. This alone cured Marsh Harrier, but not Red Kite or Sparrowhawk.
- `UNetplus.forward` multiplies `out_h` by `prior_shape > 0`. One rule (sun < −6°) defines
  night for both the prior and the output, with no extra input. This is a hard mask, but not
  the one rejected above: that one used fixed UTC hours and cut July's dawn surveys, this one
  follows the season. Red Kite went from −0.28 to +0.16 skill vs phenology and from 6.4 h to
  1.9 h peak-hour error; Sparrowhawk's season total ratio from 4.0 to 1.9.

Cost, not yet sized: Common Buzzard, which had no night problem, scores 0.33–0.44 skill vs
phenology over 3 seeds on this architecture against 0.47 for one seed before it, with a
season total ratio of 1.33–1.45 against 1.13. The seed spread is as large as the gap, so this
needs seeds on both sides before it is called a regression.

Two follow-ups after checking the prior against the raw counts:

- The night mask sampled the sun at the *start* of each hour, so a mostly-twilight dawn hour was
  night, and with the hard mask unpredictable: 482 Red Kites counted in ≤1 h periods at 05 UTC
  in October fell there. An hour is now night only if the sun stays below −6° for all of it
  (0 birds left in night hours). Model metrics moved within seed noise (Red Kite, 3 seeds:
  0.16/0.17/0.16 → 0.04/0.16/0.14, seed 0 an outlier).
- The `ratio` GAM was predicted on 06–17 UTC only, though it was fitted on samples from every
  surveyed hour, and the edge hour was copied into 05/18, overstating both (Marsh Harrier hour
  18: observed 0.15 of the daily rate, prior 0.85). `RATIO_HOURS` is now 04–18, every hour with
  data. The fit also weighted a 2-bird day like a 2 000-bird one, which pulled Honey Buzzard's
  peak to 09 UTC against an observed 13; days are now weighted by √count. Fit on even years,
  scored on odd years' observed hourly profile (L1, 7 species): 0.277 before, 0.253 with the
  wider grid, 0.213 with both (full count weighting: 0.229, a few huge days dominate). Model,
  seed 0 unless noted: Common Buzzard 0.41 → 0.46 skill and season total ratio 1.30 → 1.07;
  Sparrowhawk 0.28 → 0.29 (ratio 2.37 → 1.86); Red Kite 3 seeds 0.04/0.16/0.14 →
  0.15/0.14/0.12; Marsh Harrier 0.21 → 0.15, the one regression, unsized.

Red Kite's remaining over-prediction is its pre-1993 test years (the species was rare then and
`year_used: "constant"` hides the year), not night — see `DEVELOPMENT.md`.

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

**No automatic publish guard, deliberately.** A check refusing non-finite or all-but-zero
forecasts was written and dropped: during development, deciding whether a model is fit to
publish is a human review of its test report before promotion, not a threshold in the daily
job. What the job does instead is make each forecast traceable — every NetCDF carries
`species`, `issued_at`, `weather_model`, `checkpoint_sha256` and `git_sha` — and run each
species in its own process, so one failure neither hides nor blocks the rest.
