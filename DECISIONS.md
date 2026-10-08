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

**Entries timed less than 10 min outside their survey move into it; further out, they are
dropped.** The tolerance is a model choice, made here: the source keeps the recorded times
untouched. It reproduces the dataset's rule until 2026-10 (`time_adjusted` / `time_outside_survey`):
an entry before its survey moves to start + 1 min, one at or after its (exclusive) end to end -
1 min. The release keeps such entries at day level ("pending correction"), told apart from untimed
ones only by `remark_processing`; their recorded time comes from the dataset's internal
observation table (`source_count_id` -> `observation_id`), extracted by `build_counts.py` into
`entry_times.csv`. Without it they are dropped: folded into an unsplit survey as untimed they
inflated its rate (56 Black Kites on 2025-08-03) and un-split a timed count (2025-10-26). On the
2026-10 release: 162 entries (4 493 birds) moved in, 64 (2 166) dropped -- the old flags exactly,
less the entries of `partial` surveys and one butterfly.

**Trektellen counts reaching into the night are clipped to civil twilight**, reproducing the
dataset's rule until 2026-10 (the release keeps the recorded times pending correction): a count
starting more than 45 min before civil dawn or ending more than 45 min after civil dusk (sun at
-6 deg, the model's night threshold) starts at dawn / ends at dusk. Counts closed the same evening
end at most ~30 min after dusk; the 7 clipped ones, 4 left open until the next morning, were
closed days later. Without it their night hours become zeros nobody watched.

**The historical day window is a dataset matter.** Historical hourly recording omits hours with
no bird; the workbook's `startTimeDay`/`endTimeDay` (declared attendance) says which hours were
counted. Whether such an hour was counted is source evidence, needed by the explore page's effort
too, so defile-dataset emits the gaps between the declared window and the recorded hours as empty
`complete` surveys (reviewed long gaps that were real breaks are `none`), and this repo has no
historical zero-fill of its own: they become zero rows like any empty survey. 2014-2021 effort
matches the old day-window zero-fill to within a few hours a year, as fewer, longer periods; the
differences are reviewed dataset decisions (2015-08-09 is now a documented rained-off day,
2019-08-20 07:35-20:00 is kept as counted).

**Species are named by the release's English name (AviList first).** Two modelled species
change: Eurasian Kestrel -> Common Kestrel, European Honey-buzzard -> European Honey Buzzard.
Their configs, checkpoint folders, forecast file names and `species_doy_statistics.json` keys
follow; defileViz must follow too. A local eBird-name map was rejected: a second name list kept
in sync by hand.

## Explore

**The Explore export is a separate, raw aggregation of the release** (`src/explore/`,
#55), not the model's counts: no count is moved, dropped or imputed, and daily totals reconcile
with `count.csv` and the dataset's own daily totals (`tests/test_explore.py`). Effort is the union
of `complete` survey intervals per local day. `species_doy_statistics.json` stays the model's
contract, unchanged.

**Effort-adjusted values use a time-of-day profile, not birds per hour.** A taxon's profile
p(h | doy) is fitted with the model's own ratio GAM (`src.phenology.fit_ratio_surface`, one
implementation) on local clock hours, from days timed to the hour; a counted day's coverage `c` is
the share of the profile in the hours counted, the adjusted day is count / c, and the annual index
is Σ birds / Σ c over counted days in the window (a ratio estimator, so low-coverage days do not
dominate). Taxa with fewer than 500 timed birds use a group profile (raptors, pigeons,
passerines, other), then a uniform one. On the 2026-10 release, 66 taxa have their own profile.
`scripts/analyse_explore_effort.py` compares the options for Black Kite, Honey Buzzard, Red Kite,
Common Buzzard, Wood Pigeon and Chaffinch:

- Birds per hour and the uniform (daylight) index are the same curve. Neither is offered.
- The profile matters where the species has a peaked day: before 1993 Common Buzzard is at 0.25
  of its 2010-2025 level per hour and 0.82 with its profile (midday passage, morning counts).
- Profiles from the hourly sheets (2014-2020) and from Trektellen entry times (2021-2026) agree
  (total variation 0.05-0.12; Wood Pigeon 0.22, shaped by a few huge mornings), so Trektellen
  times are usable as passage times. Nothing is timed to the hour before 2014, so stability across
  decades cannot be tested.

**No adjusted value below c = 0.5** (`COVERAGE_MIN`), for a day or for a year's mean: the raw
count and `c` are shown instead. Pre-1993 counts cover a median 0.05-0.11 of a raptor's day, and
the step at 1993 survives the adjustment (Kestrel x14, Honey Buzzard x5; Common Buzzard x2.4 ->
x1.45), so effort is not what separates those years. They stay visible, unadjusted, and out of
reference bands. Taxa not counted systematically in some years (passerines before ~2007) are a
protocol matter no effort metric fixes; the dataset README has that history.

**Explore lives in this repo, contained in `src/explore/`.** It shares the forecast's release
reader and time-of-day GAM, and a population trend fitted for Explore may later help the
forecast (Red Kite's under-prediction of recent years is a missing trend). But the forecast never
imports it (`tests/test_explore.py` enforces the direction); a result crosses over only as a named
file the forecast reads, as `species_doy_statistics.json` does, decided here first. The GAM
profile is variant A of the Explore baseline, provisional until compared with a hierarchical
Gaussian process on a chronological holdout.

**Each taxon has a start year, 1993 or 2007**: its counts are compared from then on, and nothing
is adjusted before it (earlier years stay in the export, raw). Daily systematic counting began in
1993 and always targeted raptors, herons and egrets, storks, pigeons and corvids, and other large
birds counted individually (cranes, geese, ducks), which start in 1993 even if rare then (Peregrine
recovered). Passerines were hardly recorded before 2007, when their taxa double and their birds
rise 15-fold (defile-dataset `docs/sampling-history.md`): any other taxon starts in 1993 only if
recorded in at least 0.75 times as large a share of the 1993-2006 years as of the later ones. A
year counts as recorded only with at least 2% of the taxon's median year since 2007, so that a
trickle noted while the taxon was not counted is not a series ("swallow sp." 2000-2006: ~600 birds
a year against ~150 000 since). The floor stays low because it also penalises a real increase
(Common Crane, ~30 a year before 2007 and 380 since). Full tier: 45 from 1993, 41 from 2007.

**Combined series where names were split differently over the years** (`COMBINED`): all
Columba pigeons (Wood Pigeon, Stock Dove, "Columba sp."; not the local Feral and Rock Pigeons), and
all swallows and martins. "Columba sp." appears only in 2014, when the hourly sheets begin, and has
been a quarter of the pigeons since, so Wood Pigeon's own series drops there for a recording
reason; swallows were "swallow sp." in 1993-1999 and are increasingly identified since 2021. A
combined series is exported like a taxon (rank `combined`, its `members`, its own profile and start
year: pigeons 1993, swallows 2007), beside its members, which stay as they are. Its trend is the
one to read.

**Tiers count days with migrating birds**, not birds: `full` from 50 days over 5 years, `rare`
at 10 days or fewer (84 / 45 / 142). Provisional, to tune once the page exists. French names come
from the eBird taxonomy (fr_FR) by `ebird_code`.

**Trends and annual totals come from a GAM, not a Gaussian process** (`src/explore/trend.py`,
`scripts/benchmark_trend.py`). A day's count is negative binomial around coverage x exp(intercept
\+ trend(year) + season(doy) + shift(year, doy) + year level + episode), where `episode` is each
year's own short-range curve (runs of good or bad migration days, the weather's share); hours not
counted are filled with an hourly over-dispersion `kappa` fitted from the timed days (flocks). The
annual total is the birds counted plus the posterior predictive of the coverage missed. Both
priors were fitted on one engine and benchmarked on Black Kite, Honey Buzzard, Red Kite, Common
Buzzard, Osprey, all pigeons and Chaffinch:

- Gap filling (well-counted years 2014-2025 given an old year's gaps): GAM and GP both 5.0%
  error, the GP better in 49% of paired trials; the ratio index 14.6% (40% too high for pigeons),
  the season alone 5.7%.
- Calibration came from the episode term and `kappa`, not from the prior: without them both 80%
  intervals held the truth 54% of the time, with them 80% (95%: 94%).
- Predicting 2023-2025 from the years before (extrapolation, which Explore never does): GP 35%
  error, GAM 49%, almost all of it Chaffinch, where the P-spline extends its last slope; daily log
  scores equal (-3.170 / -3.171). A trend fed to the forecast would need a flat extrapolation.
- The Hilbert-space GP misbehaved at long lengthscales (prior variance 0.17 instead of 1 at the
  series' ends for a 30-year lengthscale, with the domain at 1.5 x the data), and full Bayes
  (NumPyro NUTS: 3-5 min a fit; PyMC's C backend does not compile here) bought nothing once
  intervals were calibrated. The GAM refits in 0.2 s.

Every smooth sums to zero over its grid, so the intercept, trend, season, `shift` (an interaction
only, as mgcv's `ti`) and the year level each own their part: unconstrained, the smooth trend moved
with the optimiser's stopping point (Black Kite 2025/1993: 0.80 to 1.21). Smoothing parameters
come from the Fellner-Schall update (mgcv's `efs`): the same optimum from any start, 2-6 s a taxon.
On the faster benchmark (three refits per taxon, each hiding every target year; 2 min for seven
taxa) the GAM fills gaps with 5.6% error and 82% / 95% coverage, Honey Buzzard the exception (55%
/ 82%). Trends are exported for full-tier species and combined series, not for unidentified birds
("falcon sp."), whose numbers follow identification effort.

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
