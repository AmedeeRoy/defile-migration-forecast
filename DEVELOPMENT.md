# Défilé migration forecast — state of the project and development roadmap

Status: draft, August 2026. Written after a full read of `src/`, `configs/`, `scripts/`,
`.github/workflows/`, and the contents of `data/`. Line references are to the code as it
stands today. Nothing in this document has been changed in the code yet.

## 1. What we are building

An autonomous system that, every morning without human intervention, publishes a
forecast of the **hourly passage rate of migrating raptors at Défilé de l'Écluse**, per
species, for today and the next few days. The forecast is consumed by the
[defileViz](https://github.com/Rafnuss/defileViz) web app.

Everything below should be read against that goal. In particular it means three things
that are easy to lose sight of when iterating on model architecture:

The unit of prediction is an **hourly rate**, not a daily total. The model outputs 24
values per day and the app draws the diurnal curve, so the shape of the day matters as
much as the daily sum.

The system must run **unattended**. A model that needs a human to check its output is not
finished. Failure modes that silently publish a wrong or stale forecast are as serious as
crashes, and are currently not detected at all.

The features available at 03:00 UTC on the day of the run are the only features that
exist. Anything the model learns to depend on that is not available at that moment is a
liability, not a feature.

## 2. How the system works today

The training path reads counts from `data/count/all_count_processed.csv` (one row per
species per survey period, 1966–2025, with `start`/`end` timestamps in UTC) and weather
from per-location CSVs in `data/era5/` that were exported from Google Earth Engine
(`data/era5/gee_code.js`). `DefileDataModule` (`src/data/defile_datamodule.py`) converts
counts to an hourly rate by dividing by survey duration, builds a 24-hour `mask` giving
the fraction of each hour covered by the survey, assembles three weather tensors (a
"main" hourly stack at Défilé including sun position, an hourly stack at five nearby
locations, and a lagged daily stack at seven regional locations), normalises everything
via `DataTransformer`, and splits years into train/val/test.

The model (`src/models/components/unet.py`, `UNetplus`) is a 1-D U-Net over the 24-hour
axis for the hourly branch, plus a separate 1-D conv stack over the lag axis for the
daily branch. The two branches are combined multiplicatively — `out = 8 * out_h * out_d`
where both are sigmoid-bounded — so the hourly branch learns the shape of the day and the
daily branch learns the daily magnitude. The output is in `log1p(birds/hour)` space,
capped at 8 (i.e. about 2 979 birds/hour). Hours before 05 UTC and from 19 UTC are forced
to zero by a hard-coded mask. Loss is a sum of a Tweedie term and a "probabilistic RMSE"
term (`src/models/criterion.py`), with per-species weights tuned by Optuna and stored in
`configs/experiment/*.yaml`.

The forecast path (`ForecastDataset` in the same datamodule, plus
`src/data/open_meteo.py`) does *not* read the CSVs. It calls the Open-Meteo forecast API,
converts each Open-Meteo variable into its ERA5 equivalent through `CONVERSION_DICT`,
applies the normalisation parameters pickled during training
(`data/transform_data.pickle`, which is committed to git), and runs the checkpoint stored
in `prod/models/<Species_Name>/checkpoints/best.ckpt`.

Operationally, `.github/workflows/predict_and_deploy_forecasts.yml` runs daily at 03:00
UTC, recreates the conda environment, runs `src/predict.py --multirun` over all eleven
species, and SCPs the resulting NetCDF files to a GCE host. defileViz fetches those files
directly from `https://defile.raphaelnussbaumer.com/forecasts/...`; it does not read this
repo's GitHub Pages site.

## 3. Honest assessment

The scaffolding is good. Lightning + Hydra gives clean per-species experiment configs,
the Optuna sweep is wired up, checkpoint promotion is scripted, and the daily job plus
the front end genuinely work end to end. The domain modelling is also thoughtful in
places that matter: dividing by survey duration to get a rate, the fractional-hour mask,
the multiplicative shape × magnitude decomposition, and the choice of a Tweedie loss for
zero-inflated counts are all the right instincts.

What is weak is the layer in between: the data contract between training and inference.
That is where most of the defects below live, and it is not a coincidence — it is the part
of the system with no tests, two independent implementations (CSV/GEE for training,
Open-Meteo for serving), and no automated comparison between them. Several features are
currently either constant, mis-scaled by a factor of 3.6 or 24, or geometrically
scrambled at forecast time. Until that is fixed, model architecture work is measuring
noise.

There are also no tests of any kind in the repository, and no monitoring of the daily job.

## 4. Defects found

Ordered by how much they cost us. Each has a file reference and a suggested fix. I have
not verified these by running the code — the environment is not installed here — so treat
each as "read carefully and confirm", though most are unambiguous from the source.

### P0 — actively corrupting the model

**4.1 `log_transform` never applies a logarithm, and reads back the wrong parameters.**
`src/data/data_transformer.py`. `compute_param` stores `(mean, std)` of the log-data
(lines 38–43) but `apply` unpacks that tuple as `(data_min, data_max)` and computes
`(data - data_min) / (data_max - data_min)` with no log at all (lines 55–57). This
transform is assigned to `total_precipitation` and `instantaneous_10m_wind_gust`. Because
most precipitation values are zero and get clipped to 1e-9, `mean(log) ≈ -19` and
`std(log) ≈ 3`, so the applied transform is roughly `(x + 19) / 22` for x in
[0, 0.02] m — i.e. every value lands within 0.0004 of 0.864. **Precipitation and wind
gusts are, in effect, constant inputs. The model cannot see rain.** For soaring raptors
this is likely the single most damaging defect in the codebase. Fix: apply
`(log(clip(x)) - mean) / std`, and name the stored params so this cannot recur.

**4.2 The default data config cannot be loaded.** `configs/data/defile.yaml` lists
`total_precipitation` in `era5_daily_variables`, and `era5_daily_locations` includes
Munich, Stuttgart, Frankfurt and Berlin. Those four CSVs come from
`ECMWF/ERA5_LAND/DAILY_AGGR` and their only precipitation column is
`total_precipitation_sum`. The assertion in `get_era5_hourly`
(`src/data/get_era5.py:103–106`) therefore fires on Munich. `python src/train.py` with
the committed default config should crash. Note that `configs/data/defile-small.yaml`
comments those locations and variables out, which is consistent with recent work having
been done on the small config only. Fix: decide whether the far-field daily locations are
wanted, and if so give them their own variable list and rename the column at read time.

**4.3 Wind speeds are 3.6× too large at forecast time.** Open-Meteo's default
`wind_speed_unit` is km/h (confirmed in their docs), and no unit is requested in the API
call (`src/data/open_meteo.py:130–140`). ERA5 `u/v_component_of_wind_*` and
`instantaneous_10m_wind_gust` are m/s. So `wind_speed_10m`, `wind_speed_100m` and
`wind_gusts_10m` all arrive in km/h and are fed through conversions that assume m/s. All
five wind features are inflated by 3.6 relative to what the model was trained on, which
after min-max normalisation puts them far outside [0, 1]. Fix: pass
`"wind_speed_unit": "ms"` in the request params.

**4.4 The wind direction → u/v conversion is geometrically wrong.**
`src/data/open_meteo.py:35–54` computes `u = V·cos(θ)` and `v = V·sin(θ)`. Meteorological
wind direction is the direction the wind blows *from*, measured clockwise from north, so
the correct conversion is `u = -V·sin(θ)` and `v = -V·cos(θ)`. As written, the code
returns `u_code = -v_true` and `v_code = -u_true`: the components are swapped and
sign-flipped, a reflection of the wind vector about the north-east axis. Wind direction is
one of the strongest drivers of raptor passage, and this affects only the forecast path,
so the model is being served a systematically wrong wind field at exactly the moment it
matters. Fix the two formulas (all four u/v entries) and add a unit test with a known
case, e.g. a 10 m/s wind from due north (θ=0) must give u=0, v=-10.

**4.5 Daily aggregation means three different things.** In training,
`get_era5_daily` (`src/data/get_era5.py:141–169`) calls the hourly reader and takes
`mean(dim="time")`. For Défilé, Schaffhausen and Basel the source is hourly ERA5, so this
is a true 24-hour mean. For Munich, Stuttgart, Frankfurt and Berlin the source is already
a daily aggregate stamped at 00:00, so after the union of time coordinates those
locations are NaN for 23 hours and the skipna mean returns the single daily value — and
for precipitation that value is a daily *sum*, not a mean rate, differing by roughly a
factor of 24. At forecast time, `download_forecast_daily` fetches hourly data for all
seven locations and takes a genuine 24-hour mean. So four of the seven daily locations
have different semantics in training than in inference, on top of the sum-versus-mean
problem. Fix: make daily aggregation explicit and identical on both paths, per variable
(mean for state variables, sum for accumulations).

### P1 — silently wrong results and operational hazards

**4.6 The test set is re-drawn after training.** `setup()`
(`src/data/defile_datamodule.py:389`) runs its full split logic for `stage` in
`{fit, validate, test}`, and the year shuffle at line 447 has its seed line commented
out. Lightning calls `setup("test")` again before `trainer.test()`, by which point the
global numpy RNG state has advanced, so the test years differ from the years held out
during fit. Reported test metrics are therefore computed partly on years the model
trained on. Fix: compute the split once (guard it like `self.read_era5`), and seed it
explicitly from `cfg.seed` rather than relying on global RNG state.

**4.7 `ProbaRMSE` normalises by 24 instead of by the survey length.**
`src/models/criterion.py:163–166` computes `mean(y_pred[:,0,:] * mask, dim=1)`, i.e.
`sum(pred·mask)/24`, and compares it to `log1p(observed hourly rate)`. The correct
denominator is `sum(mask)`. As written the implied target is scaled by
`survey_hours / 24`, which varies row by row — a three-hour survey and a twelve-hour
survey imply targets that differ by a factor of four for the same true rate. Fix: divide
by `mask.sum(dim=1)`. Note the Tweedie term does this correctly via `applyMask`, so the
two loss terms currently disagree about what quantity they are fitting.

**4.8 The uncertainty output channel is untrained.** `configs/model/unet.yaml` sets
`nb_output_features: 2`, and `ProbaRMSE.alpha` is 1 in the base config and in every
`configs/experiment/*.yaml`. With `alpha = 1` the loss is `1·rmse_term + 0·nll_term`, and
only `nll_term` touches channel 1 — so channel 1 receives no gradient and remains
essentially random. `save_test` nonetheless exports `pred_count_up` / `pred_count_down`
from it. The `hparams_search` sweep has `model.criterion.rmse.alpha` commented out, so
this is true for all tuned models. Fix: either drop to one output channel, or let alpha
be tuned in (0, 1) and re-tune. Note that the app's uncertainty bands come from
climatological quantiles in `species_doy_statistics.json`, not from the model, so nothing
user-facing is currently wrong — but the exported bands should not be trusted.

**4.9 Normalisation statistics are fitted on train + val + test.** `DataTransformer` is
built from the full filtered ERA5 arrays (`defile_datamodule.py:476–483`) before the
split is used. Mild leakage, and min/max normalisation is outlier-sensitive on top of it.
Fix: fit on training years only.

**4.10 Checkpoint promotion can write where prediction never reads.**
`configs/hydra/default.yaml` sets `run.dir` using the raw species name (with spaces)
while `sweep.subdir` uses `${underscore:...}`. `move_checkpoints_to_prod.py` copies using
the run subdirectory name verbatim, and `configs/predict.yaml` reads
`prod/models/${underscore:'${data.species}'}/...`. So promoting after a *single-species*
run lands in `prod/models/Black Kite/` while prediction reads `prod/models/Black_Kite/`.
Both directory spellings exist on disk today, which is the fingerprint of this bug. Fix:
use `${underscore:...}` in `run.dir` too, and delete the space-named directories.

**4.11 There is no season guard.** The cron runs every day of the year, but training is
restricted to `doy` 196–335 (mid-July to end of November) by `configs/data/defile.yaml`.
For roughly seven months a year the job publishes confident extrapolations from a model
that has never seen that part of the year. Fix: skip or clearly flag out-of-season runs,
and have the app render them as unavailable rather than as a forecast.

**4.12 `era5_main.squeeze()` will drop the batch dimension.**
`src/models/components/unet.py:209`. With a single-location main stack the intent is to
drop the trailing location axis, but a batch of size 1 — which happens whenever the last
training batch has one sample — also loses the batch axis and corrupts the subsequent
`cat`. Fix: `squeeze(-1)`.

**4.13 `predict_step` computes a meaningless loss and divides by zero.**
`src/models/defile_module.py:440` calls `model_step`, which evaluates the criterion
against the dummy `count` and length-1 `mask` that `ForecastDataset` supplies
(`defile_datamodule.py:206`). `applyMask` then divides by `mask.sum() = 0`, producing NaN,
and a whole forward pass is wasted. Fix: drop the `model_step` call from `predict_step`.

### P2 — efficiency

**4.14 Every validation, test and predict step runs the network twice.** `model_step`
performs a forward pass and returns only the loss, then each step method calls
`self.forward(...)` again on the same batch (`defile_module.py:164/172`, `241/252`,
`440/444`). Returning `(loss, count_pred)` from `model_step` roughly halves
validation and test time.

**4.15 Per-sample xarray label lookups dominate data loading.**
`DefileDataset.__getitem__` (`defile_datamodule.py:69–94`) performs three `.sel(date=...)`
label-based selections per item. With `batch_size: 1024` and `num_workers: 0` this is the
main training bottleneck; xarray `.sel` overhead is on the order of hundreds of
microseconds, so a single epoch spends most of its time in index lookups. Fix: after
`setup`, materialise each stack once into a contiguous `float32` numpy array ordered to
match `count`, and index it positionally. This is likely the largest single speed win
available and should be done before any hyperparameter sweeping.

**4.16 ERA5 CSVs are parsed repeatedly and densified before filtering.** `setup` calls
`get_era5_hourly` three times, and locations overlap between the three stacks, so
`Basel.csv` (136 MB) is parsed three times and `Defile.csv` twice — roughly 1 GB of CSV
per run. The full 1966–2025 range is then unstacked into a dense
`(date × time × location)` array *before* `filter_by_date` narrows it to
`doy ∈ [196, 335]`. Fix: convert the CSVs once to a single netCDF/Zarr store keyed by
location, read only the requested years and days, and memoise per-location reads within a
run.

**4.17 Saliency is computed on every test batch and kept in memory.**
`defile_module.py:219–257` builds a `Saliency` attribution for each batch and accumulates
it, which also forces `inference_mode: False` globally in `configs/trainer/default.yaml`.
Fix: gate this behind a config flag and compute it on a subsample.

**4.18 Small things.** `pred_mask` is rebuilt as a numpy array and moved to device on
every forward pass (`unet.py:239–243`) — make it a `register_buffer`. The criterion
dataclasses annotate fields as `alpha: 1`, using the value as a type annotation so there
is no default (`criterion.py:69–70`, 135, 219, 254) — should be `alpha: float = 1.0`.
`plt_predict` hard-codes a 2×3 subplot grid and breaks if `forecast_day != 5`
(`src/plots/save_predict.py:8`). `configs/data/defile.yaml` sets
`years: range(1966, 2024)`, silently excluding the 2024 and 2025 seasons that are already
in the CSVs. `src/export/` contains only a stale `__pycache__` — the source module is
gone, and the README still documents `src/export` and a `configs/export` that does not
exist.

## 5. The four open questions

These are the questions that will actually decide whether the forecast is good. The
defects above are prerequisites: none of these questions can be answered while
precipitation is a constant and the forecast wind field is scrambled.

### 5.1 Which years should we train on?

`data/count/readme.md` is unusually candid about the survey history, and it says the
protocol changed repeatedly: coverage before 1993 was sporadic and pigeon-focused (with
1983 and 1992 as exceptions); daily volunteer monitoring from 1993; a salaried observer on
weekdays from 2008 to 2016; two salaried observers Monday to Saturday from 2017. Recording
granularity changed too — daily totals up to 2013, hourly paper forms 2014–2016,
Naturalist entry 2017–2020, Trektellen from 2021. Observer numbers were never digitised,
so true effort is unavailable.

Two things follow. First, the *current* configuration trains on 1966–2023 with
`year_used: "constant"`, which sets the year feature to a constant zero — so the model is
given no way at all to distinguish a 1970 pigeon-focused half-day from a 2019 two-observer
full day. Second, there is an implicit and undocumented reweighting already in effect: a
date recorded as twelve hourly slots contributes twelve rows sharing the same weather,
while a date recorded as a daily total contributes one. The hourly-resolution era
therefore already dominates the loss, which is probably desirable but is happening by
accident rather than by choice.

What I would do, in order. Run a **year-subset ladder** as a first-class experiment —
1966+, 1993+, 2008+, 2014+, 2017+ — holding everything else fixed, and compare on a
*chronological* holdout rather than the current random-years split. This is cheap, it is
the single most informative experiment available, and it directly answers the question
rather than guessing at it. Add `train_val_test: "chronological"` to make that comparison
honest: for an operational forecast the only evaluation that matters is "trained on the
past, tested on the following season", and the present random-year split systematically
flatters the model by letting it interpolate across eras.

On year-to-year variability, the more interesting structural idea is to stop asking one
model to predict absolute counts. Passage at a site is roughly (annual population
passing) × (how the season is distributed) × (how a given day's weather redistributes
birds within that). Predicting the *share* of the season's total that falls on a given day
and hour, and forecasting the annual total separately (from population trends, or simply
carried over from the previous years and updated as the season progresses via Trektellen),
would remove most of the year-to-year variance from the hard part of the problem. This is
a bigger change and should come after the ladder experiment, but it is where I would place
the biggest bet on model improvement.

On the trend, `year_used: "period"` (three coarse era buckets) is available and untested in
the committed configs; a smooth year term or a per-year offset would be better than either
constant or three buckets. Worth including in the ladder sweep.

### 5.2 How do we keep training features and forecast features comparable?

Right now we do not, and the mechanism is structural: there are two independent
implementations of "get the weather", one reading GEE ERA5 CSVs and one calling
Open-Meteo, and nothing compares them. Defects 4.3, 4.4 and 4.5 are all instances of the
same failure. Fixing them individually is necessary but will not prevent the next one.

The strategic fix is to **use one provider for both paths**. Open-Meteo serves an ERA5
archive (`archive-api.open-meteo.com`) with the same variable names, units and API shape
as the forecast endpoint. Building the training features through the same code path as the
forecast features eliminates the entire class of unit, naming, convention and aggregation
mismatches at once, and removes the GEE export step from the loop. The cost is refetching
history and revalidating against the current CSVs; the benefit is that the train/serve
contract becomes true by construction rather than by careful maintenance of
`CONVERSION_DICT`.

There is a deeper version of the same question that is worth being explicit about. Even
with identical variables and units, **training on reanalysis and predicting from a
forecast is a distribution shift**. ERA5 is a hindcast that assimilates observations; a
five-day forecast is not, and its error grows with lead time. A model trained only on
reanalysis has never encountered forecast error and cannot learn to hedge against it,
which typically shows up as forecasts that are too confident and too sharp at long leads.
Open-Meteo's Historical Forecast API serves *past forecasts* at known lead times, which
makes the principled fix available: train (or at least fine-tune and evaluate) on
forecast-derived features at matching lead times, and report skill separately for day+0
through day+5. Even before doing that, simply *measuring* skill by lead day would tell us
how far ahead the forecast is worth publishing — something we currently do not know.

Regardless of provider decisions, we should add a **train/serve parity check** to CI: for
a handful of dates in the recent past, build the feature tensors through both paths and
assert per-variable agreement within a tolerance. That test would have caught 4.3, 4.4 and
4.5 immediately, and it is the durable answer to "how do we ensure the data is similar".

### 5.3 Can we use recent Trektellen counts as model inputs?

Yes, and the plumbing largely exists already: there is a working Trektellen proxy on the
GCE host that defileViz calls at
`https://defile.raphaelnussbaumer.com/trektellen/{siteId}/{yyyymmdd}` (site 2422 is
Défilé), and `data/Trektellen_raptor_2015_2024/` already holds a 47 MB export covering
many north-west European sites plus site metadata. So both the historical feature for
training and the live feature for inference are obtainable.

Three design points matter more than the implementation.

**Missing days must be represented, not filled.** Zero is a meaningful value in this
dataset — it means "watched and saw nothing" — so zero-filling an unobserved day teaches
the model that absent data implies absent birds, which is exactly backwards and will bite
hardest on precisely the days when observers stayed home because the weather was bad. The
standard remedy is to pass two channels per lagged day: an imputed value (climatological
day-of-year mean for that species is a reasonable default) and a binary observed flag. The
model can then learn to discount the imputed value. This also means the feature degrades
gracefully when the API is down, which matters for an unattended job.

**Availability depends on lead time.** Yesterday's count at Défilé is a strong predictor
for today and tomorrow, and progressively useless for day+3 to day+5. Feeding it as a
single feature to a model that predicts all six days at once will let the model lean on it
for leads where it is not informative. Either train separate heads per lead day, or make
the lag features explicitly lead-aware, or accept that the near-term forecast improves and
the far-term does not.

**The bigger prize is upstream sites, not autoregression on Défilé.** Counts from sites to
the north and east give genuine spatial early warning of a wave in transit, which is
information the weather fields only carry indirectly. The NW-Europe export is already on
disk. I would scope the first experiment as: pick a small set of candidate upstream sites,
build lagged count features with observed-flags for both Défilé and those sites, and
measure the gain on the chronological holdout. This is the highest-upside modelling change
on the list and it is largely independent of the U-Net architecture.

One caution: adding a feature that depends on a third-party API to the *forecast* path
adds a new operational failure mode. It needs a defined fallback (use the observed-flag
channel and carry on) rather than an exception that kills the daily run.

### 5.4 What does "runs itself every morning" actually require?

The job runs, but nothing watches it. If `predict.py` fails, the previous day's NetCDF
files stay on the GCE host and the app keeps serving them with no indication that they are
stale — a silent failure that could persist for weeks. The minimum is a freshness check
(assert the published file's date matches today), a notification on workflow failure, and
a visible "last updated" in the app.

Beyond monitoring, three fragilities are worth addressing. The workflow recreates the
conda environment on every run with `conda env create || conda env update`, which is slow
and can drift; a lockfile or a prebuilt container image would make the daily run
reproducible. `data/transform_data.pickle` is a single global file that must correspond to
the promoted checkpoints — it is committed, so it is at least versioned, but nothing
enforces the correspondence, and retraining silently rewrites it. Saving the transform
parameters *inside* the checkpoint would remove the coupling entirely. And the season
guard from 4.11 belongs here too.

## 6. Suggested sequencing

**First, make the pipeline honest.** Fix 4.1 (log transform), 4.2 (config crash), 4.3 and
4.4 (wind units and convention), 4.5 (daily aggregation), 4.6 (test split), 4.7
(ProbaRMSE denominator). Add the train/serve parity check and a smoke test that runs
`train.py debug=default` end to end. Nothing about model quality can be assessed before
this is done, and the previously tuned hyperparameters in `configs/experiment/*.yaml`
should be considered void afterwards, since they were tuned against corrupted features.

**Second, make it fast enough to experiment.** Fix 4.14 (double forward) and 4.15
(per-sample xarray lookups), and cache the ERA5 reads (4.16). The point is to make the
year-subset ladder and the Optuna sweeps cheap enough to run often.

**Third, answer 5.1.** Chronological split, year-subset ladder, year-representation
comparison. Re-tune hyperparameters on the winning configuration. Report skill by lead
day.

**Fourth, decide on 5.2.** Whether to unify on Open-Meteo for both paths, and whether to
train on historical forecasts rather than reanalysis.

**Fifth, build 5.3.** Lagged Trektellen counts with observed-flags, Défilé first, then
upstream sites.

**Throughout, close out 5.4.** Freshness monitoring, failure alerting, season guard,
reproducible environment, transform-in-checkpoint.

## 7. Things to verify that I could not confirm here

The environment is not installed in this session, so the following are reasoned from
source rather than observed, and should be checked before acting:

Whether `python src/train.py` with the committed `data=defile` config does in fact raise
the assertion described in 4.2, or whether some local state avoids it.

The exact accumulation semantics of `total_precipitation` and
`surface_solar_radiation_downwards` in the `ECMWF/ERA5/HOURLY` GEE collection used for the
export. The Open-Meteo side of the radiation conversion (W/m² × 3600 → J/m²) looks
arithmetically right, though the inline comment states the conversion backwards.

The count timezone handling looks **correct** — `notebooks/processing_count_data.ipynb`
uses `tz_localize("Europe/Paris", ...)` followed by `tz_convert`, and the distribution of
stored survey start hours peaks at 06–07 UTC (08–09 local), which is consistent with
post-sunrise starts. Worth confirming once explicitly, since a silent one- or two-hour
shift between the count mask and the ERA5 time axis would distort the whole diurnal curve
and would be nearly invisible in aggregate metrics.

Whether the `prod/models/<species with spaces>/` directories are genuinely orphaned
(4.10), and which checkpoint each currently-deployed forecast actually came from.
