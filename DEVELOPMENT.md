# Défilé migration forecast — state of the project and development roadmap

Status: draft, August 2026. Written after a full read of `src/`, `configs/`, `scripts/`,
`.github/workflows/`, and the contents of `data/`. Line references are to the code as it
stands today.

**Update, 5 August 2026:** fixes for defects 4.1–4.4, 4.6, 4.7, 4.10, 4.12, 4.13, 4.14, and
part of 4.18 have merged as pull requests #26, #29–#35 against this repo. Separately, the
provider unification from section 5.2 is implemented on branch `centralize-weather`
(rebased onto `main` post-merge), which supersedes PRs #27 and #28 — both left open,
unmerged — see the status note in section 4 and the decision record in 5.2.

None of the checkpoints in `prod/models/` have been retrained against any of this yet, so
the "until that is fixed" caveats in section 3 and the sequencing in section 6 still apply
until `centralize-weather` also merges and retrain happens. Per-defect status is noted
inline in section 4.

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
from the local Parquet cache in `data/weather/`, built from the Open-Meteo ERA5 archive by
`scripts/build_weather_cache.py`. (Until August 2026 the weather came from per-location CSV
exports from Google Earth Engine in `data/era5/`; see 5.2.) `DefileDataModule` (`src/data/defile_datamodule.py`) converts
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

The forecast path (`ForecastDataset` in the same datamodule) does *not* read the cache. It
calls the Open-Meteo forecast API through the same `src/data/weather.py` entry point, so it
shares the training path's variable names, unit conversions and daily aggregation. It
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

*Update, August 2026:* the two implementations are now one (5.2), and the weather layer has
tests. The rest of the codebase still has none, and the daily job still has no monitoring
(5.4). Model quality remains unassessable until `centralize-weather` also merges and the
models are retrained against every fix landed so far.

## 4. Defects found

Ordered by how much they cost us. Each has a file reference and a suggested fix. I have
not verified these by running the code — the environment is not installed here — so treat
each as "read carefully and confirm", though most are unambiguous from the source.

> **Status, August 2026.** The weather-provider migration described in 5.2 has been
> implemented on branch `centralize-weather`. Both paths now go through
> `src/data/weather.py`, the ERA5 CSV exports and `src/data/get_era5.py` /
> `src/data/open_meteo.py` are gone, and `tests/test_weather.py` covers the conventions and
> the train/serve contract.
>
> That branch closes **4.2, 4.5, 4.16** and the `years` half of **4.18**, and it supersedes
> two PRs that stayed unmerged: **#27** (it patched `open_meteo.py`, which no longer exists —
> the wind fixes live on in `weather.py`, now with tests) and **#28** (it dropped the
> far-field daily locations to avoid 4.5; with 4.5 fixed properly they are back). It also
> turned up four defects the audit had missed, recorded in section 4.19 — including 4.19d, a
> resolution mismatch in the wind field that provider unification does *not* fix and that
> may matter more than anything else on this list.
>
> The other defects were addressed by PRs #26, #29–#35, all now merged to `main`, and
> annotated inline below. Still unowned: **4.9** and the remainder of **4.18**. The
> sequencing in section 6 is otherwise unchanged, and the tuned hyperparameters in
> `configs/experiment/*.yaml` remain void — they were tuned against features that were both
> corrupted and, per 4.19, systematically offset from what production serves.

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

**Fixed in [PR #26](https://github.com/AmedeeRoy/defile-migration-forecast/pull/26)**
(`fix/log-transform`, merged). The stored `transform_data.pickle` parameters
remain valid; the checkpoints in `prod/models/` were trained against the broken transform
and must be retrained before this is deployed.

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

*Resolved on `centralize-weather`, by dissolution.* Open-Meteo serves hourly data for
every location from one dataset, so there is no longer a class of location that only has
daily aggregates. Munich, Stuttgart, Frankfurt and Berlin are back in
`era5_daily_locations`, and the assertion that used to fire is gone along with the CSV
reader. This supersedes PR #28, which avoided the problem by dropping those locations.

**Fixed in [PR #28](https://github.com/AmedeeRoy/defile-migration-forecast/pull/28)**
(`fix/daily-locations-consistency`, open, not merged): removes Munich, Stuttgart,
Frankfurt and Berlin from `era5_daily_locations` rather than giving them a separate
variable list, so the config loads again. This also changes `nb_input_features_daily`
from 37 to 17 and requires retraining. See 4.5 below — the same PR addresses part of that
defect too.

**4.3 Wind speeds are 3.6× too large at forecast time.** Open-Meteo's default
`wind_speed_unit` is km/h (confirmed in their docs), and no unit is requested in the API
call (`src/data/open_meteo.py:130–140`). ERA5 `u/v_component_of_wind_*` and
`instantaneous_10m_wind_gust` are m/s. So `wind_speed_10m`, `wind_speed_100m` and
`wind_gusts_10m` all arrive in km/h and are fed through conversions that assume m/s. All
five wind features are inflated by 3.6 relative to what the model was trained on, which
after min-max normalisation puts them far outside [0, 1]. Fix: pass
`"wind_speed_unit": "ms"` in the request params.

**Fixed, together with 4.4, in [PR #27](https://github.com/AmedeeRoy/defile-migration-forecast/pull/27)**
(`fix/openmeteo-wind`, open, not merged).

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

**Fixed in [PR #27](https://github.com/AmedeeRoy/defile-migration-forecast/pull/27)**
(`fix/openmeteo-wind`, open, not merged), together with 4.3. Verified against all four
cardinal directions; a 10 m/s wind from due north now gives u=0, v=-10 as expected.

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

*Resolved on `centralize-weather`.* `DAILY_AGGREGATION` in `src/data/weather.py` states
the rule per variable — `total_precipitation` and `surface_solar_radiation_downwards` sum,
everything else averages — and `to_daily` is called by both the training and the forecast
path, so the two cannot disagree.
`test_accumulations_sum_and_state_variables_average` guards it. Gust is averaged; a daily
maximum is arguably the better summary, but that is a modelling change rather than a bug
fix, so it was left alone deliberately.

**Partially addressed in [PR #28](https://github.com/AmedeeRoy/defile-migration-forecast/pull/28)**
(`fix/daily-locations-consistency`, open, not merged): drops the four far-field locations
whose daily CSVs had mismatched semantics (Munich, Stuttgart, Frankfurt, Berlin), so every
remaining daily location is hourly-ERA5-backed and the 24-hour mean means the same thing
everywhere. This sidesteps the defect rather than fixing the general case — if far-field
locations are ever re-added, they still need their own variable list and an explicit
per-variable aggregation rule, as described above.

### P1 — silently wrong results and operational hazards

**4.6 The test set is re-drawn after training.** `setup()`
(`src/data/defile_datamodule.py:389`) runs its full split logic for `stage` in
`{fit, validate, test}`, and the year shuffle at line 447 has its seed line commented
out. Lightning calls `setup("test")` again before `trainer.test()`, by which point the
global numpy RNG state has advanced, so the test years differ from the years held out
during fit. Reported test metrics are therefore computed partly on years the model
trained on. Fix: compute the split once (guard it like `self.read_era5`), and seed it
explicitly from `cfg.seed` rather than relying on global RNG state.

**Fixed in [PR #29](https://github.com/AmedeeRoy/defile-migration-forecast/pull/29)**
(`fix/test-split-reproducibility`, merged): draws the split from a generator
seeded by a new `split_seed` datamodule parameter instead of the global RNG, so fit and
test agree regardless of how much randomness training consumed. 4.9 (normalisation
statistics fitted on all years) is explicitly called out in that PR as a follow-up not
included.

**4.7 `ProbaRMSE` normalises by 24 instead of by the survey length.**
`src/models/criterion.py:163–166` computes `mean(y_pred[:,0,:] * mask, dim=1)`, i.e.
`sum(pred·mask)/24`, and compares it to `log1p(observed hourly rate)`. The correct
denominator is `sum(mask)`. As written the implied target is scaled by
`survey_hours / 24`, which varies row by row — a three-hour survey and a twelve-hour
survey imply targets that differ by a factor of four for the same true rate. Fix: divide
by `mask.sum(dim=1)`. Note the Tweedie term does this correctly via `applyMask`, so the
two loss terms currently disagree about what quantity they are fitting.

**Fixed in [PR #30](https://github.com/AmedeeRoy/defile-migration-forecast/pull/30)**
(`fix/probarmse-denominator`, merged): uses a clamped `mask.sum(dim=1)` for both
the mean and std channel. The per-species criterion weights in `configs/experiment/*.yaml`
were tuned against the old, inconsistent denominator and should be re-tuned.

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

**Fixed for new runs in [PR #33](https://github.com/AmedeeRoy/defile-migration-forecast/pull/33)**
(`fix/checkpoint-promotion-path`, merged): aligns `run.dir` with `sweep.subdir`
via the `underscore` resolver, and registers that resolver in `src/eval.py` too (it was
missing there, so `python src/eval.py --multirun` failed outright). The orphaned
space-named directories already on disk under `prod/models/` still need manual deletion —
this PR only stops new ones being created.

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

**Fixed in [PR #32](https://github.com/AmedeeRoy/defile-migration-forecast/pull/32)**
(`fix/unet-squeeze-and-mask-buffer`, merged). Numerically identical output on
batches larger than 1; no retraining needed.

**4.13 `predict_step` computes a meaningless loss and divides by zero.**
`src/models/defile_module.py:440` calls `model_step`, which evaluates the criterion
against the dummy `count` and length-1 `mask` that `ForecastDataset` supplies
(`defile_datamodule.py:206`). `applyMask` then divides by `mask.sum() = 0`, producing NaN,
and a whole forward pass is wasted. Fix: drop the `model_step` call from `predict_step`.

**Fixed, together with 4.14, in [PR #31](https://github.com/AmedeeRoy/defile-migration-forecast/pull/31)**
(`fix/predict-step-nan-and-double-forward`, merged): `predict_step` no longer
calls `model_step`.

### P2 — efficiency

**4.14 Every validation, test and predict step runs the network twice.** `model_step`
performs a forward pass and returns only the loss, then each step method calls
`self.forward(...)` again on the same batch (`defile_module.py:164/172`, `241/252`,
`440/444`). Returning `(loss, count_pred)` from `model_step` roughly halves
validation and test time.

**Fixed, together with 4.13, in [PR #31](https://github.com/AmedeeRoy/defile-migration-forecast/pull/31)**
(`fix/predict-step-nan-and-double-forward`, merged): `model_step` now returns
`(loss, count_pred)` and `validation_step`/`test_step`/`predict_step` reuse it instead of
calling `self.forward(...)` again. No change to reported numbers — the second forward pass
was computing the same thing (dropout/batch-norm are in eval mode during val/test).

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

*Resolved on `centralize-weather`.* The CSVs are replaced by a Parquet store partitioned
by location (`data/weather/`), built once by `scripts/build_weather_cache.py`. `load_cache`
pushes the location, year and day-of-year filters down into the read, so unrequested
locations are never opened and the dense `(date, time, location)` array is built only from
rows that survive filtering. Reading one location-season now takes about 2 s rather than
parsing roughly 1 GB of CSV per run.

**4.17 Saliency is computed on every test batch and kept in memory.**
`defile_module.py:219–257` builds a `Saliency` attribution for each batch and accumulates
it, which also forces `inference_mode: False` globally in `configs/trainer/default.yaml`.
Fix: gate this behind a config flag and compute it on a subsample.

**4.18 Small things.** `pred_mask` is rebuilt as a numpy array and moved to device on
every forward pass (`unet.py:239–243`) — make it a `register_buffer`.
**Fixed in [PR #32](https://github.com/AmedeeRoy/defile-migration-forecast/pull/32)**
(`fix/unet-squeeze-and-mask-buffer`, merged): now a non-persistent
`register_buffer`, so it follows the module's device without being rebuilt every forward
pass and without breaking existing checkpoints. The rest of this item is still open. The criterion
dataclasses annotate fields as `alpha: 1`, using the value as a type annotation so there
is no default (`criterion.py:69–70`, 135, 219, 254) — should be `alpha: float = 1.0`.
`plt_predict` hard-codes a 2×3 subplot grid and breaks if `forecast_day != 5`
(`src/plots/save_predict.py:8`). `configs/data/defile.yaml` sets
`years: range(1966, 2024)`, silently excluding the 2024 and 2025 seasons that are already
in the CSVs. **Resolved on `centralize-weather`:** the range is now
`range(1966, 2026)`. `src/export/` contains only a stale `__pycache__` — the source module is
gone, and the README still documents `src/export` and a `configs/export` that does not
exist.

### 4.19 Found while migrating the weather provider

These three were not in the original audit. All were confirmed by running code against the
live APIs, not read off the source.

**4.19a Production has been serving temperature and pressure about 260 m of elevation away
from what the model trained on.** Open-Meteo corrects `temperature_2m` and
`surface_pressure` to the DEM elevation of the requested coordinate — 417 m at Défilé — while
the GEE export returned the raw ERA5 cell value, which sits near 750 m. Measured over 8832
hours (August–October of 1970, 1995, 2015 and 2024), Open-Meteo runs **+1.84 K** warmer and
**+3151 Pa** higher than `data/era5/Defile.csv`. The two biases are mutually consistent:
3151 Pa is about 260 m, and 260 m at a 6.5 K/km lapse rate is 1.7 K. So this was a genuine
train/serve mismatch in production, on top of the unit and convention defects, and it
affected every species. *Resolved on `centralize-weather`:* both endpoints report the same
elevation, so training and serving now agree by construction. It is also a reason the
existing checkpoints cannot simply be carried over.

**4.19b The forecast path had no HTTP timeout.** `src/data/open_meteo.py` passed a
`requests_cache.CachedSession` into `retry_requests.retry()`. That function only applies its
5-second default timeout to a session it creates itself, so the supplied session kept
`requests`' default of waiting indefinitely. A hung Open-Meteo request would therefore stall
the 03:00 job forever rather than failing and alerting. *Resolved on `centralize-weather`:*
`_client` sets an explicit timeout on both paths (300 s for archive requests, 60 s for
forecast requests). Note the flip side, which bit during the backfill: `retry()` *does*
apply a 5-second timeout when it creates the session, which is far too short for a
multi-year archive request.

**4.19c CAPE does not exist in the ERA5 archive.**
`convective_available_potential_energy` is in `data/era5/Defile.csv` and in
`CONVERSION_DICT`, but the Open-Meteo archive returns all-null for `cape` with unit
`undefined`. No committed config currently requests it, so nothing is broken today.
*Handled on `centralize-weather`:* it is marked forecast-only and requesting it for training
raises instead of silently producing a NaN feature.

**4.19d The model trains on 25 km wind and is served ~2 km wind, and at Défilé those are
substantially different fields.** This is the most consequential of the four, and unifying the
provider does *not* fix it. The ERA5 archive is a 0.25° (~25 km) grid; the Open-Meteo forecast
endpoint serves high-resolution NWP. Measured over 61 days against the forecast path:

| | Défilé (gorge) | Frankfurt (flat) |
|---|---|---|
| `u_component_of_wind_10m` corr | **0.27** | 0.90 |
| `v_component_of_wind_10m` corr | 0.66 | 0.81 |
| `u_component_of_wind_100m` corr | 0.50 | 0.92 |
| `temperature_2m` corr | 0.97 | 0.97 |
| `surface_pressure` corr | 0.98 | 0.99 |
| `u_wind_10m` std ratio (fcst/ERA5) | **1.73** | 0.71 |

Temperature and pressure agree everywhere, so this is not a mapping error — it is resolution.
ERA5's cell cannot resolve the gorge that makes Défilé a bottleneck in the first place, while
the forecast model can, and produces 73% more variance in the along-valley component. So the
model learns wind–passage relationships from a field that does not contain the channelling
effect, then at 03:00 is handed a field that does.

Wind direction is one of the strongest drivers of raptor passage, which makes this a strong
candidate for why forecast skill might disappoint even after every defect above is fixed. It
also gives the Historical Forecast API proposal in 5.2 a concrete quantitative motivation
rather than a theoretical one: training on archived *forecasts* at matching lead times would
put both sides of the contract on the same model at the same resolution. A cheaper partial
mitigation is to pin the forecast endpoint to a coarse global model (`models=ecmwf_ifs025`)
so it matches ERA5's scale — note that line exists, commented out, in the retired
`open_meteo.py`, so someone had already considered it. That trades forecast sharpness for
train/serve consistency and should be measured, not assumed.

`tests/test_weather.py::test_wind_over_complex_terrain_is_documented_as_divergent` records the
effect so it cannot be quietly forgotten, and the parity tests deliberately assert scale
everywhere but correlation only where the two products genuinely track each other.

Two of the open questions in section 7 were also settled empirically. The radiation
conversion (W/m² × 3600 → J/m²) is **correct**: it agrees with the GEE export to 0.1% over
the same 8832 hours, so ERA5's hourly `surface_solar_radiation_downwards` is indeed an
hourly accumulation. And the space-named `prod/models/<species with spaces>/` directories
were **genuinely orphaned** — they contained only the gitignored `last.ckpt`, never the
tracked `best.ckpt` — confirming 4.10 and making them safe to delete.

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

> **Decided and implemented, August 2026 (branch `centralize-weather`).** Unify on
> Open-Meteo for both paths, as recommended below. What was built:
>
> - `src/data/weather.py` — one `get_weather(locations, variables, source=...)` entry point
>   over three sources (`"cache"` for training, `"archive"` to build that cache, `"forecast"`
>   for the daily job), sharing one `CONVERSION_DICT`, one set of pinned units, and one
>   `DAILY_AGGREGATION` rule per variable.
> - The archive is pinned to `models=era5` (0.25°, 1940-present), the same product the GEE
>   export used. Left unset, Open-Meteo would silently switch between ERA5, ERA5-Land, IFS
>   and CERRA depending on the date, changing dataset and resolution mid-history.
> - `scripts/build_weather_cache.py` writes a Parquet store under `data/weather/`, so
>   training never touches the network. Resumable at chunk granularity, because a full
>   1966-present backfill of all locations costs about 28 000 weighted API calls against a
>   free-tier allowance of 10 000/day and therefore takes roughly three days. (Open-Meteo
>   weights a call by variables × days, not per HTTP request.)
> - `tests/test_weather.py` — the parity check proposed at the end of this section, plus unit
>   tests for the wind convention, the unit conversions and the aggregation rules.
>
> Validated against the retired CSVs over 8832 hours: correlations 0.95–0.999 on every
> variable, confirming it is the same underlying ERA5. The one substantive difference found
> was the elevation correction, now recorded as defect 4.19a. `src/data/get_era5.py`,
> `src/data/open_meteo.py`, `data/era5/*.csv` and `data/era5/gee_code.js` are deleted.
>
> **Not fixed by this, and still open:** the reanalysis-versus-forecast distribution shift
> discussed below, and skill reporting by lead day. Those remain the substance of this
> question.

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
*Update: 4.1 (PR #26), 4.6 (PR #29) and 4.7 (PR #30) are merged. 4.2, 4.3, 4.4 and 4.5 are
fixed on `centralize-weather`, superseding PRs #27 and #28 (both left open, unmerged). The
train/serve parity check exists as networked tests in `tests/test_weather.py`; a
`train.py debug=default` smoke test does not yet exist as a committed test, though it has
been run manually to confirm the migration trains end to end. None of the
`prod/models/` checkpoints have been retrained against any of this, so no model-quality
comparison should happen until `centralize-weather` merges and retrain happens.*

**Second, make it fast enough to experiment.** Fix 4.14 (double forward) and 4.15
(per-sample xarray lookups), and cache the ERA5 reads (4.16). The point is to make the
year-subset ladder and the Optuna sweeps cheap enough to run often.
*Update: all three are done. 4.14 merged as PR #31 (bundled with the unrelated 4.13 fix).
4.15 merged as PR #35 (materialises the transformed stacks into contiguous arrays once per
split instead of per-sample `.sel(date=...)` calls). 4.16 is resolved on
`centralize-weather` via the Parquet cache.*

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
the assertion described in 4.2, or whether some local state avoids it. *Update: confirmed
moot on `centralize-weather` — `get_era5_hourly` and its assertion are gone, `train.py`
was run end to end against the current `defile.yaml` (including the restored far-field
daily locations) with no error.*

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
