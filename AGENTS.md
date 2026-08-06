# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, etc.) working in this repo.

## What this project does

Trains and runs deep-learning models (PyTorch Lightning + Hydra) that forecast daily
raptor migration counts at Défilé de l'Écluse, France, from weather data (ERA5
reanalysis + Open-Meteo forecasts) and historical count data (Trektellen). The
resulting forecasts are consumed by a separate companion repo, **defileViz**
(Vue 3 + Vite frontend, deployed to GitHub Pages), which renders them for site
visitors. Changes to output format/paths here should stay compatible with that repo,
or be coordinated with it.

## Repo layout

```
configs/        Hydra configs
  callbacks/       Lightning callback configs
  data/            Dataset configs (defile.yaml = full, defile-small.yaml = fast/small subset)
  debug/           Fast smoke-test configs (default, limit, overfit, profiler, fdr)
  experiment/      One config per species — see "Species pattern" below
  hparams_search/  Optuna/Hydra sweep configs
  hydra/           Hydra run-dir / logging configs
  logger/          Logger backends (csv, wandb, mlflow, tensorboard, ...)
  model/           Model architecture configs (unet, transformer, convnet)
  paths/           Path configs (default = local dev, production = deployed)
  trainer/         Lightning Trainer configs (cpu, gpu, mps, ddp)
  train.yaml, eval.yaml, predict.yaml, test.yaml   Top-level entry configs
data/            Raw + processed data (count/, era5/, taxonomy.csv, ...) — gitignored
logs/            Lightning/Hydra run outputs — gitignored
notebooks/       Exploration notebooks; retired ones live in notebooks/old/
prod/            Production artifacts (gitignored, generated):
  models/<species>/checkpoints/best.ckpt   the deployed checkpoint per species
  forecasts/                                exported forecasts consumed by defileViz
scripts/
  schedule.sh                    cron entrypoint for scheduled predict+deploy runs
  move_checkpoints_to_prod.py    promotes the latest training run's best.ckpt to prod/
  build_phenology_stats.py       builds data/count/species_doy_statistics.json
src/
  train.py, eval.py, predict.py   entry points (Hydra @hydra.main)
  data/                            DefileDataModule, ERA5/Open-Meteo fetch + transform
  models/                          LightningModule, criterion (Tweedie loss, etc.), components/ (unet/transformer/convnet)
  plots/                           saved-prediction and test-set plotting helpers
  utils/                           logging, instantiators, misc helpers
```

## Environment

Managed with [uv](https://docs.astral.sh/uv/), Python 3.10 (`.python-version`; uv provisions
this itself, no separate Python install needed). Exact versions pinned in `pyproject.toml`
and locked in `uv.lock` -- deliberately pinned, not ranged, so `uv lock` alone can't silently
drift to a newer release. Core deps: `torch==2.5.1`, `lightning==2.5.5`,
`hydra-core==1.3.2`, `pandas==2.3.3`, `xarray==2023.6.0`, `rich`, `matplotlib`, `einops`,
`cloudpickle`, `hydra-optuna-sweeper`, `hydra-colorlog`, `rootutils`, `suncalc`,
`openmeteo-sdk`/`openmeteo-requests`, `requests-cache`, `retry-requests`,
`captum` (used for saliency/explanations in `src/plots/explanations.py`).

```bash
uv sync                          # creates .venv, installs everything at the pinned versions
uv run python src/train.py       # run any command inside it
# or: source .venv/bin/activate && python src/train.py
```

## Key commands

```bash
# train (default species = Common Buzzard, from configs/data/defile.yaml)
python src/train.py

# train a specific species (see configs/experiment/)
python src/train.py experiment=red_kite

# train all 11 species in one multirun (as used in production, see scripts/schedule.sh)
python src/train.py --multirun experiment=common_buzzard,red_kite,black_kite,honey_buzzard,marsh_harrier,sparrowhawk,kestrel,osprey,hen_harrier,merlin,hobby trainer=gpu

# hyperparameter search for one species (Optuna via hydra-optuna-sweeper)
python src/train.py task_name=optim data.species="Red Kite" hparams_search=unet trainer=gpu

# ad hoc overrides
python src/train.py data.species="Red Kite" data.years="[2020,2021,2022,2023]"

# fast smoke test — run this after a code change before a full training run
python src/train.py debug=default
# other debug variants: debug=limit / debug=overfit / debug=profiler / debug=fdr

# evaluate a checkpoint
python src/eval.py ckpt_path=<path/to/checkpoint.ckpt>

# predict with the production model for all species (as run in CI, see below)
python src/predict.py --multirun experiment=common_buzzard,red_kite,black_kite,honey_buzzard,marsh_harrier,sparrowhawk,kestrel,osprey,hen_harrier,merlin,hobby

# predict for one species with the production checkpoint
# (reads prod/models/<species>/checkpoints/best.ckpt, per configs/predict.yaml)
python src/predict.py experiment=red_kite

# promote a freshly trained checkpoint to production
python scripts/move_checkpoints_to_prod.py --dry-run   # check first
python scripts/move_checkpoints_to_prod.py             # then actually move
```

`trainer=gpu` is used for real training/hyperparameter search (see `configs/trainer/`
for `cpu`/`gpu`/`mps`/`ddp` options) — default CPU is fine for the `debug=` smoke tests
above.

Pre-commit hooks (`.pre-commit-config.yaml`) — run `pre-commit run --all-files` before
treating a change as done. Configured hooks: standard hygiene checks
(`check-added-large-files`, `check-merge-conflict`, `check-yaml`, `end-of-file-fixer`,
`trailing-whitespace`), `black` (line length 99), `isort` (black profile),
`docformatter`, `prettier` for YAML, `mdformat`,
`codespell` (skips `logs/`, `data/`, `*.ipynb`), and — importantly —
**`nbstripout`**, which should strip notebook outputs automatically on commit. If a
notebook diff still carries megabytes of embedded output, the hook likely isn't
installed locally: run `pre-commit install` once per clone.

## Model architecture

`UNetplus` (`src/models/components/unet.py`) has two branches, wrapped by
`DefileLitModule` (`src/models/defile_module.py`):

- **Hourly branch** — a 1-D U-Net over the 24-hour axis. Input channels are the "main"
  ERA5 stack at Défilé (14 variables + sun altitude/azimuth), the hourly stack at 5
  nearby locations (5 × 14 channels), plus day-of-year and year broadcast across hours.
  Learns the *shape* of the day.
- **Daily branch** — a 1-D conv stack over the lag axis (`lag_day` days of history) on the
  daily ERA5 stack at 7 regional locations. Learns the daily *magnitude*.
- Combined multiplicatively: `out = 8 * out_h * out_d`, both sigmoid-bounded, so output is
  in `log1p(birds/hour)` capped at 8 (≈2 979 birds/h). Hours <05 and ≥19 UTC are forced to
  zero by a hard-coded mask in `forward`.

Targets are hourly rates (`count / survey duration`), with a 24-element `mask` giving the
fraction of each hour covered by the survey. Loss is `TweedieLoss + ProbaRMSE`
(`src/models/criterion.py`), weighted per species by Optuna-tuned values in
`configs/experiment/*.yaml`.

Feature-count expressions in `configs/model/unet.yaml` are derived from the data config via
Hydra resolvers (`${len:...}`, `${eval:...}`) — if you change the variable/location lists
in `configs/data/*.yaml`, the input dimensions follow automatically. Don't hardcode them.

## One weather path (important)

All weather goes through `src.data.weather.get_weather`, which takes a `source`:

- `source="cache"` — the local Parquet store in `data/weather/`, built by
  `scripts/build_weather_cache.py`. This is what training reads; no network access.
- `source="archive"` — the Open-Meteo ERA5 archive API, pinned to `models=era5`. Used only
  to build the cache.
- `source="forecast"` — the Open-Meteo forecast API. Used by the daily prediction job.

All three share one `CONVERSION_DICT`, one set of pinned request units, and one
`DAILY_AGGREGATION` rule per variable, so a change to variables, units or aggregation is
made **once**. This replaced two independent implementations (GEE CSVs for training,
Open-Meteo for serving) that had silently drifted apart.

When touching this module, keep that single-path property: add a variable to
`CONVERSION_DICT`, not to a caller, and never special-case one `source` in a way that
changes the resulting values. `tests/test_weather.py` enforces the conventions and the
train/serve contract — run `pytest tests/` and, for API changes, `pytest tests/ -m network`.

The cache is gitignored and takes about three days to backfill fully (Open-Meteo weights
API calls by variables x days, and a full 1966-present fetch of all locations costs roughly
28 000 against a 10 000/day free allowance). The build script is resumable at chunk
granularity — re-run it, it skips what it already has. If the cache is only partially built,
`load_cache` warns about the requested years it cannot supply rather than silently training
on fewer.

Normalisation parameters are fitted at training time and pickled to
`data/transform_data.pickle` (committed to git); `predict.py` loads that file. It must
correspond to the checkpoints in `prod/models/`, and retraining overwrites it.

## Known defects — read before making changes

`DEVELOPMENT.md` at the repo root is the live roadmap: open defects and the phased plan to
close them, nothing archived. **Read it before trusting any model metric or tuning
hyperparameters** — the values currently in `configs/experiment/*.yaml` were tuned before
several of these fixes landed and before this migration, so they should be considered void
until retrained.

The one worth knowing about even without opening the file: Défilé sits in a gorge that
ERA5's 25 km cell cannot resolve, so 10 m wind correlates only ~0.27 between the training
and serving products there (~0.90 over flat terrain). That's a genuine train/serve
distribution shift which unifying the provider did **not** fix, and it is plausibly the
biggest remaining limit on forecast skill (`DEVELOPMENT.md` 4.19d).

Tests live in `tests/` and cover the weather layer only. The rest of the codebase has none.

## Species / experiment pattern

Each of the 11 species trained here has its own config in `configs/experiment/`:
`black_kite`, `common_buzzard`, `hen_harrier`, `hobby`, `honey_buzzard`, `kestrel`,
`marsh_harrier`, `merlin`, `osprey`, `red_kite`, `sparrowhawk`. Each overrides
`data.species` and tuned model hyperparameters (optimizer, Tweedie-loss criterion, etc.)
on top of the shared `data: defile` and `model: unet` defaults. When adding a species
or retuning one, follow this pattern — add/edit a file in `configs/experiment/` —
rather than editing `configs/data/defile.yaml` directly.

## Data pipeline notes

- Counts: `data/count/all_count_processed.csv`, one row per species per survey period,
  1966–2025, `start`/`end` in UTC. Survey protocol and recording granularity changed
  repeatedly over that span — `data/count/readme.md` documents the history and is
  essential reading before making modelling decisions about which years to use.
- Weather: `data/weather/` holds hourly ERA5 for all 13 locations in one Parquet store,
  partitioned by location, at ~3.5 MB per location per decade. Every location has the same
  columns and the same semantics — the old split between hourly CSVs and daily far-field
  aggregates is gone.
- `lag_day` / `forecast_day` in `configs/data/defile.yaml` control how many days of
  history feed the model and how many days ahead it forecasts.
- `doy: [196, 335]` restricts training to the migration season (mid-July to end of
  November). The model has never seen other parts of the year.
- `configs/data/defile-small.yaml` is a reduced dataset for fast local iteration —
  prefer it over the full `defile.yaml` when just testing a code change.
- `data/`, `logs/`, and `prod/forecasts/` are gitignored, generated artifacts. Don't
  hand-edit them or assume they're committed — regenerate via the commands above.
  Exceptions that *are* committed: `data/transform_data.pickle`,
  `data/count/readme.md`, `data/count/species_doy_statistics.json`, and
  `prod/models/*/checkpoints/best.ckpt`.
- `data/count/species_doy_statistics.json` — per-species day-of-year phenology
  (`src.phenology.Phenology`): the skill-score baseline every test-report metric is judged
  against, and also what defileViz renders as its uncertainty band. Built by
  `python scripts/build_phenology_stats.py` (species list read from
  `configs/experiment/*.yaml`, so it never needs a second species list kept in sync by
  hand). `notebooks/phenology_baseline.ipynb` calls the same builder for exploration only — it
  does not write the file itself anymore.

## Known environment gotcha: OneDrive sync

This repo currently lives inside a OneDrive-synced folder. Two things to watch for:

1. OneDrive "Files On-Demand" keeps rarely-opened files as cloud-only placeholders.
   Reading one can fail with errors like "Resource deadlock avoided", or hang, even
   though it shows up with the right name/size in a directory listing. If this
   happens, ask the user to open the file in Finder (or mark the folder "Always keep
   on this device") to force a download, then retry.
2. The same sync layer can make `git status` show many files as modified with no
   real content change (it's touching metadata/permissions, not content). Always
   check `git diff`, don't trust `git status` alone before committing.
3. `.venv/` (created by `uv sync`) is thousands of small files living in this same synced
   folder — a prime candidate for the placeholder/deadlock issue in (1), and there is no
   reason to sync it at all (it's gitignored and fully reproducible from `uv.lock`). If
   `uv sync`/`uv run` hang or error strangely, that's the first thing to suspect.

## Things not to do

- Don't commit anything under `data/`, `logs/`, `prod/forecasts/`, or `.secrets.env`
  (already gitignored — keep it that way).
- Don't re-download ERA5/count/Open-Meteo data unless asked to — these come from
  external APIs/services and re-fetching can be slow or rate-limited.

## Production pipeline (`.github/workflows/predict_and_deploy_forecasts.yml`)

Runs daily at 03:00 UTC (cron), on every push to `main`, and on manual dispatch:

1. Installs the pinned environment with `uv sync --locked`.
2. Runs `uv run python src/predict.py --multirun experiment=<all 11 species>`, producing
   NetCDF forecast files under `prod/forecasts/`.
3. Uploads those forecasts to a GCE host via SCP (`secrets.GCE_HOST/USER/SSH_KEY`) —
   this is what actually serves the files defileViz consumes (see below).
4. Also deploys `www/` (this repo's own minimal page) to this repo's GitHub Pages.

There's a second workflow, `.github/workflows/test_gce.yml`, presumably a
connectivity check for the GCE deploy step — I haven't been able to read its contents
yet (OneDrive placeholder issue), so treat that description as unconfirmed.

## Related repo

The frontend that displays these forecasts — **defileViz** (Vue 3 + Vite + Plotly,
deployed to its own GitHub Pages) — is a separate git repo, not part of this working
directory. It does **not** read `prod/forecasts/` directly or go through this repo's
own GitHub Pages site: `src/services/netcdf.js` in defileViz fetches NetCDF files
straight from `https://defile.raphaelnussbaumer.com/forecasts/<species>/<YYYYMMDD>_<species>.nc`
— i.e. from the GCE host this workflow uploads to. Any change here to the
forecast filename pattern, folder structure, or NetCDF variable names
(e.g. `pred_log_hourly_count`) must stay in sync with that URL scheme and with
defileViz's expectations, or be coordinated across both repos.
