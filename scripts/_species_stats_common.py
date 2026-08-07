"""Shared scaffolding for the `build_*_stats.py` scripts (`build_phenology_stats.py`,
`build_trend_stats.py`): sourcing the species list and season range from the project's
own Hydra configs rather than a second hardcoded copy, and writing output atomically.

Not a processing/statistics module -- the actual fitting (GAM day-of-year/hour surfaces,
Holt damped-trend smoothing, ...) stays in each script, since those have little in common
beyond "read counts, fit something per species, write JSON."
"""

import glob
import json
import os

import yaml


def species_from_experiments(configs_dir: str) -> list:
    """The set of modelled species, read from `configs/experiment/*.yaml`.

    Deliberately not a hardcoded list living in a build script: those experiment files
    are already the authoritative "what species does this project model"
    (`AGENTS.md` "Species / experiment pattern"), and a second copy is exactly the kind
    of drift that left `build_phenology_stats.py`'s predecessor notebook out of sync
    with its own output.
    """
    species = []
    for path in sorted(glob.glob(os.path.join(configs_dir, "experiment", "*.yaml"))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        name = cfg.get("data", {}).get("species")
        if name is None:
            raise ValueError(f"{path} has no data.species entry")
        species.append(name)
    return species


def default_doy_range(configs_dir: str) -> list:
    """The trained season, read from `configs/data/defile.yaml`'s `doy` field.

    Fitting a per-species statistics file over a different season than the model is
    trained on would make it a baseline/prior for a question nobody is asking; reading
    the value rather than copying it keeps the two from silently diverging.
    """
    with open(os.path.join(configs_dir, "data", "defile.yaml")) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["doy"])


def write_json_atomic(records: list, out_path: str) -> None:
    """Writes `records` to `out_path` via a temp file + rename.

    These files are read by every training/eval run and, once deployed, by defileViz or
    the daily forecast job -- a reader must never observe a half-written file.
    """
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp_path, out_path)
