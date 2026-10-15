# Data correctness follow-up to #58

The release reader and faster loading are already on main. This PR repairs two remaining data representation problems without changing BatchNorm ordering, checkpoints, the committed phenology prior or normalization. Data correctness is tested directly; model skill is not an acceptance criterion for this step.

- Hourly masks now retain exposure across UTC midnight, including full-day weather stops. Equal start/end clock hours previously gave a zero mask and NaN loss for a 24-hour interval.
- Preparation writes `data/count/unknown_species_dates.csv`. Positive untimed counts discarded by hourly splitting, entries whose timing cannot be placed inside their survey, and presence-only records mark the affected species-day as unknown. Training and phenology omit these species-days rather than inventing numerical zeros. Other species retain their valid targets. Source counts and named exclusions remain available in the prepared count file and QA report.

The current release produces 120,081 prepared rows and 184 unknown species-days across all taxa. Presence-only records are absent in this snapshot, but their behavior is regression-tested. Birds reconcile by source and year; the release reader matches Explore. Incomplete intervals supply no effort, weather stops retain their explicit assumed-zero interpretation, and clock-hour exposure sums to interval duration.

Rebuild the prepared inputs with `uv run python scripts/build_counts.py --dataset ../defile-dataset/output` before loading them; the unknown-species-day companion file is required. This PR does not rebuild or promote the committed prior, because changing it changes existing forecasts. Prior and checkpoint updates need separate review after the data definition is settled.

PR #47 is stacked on this branch and contains only the daily layer-order change and its documentation/tests. Earlier fresh checkpoint runs are archived as a combined experiment, not proposed for deployment. Their mean skill change (0.229 to 0.170) changed training inputs, prior and architecture together; it cannot be attributed to either data or BatchNorm. Matched retraining would isolate the architecture effect, but it is not needed for the data correctness checks and is deferred.
