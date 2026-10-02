# Count data

The counts come from the **defile-dataset** repo (`Rafnuss/defile-dataset`, private), which
holds the raw files, the protocol history (essential before choosing which years to use), the
data corrections and the documentation of every column. Its report lists the Trektellen entries
to correct. This folder only holds what the forecast builds from it.

```bash
# in defile-dataset
uv run python scripts/build_dataset.py
# here: copy its tables into data/count/dataset/ and build the model's counts
python scripts/build_counts.py --dataset ../defile-dataset/output
```

`data/count/dataset/` then holds `surveys.csv`, `observations.csv` and `metadata.json` (which
dataset build, from which commit and inputs). `all_count_processed.csv` is the model's input,
read by `DefileDataModule.read_counts` and `scripts/build_phenology_stats.py`; after it changes,
rebuild `species_doy_statistics.json` and retrain. The report,
`logs/qa/counts/count_processing_report.html`, shows every check and, for every rule below, the
rows and birds it touched.

## What the model needs

`all_count_processed.csv` has one row per species per survey period: `species` (the eBird English name of the dataset's `avibase_id`, as `configs/experiment/` and defileViz use them; the AviList name where eBird has none; `No species` and `Non-bird` placeholders),
`date` (local), `count`, `start`/`end` (UTC). Effort is not stored separately: **a survey period
is any `(start, end)` that appears on at least one row**, and a species absent from a period is
counted as zero there. Two consequences:

- An hour surveyed with no bird of any species must still get a row, or it is not effort at
  all. These rows have species `No species` and count 0.
- Rows of non-birds (butterflies, dragonflies) are kept as `Non-bird`: they count for no
  species but still mark their period as surveyed.

The model works on clock hours (UTC). Data recorded hour by hour is kept as hours; data recorded
as one total for a long period stays one long period (the datamodule uses its mean hourly rate,
with a mask of the hours it covers).

## Rules

Observations the dataset flags `no_time`, `no_survey`, `duplicate_survey` or
`time_outside_survey` are dropped first. Survey times are the dataset's corrected ones.

### Historical (1966-2021)

1. Records of the same species in the same period are summed.
2. **Zero-fill.** A day counts as recorded hour by hour when one of its periods lies strictly inside the day's survey window (`startTimeDay`/`endTimeDay`, longer than 2 h). Every clock hour of that window that overlaps no record gets a `No species` row with count 0. Partial first/last hours are kept as such.

### Trektellen (2022 on)

The count is `direction1` (birds moving in the main migration direction).

1. **Splitting.** A count period is split into clock hours if it lasts more than 2 h and fewer than half of its entries with migrating birds (`count > 0`) lack a timestamp. Otherwise it stays one period. Entries of local birds only are left out of that share: they are often untimed even in a timed count.
2. In a split period, entries without a timestamp are dropped: they cannot be placed in an hour. They are mostly local birds, or totals entered at the end of the day.
3. Each entry of a split period gets the clock hour of its timestamp as its period, clipped to the count period.
4. **Zero-fill.** Each clock hour (at least 10 min) of a split period with no entry of any species gets a `No species` row with count 0.
5. Periods shorter than 10 min are dropped, with their birds. This is mostly the partial last hour of a split period.
6. Entries of the same species in the same period are summed.

## Checks

Run on every build and listed at the top of the report: every bird of the dataset is either in
the output or dropped by a named rule; no two periods overlap; no period has zero length;
periods under 15 min or windows over 16 h are flagged; one row per species and period; every
species has an English name; every clock hour of a split window exists as a period.

Until 2026-10 the Trektellen zero-fill (rule 4) never ran: the notebook this replaced tested each
empty hour against its whole count period, which it always overlaps. Fixing it, with the
dataset's night and overlap corrections, added 929 survey periods (812 h) with their zero counts
to 2022-2026 and removed 258 birds counted twice. The historical data is unchanged.
