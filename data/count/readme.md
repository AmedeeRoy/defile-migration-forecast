# Count data

The counts come from the **defile-dataset** repo (`Rafnuss/defile-dataset`, private), which
holds the raw files, the protocol history (essential before choosing which years to use), the
data corrections and the documentation of every column (its `output/dataset/README.md`). Its
audit lists the Trektellen entries to correct. This folder only holds what the forecast builds
from it.

```bash
# in defile-dataset
uv run python scripts/build_dataset.py
# here: copy its release tables into data/count/dataset/ and build the model's counts
python scripts/build_counts.py --dataset ../defile-dataset/output
```

`data/count/dataset/` then holds the release tables `count.csv`, `survey.csv`, `taxonomy.csv`
and `datapackage.json`, and the build's `metadata.json` (which dataset build, from which commit
and inputs). `all_count_processed.csv` is the model's input, read by
`DefileDataModule.read_counts` and `scripts/build_phenology_stats.py`; after it changes, rebuild
`species_doy_statistics.json` and retrain. The report,
`logs/qa/counts/count_processing_report.html`, shows every check and, for every rule below, the
rows and birds it touched.

## What the model needs

`all_count_processed.csv` has one row per species per survey period: `species` (the dataset's
English name of the taxon, AviList first, as `configs/experiment/` uses it; `No species` as the
effort placeholder), `date` (local), `count`, `start`/`end` (UTC). Effort is not stored
separately: **a survey period is any `(start, end)` that appears on at least one row**, and a
species absent from a period is counted as zero there. So a period surveyed with no bird of any
species must still get a row, or it is not effort at all. These rows have species `No species`
and count 0.

The model works on clock hours (UTC). Data recorded hour by hour is kept as hours; data recorded
as one total for a long period stays one long period (the datamodule uses its mean hourly rate,
with a mask of the hours it covers).

## Rules

The count is the main migration direction (`count_category = normal`); `reverse` and `local`
counts are other quantities, not modelled. A count takes its own timing when it has one, else its
survey's interval.

### Both sources, first

- **Coverage.** Only `complete` surveys are used. `none` surveys (rain, low cloud, closures) have
  no counts and are not effort. `partial` and `unknown` surveys are dropped with their birds: with
  unknown gap times, neither their rate over the interval nor their empty hours can be trusted.
- **Presence only** (`count_estimation = x`, no number) is dropped: never turned into a count.
- **Timed outside its survey.** The dataset keeps such an entry at day level, like an untimed
  one; its `remark_processing` ("Entry time outside the native survey") tells them apart. It
  belongs to no counted period, so it is dropped.

### Historical (1966-2021)

Historical counts carry no timing of their own; days recorded hour by hour are one survey per
hour in the dataset.

1. Records of the same taxon in the same survey (age/sex subgroups, repeated entries) are summed.

### Trektellen (2022 on)

1. **Splitting.** A count period is split into clock hours if it lasts more than 2 h and fewer
   than half of its entries with migrating birds (`count > 0`) lack a timestamp. Otherwise it
   stays one period. Entries of local birds only are left out of that share: they are often
   untimed even in a timed count.
2. In a split period, entries without a timestamp are dropped: they cannot be placed in an hour.
   They are mostly local birds, or totals entered at the end of the day.
3. Each entry of a split period gets the clock hour of its timestamp as its period, clipped to
   the count period.
4. **Zero-fill.** Each daylight clock hour (at least 10 min) of a split period with no entry of
   any species gets a `No species` row with count 0. An hour lying wholly at night (sun below
   -6 deg, the model's one definition of night) is not zero-filled: counts left open overnight
   would otherwise add hours of zeros nobody watched.
5. Periods shorter than 10 min are dropped, with their birds. This is mostly the partial last
   hour of a split period.
6. Entries of the same species in the same period are summed.

### Both sources, last

- **Empty surveys.** A `complete` survey of at least 10 min that no record overlaps gets a
  `No species` row with count 0: a period counted with nothing seen (explicit "no species"
  entries included). Days without counting are `none` in the dataset, so they never get here.

## Checks

Run on every build and listed at the top of the report: every main-direction bird of the dataset
is either in the output or dropped by a named rule; no two periods overlap; no period has zero
length; periods under 15 min or windows over 16 h are flagged; one row per species and period;
every species has an English name; every daylight clock hour of a split window exists as a
period; empty surveys alone on their day are flagged (real zeros only if the day was counted).

## History

Until 2026-10 the Trektellen zero-fill (rule 4) never ran: the notebook this replaced tested each
empty hour against its whole count period, which it always overlaps. Fixing it, with the
dataset's night and overlap corrections, added 929 survey periods (812 h) with their zero counts
to 2022-2026 and removed 258 birds counted twice. The historical data is unchanged.

In 2026-10 the input moved from the dataset's interim two-table layout (`surveys.csv`,
`observations.csv`, with flags) to its release tables. Historical birds are unchanged; see
`DECISIONS.md` (Counts) for what changed in effort and Trektellen birds, and why.
