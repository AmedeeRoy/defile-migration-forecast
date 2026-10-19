# Count data

The counts come from [defile-dataset](https://github.com/Rafnuss/defile-dataset), which
publishes the consolidated count, survey and taxonomy tables of the Défilé de l'Écluse raptor
counts as [GitHub releases](https://github.com/Rafnuss/defile-dataset/releases). Its release
notes and data dictionary document the source data, its corrections and every column; none of
that is repeated here. This folder only holds what the forecast model builds from it.

```bash
python scripts/build_counts.py        # download the pinned release and build the model's counts
```

The script downloads the release assets into `data/count/dataset/` (checking them against the
release's `SHA256SUMS`), applies the processing rules below, and writes:

- `all_count_processed.csv`, the model's input (read by `DefileDataModule.read_counts` and
  `scripts/build_phenology_stats.py`);
- `unknown_species_dates.csv`, the species-days excluded from training (see the last section);
- `logs/qa/counts/count_processing_report.html`, which shows every check and, for each rule
  below, the rows and birds it touched.

The release used is pinned by `DATASET_RELEASE` in `src/data/counts.py`; pass
`--release <tag>` to use another. After the counts change, rebuild
`species_doy_statistics.json` (`scripts/build_phenology_stats.py`) and retrain. To test an
unpublished dataset build, `--dataset <path to its output/ folder>` copies it instead of
downloading.

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

- **Completeness.** Complete surveys (`survey_complete`) supply effort. Incomplete surveys are dropped with their birds. Complete weather stops (`weather_stop`) supply zero passage, following the release's documented assumption. Unmonitored absences have no survey row.
- **Presence only** (`count_estimation = x`, no number) is dropped: never turned into a count.
- **Count intervals.** An own interval up to one hour is timed at its midpoint; longer intervals stay at day level. UTC times determine the Europe/Paris date.
- **Timed outside its survey.** Day-level entries marked outside their survey cannot be placed in model periods and are dropped. Only the released tables are read.

### Historical (1966-2021)

Historical counts carry no timing of their own; days recorded hour by hour are one survey per
hour in the dataset. Hours of the declared day window with no record are empty surveys there
(see its README), so they become zero rows under "Empty surveys" below.

1. Records of the same taxon in the same survey (age/sex subgroups, repeated entries) are summed.

### Trektellen (2022 on)

0. **Night.** A count starting more than 45 min before civil dawn or ending more than 45 min
   after civil dusk (sun at -6 deg) starts at dawn / ends at dusk: a count left open in the dark
   is an entry error.
   **Tolerance.** Then an entry timed less than 10 min before its count starts or after it ends
   (its end is exclusive: an entry at that very minute is outside by zero) is moved 1 min inside
   it: clock rounding, a late entry. Further out, it belongs to no counted period and is dropped.
1. **Splitting.** A count period is split into clock hours if it lasts more than 2 h and fewer
   than half of its entries with migrating birds (`count > 0`) lack a timestamp. Otherwise it
   stays one period. Entries of local birds only are left out of that share: they are often
   untimed even in a timed count.
2. In a split period, entries without a timestamp are dropped: they cannot be placed in an hour.
   They are mostly local birds, or totals entered at the end of the day.
3. Each entry of a split period gets the clock hour of its timestamp as its period, clipped to
   the count period.
4. **Zero-fill.** Each clock hour (at least 10 min) of a split period with no entry of any
   species gets a `No species` row with count 0.
5. Periods shorter than 10 min are dropped, with their birds. This is mostly the partial last
   hour of a split period.
6. Entries of the same species in the same period are summed.

### Both sources, last

- **Empty surveys.** A `complete` survey of at least 10 min that no record overlaps gets a
  `No species` row with count 0: a period counted with nothing seen (explicit "no species"
  entries included), including weather stops with assumed zero passage. Unmonitored absences have no survey row.

## Checks

Run on every build and listed at the top of the report: every main-direction bird of the dataset
is either in the output or dropped by a named rule; no two periods overlap; no period has zero
length; periods under 15 min or windows over 16 h are flagged; one row per species and period;
every species has an English name; every clock hour of a split window exists as a
period; empty surveys alone on their day are flagged (real zeros only if the day was counted).

## Unknown species-days

`unknown_species_dates.csv` accompanies the prepared counts. A presence-only record, unresolved outside-survey timing, or positive untimed count removed from a split survey makes that species-day unknown for model targets. Training and phenology exclude it rather than zero-fill it; other species retain their valid targets. Rebuild both files with `scripts/build_counts.py`.
