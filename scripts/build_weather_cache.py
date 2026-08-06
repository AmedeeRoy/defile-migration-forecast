"""Builds the local hourly weather cache that training reads.

Downloads the full ERA5 history for every location in `src.data.weather.LOCATIONS` from the
Open-Meteo archive API and writes it to a Parquet store partitioned by location. Training
never touches the network: it reads this store through
`src.data.weather.get_weather(source="cache", ...)`.

The store replaces the per-location CSV exports from Google Earth Engine that used to live
in `data/era5/`. It is deliberately gitignored, like those CSVs were — re-running this
script is the documented way to reproduce it.

Run it once to populate the cache, and again to extend it as ERA5 catches up (the archive
runs about five days behind real time):

    python scripts/build_weather_cache.py

Useful options:

    --start 1993-01-01          only fetch from a given date
    --locations Defile Basel    only fetch some locations
    --overwrite                 refetch chunks already present
    --chunk-years 5             tune the request size
    --pace 30                   seconds between requests

Every variable the archive serves is fetched regardless of what the current config uses, so
changing `era5_*_variables` never requires a refetch.

A full 1966-present backfill of all locations costs roughly 28 000 weighted API calls
against a free-tier allowance of 10 000 per day, so it takes about three days. The script is
resumable at chunk granularity: it writes each chunk as it arrives, skips chunks already on
disk, waits out per-minute rate limits, and exits cleanly when the daily limit is reached.
Re-running it the next day picks up at the first missing chunk. To fit a single day instead,
narrow the work with `--start` or `--locations`.
"""

import argparse
import os
import sys
import time

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.weather import (  # noqa: E402
    ARCHIVE_MODEL,
    CACHE_SUBDIR,
    CONVERSION_DICT,
    LOCATIONS,
    fetch_archive,
)

# ERA5 starts in 1940; the count data starts in 1966, so there is nothing to gain earlier.
DEFAULT_START = "1966-01-01"

# The ERA5 archive runs roughly five days behind real time. Asking for more than it has
# returns a shorter series rather than an error, but stopping short keeps the cache free of
# a ragged tail.
ARCHIVE_LAG_DAYS = 7

# Variables available from the archive. CAPE is excluded: it is forecast-only.
ARCHIVE_VARIABLES = [
    name for name, spec in CONVERSION_DICT.items() if spec.get("archive", True)
]


def default_end_date():
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=ARCHIVE_LAG_DAYS)).strftime(
        "%Y-%m-%d"
    )


# Chunks are aligned to fixed calendar blocks counted from this epoch rather than from
# whatever `--start` was passed. That makes a chunk's identity, and therefore its filename,
# independent of how the script was invoked, so re-running with a different `--start` or
# `--end` skips the blocks already on disk instead of refetching or duplicating them.
CHUNK_EPOCH_YEAR = 1940


def date_chunks(start, end, chunk_years):
    """Splits [start, end] into inclusive chunks aligned to fixed calendar blocks.

    Fetching in chunks means a transient failure or a rate limit costs one chunk rather than
    the whole history, and lets an interrupted backfill resume where it stopped.

    A block is always fetched whole rather than clipped to `start`, so the file for a block
    always holds that block's entire contents. Clipping would write a partial block under
    the full block's name, and a later run over a wider period would then skip it and leave
    a permanent hole in the cache. The only partial block is the last one, which `end`
    truncates; it is marked incomplete so later runs refetch it as ERA5 catches up.

    Returns a list of ``(start, end, block_year, complete)`` tuples.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    block = (start.year - CHUNK_EPOCH_YEAR) // chunk_years
    chunks = []
    while True:
        lo = pd.Timestamp(year=CHUNK_EPOCH_YEAR + block * chunk_years, month=1, day=1)
        block_end = pd.Timestamp(
            year=CHUNK_EPOCH_YEAR + (block + 1) * chunk_years, month=1, day=1
        ) - pd.Timedelta(days=1)
        if lo > end:
            break
        chunks.append(
            (
                lo.strftime("%Y-%m-%d"),
                min(block_end, end).strftime("%Y-%m-%d"),
                lo.year,
                block_end <= end,
            )
        )
        block += 1
    return chunks


def chunk_path(cache_dir, location, block_year):
    """Path of the Parquet file holding one location's chunk.

    `location` is encoded in the directory name rather than stored as a column, which is
    what lets a read for one location skip every other location's files entirely.
    """
    return os.path.join(cache_dir, f"location={location}", f"part-{block_year}.parquet")


def chunk_is_up_to_date(path, hi):
    """Whether an already-cached trailing chunk already covers through `hi`.

    The trailing (current) chunk is deliberately marked incomplete by `date_chunks` so a
    later run can extend it as ERA5 catches up -- but "incomplete" only means there is
    something new to fetch if the requested end date has actually moved past what is
    already on disk. Without this check, running the script twice within the same
    ARCHIVE_LAG_DAYS window re-downloads the exact same range for no benefit, since
    `default_end_date()` only changes once a day.
    """
    if not os.path.exists(path):
        return False
    existing_max = pd.read_parquet(path, columns=["datetime"])["datetime"].max()
    return existing_max >= pd.Timestamp(hi)


def write_chunk(path, df):
    """Writes one (location, chunk) frame to Parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = df.drop(columns=["location"]).sort_values("datetime").reset_index(drop=True)

    # `year` and `doy` exist purely so a read can push those filters down to Parquet and
    # avoid materialising rows outside the requested years and migration season.
    df.insert(1, "year", df["datetime"].dt.year.astype("int16"))
    df.insert(2, "doy", df["datetime"].dt.dayofyear.astype("int16"))

    df.to_parquet(path, index=False, compression="zstd")
    return path


# Free non-commercial tier limits, in weighted calls (see `api_call_weight`).
MINUTELY_CALL_LIMIT = 600
HOURLY_CALL_LIMIT = 5_000
DAILY_CALL_LIMIT = 10_000


def api_call_weight(n_days, n_variables):
    """Open-Meteo's weighted cost of a single-location request.

    Open-Meteo does not count one HTTP request as one API call. A request is weighted by
    both its variable count and its length: their documented example is that 14 days of 15
    variables costs 1.5 calls. The free non-commercial tier allows 600 calls/minute,
    5 000/hour and 10 000/day, so a full 1966-present backfill of every location costs
    roughly 28 000 calls and has to be spread over about three days.
    """
    return (n_variables / 10) * (n_days / 14)


def fetch_with_retry(max_retries=6, wait=90, **kwargs):
    """Calls `fetch_archive`, waiting and retrying when the API rate limit is hit.

    The rate limit is reported as an ordinary error response rather than an HTTP 429, so the
    session's own retry logic does not see it and it has to be handled here.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_archive(**kwargs)
        except Exception as exc:  # noqa: BLE001 - the SDK wraps every failure in one type
            rate_limited = "limit exceeded" in str(exc).lower()
            if not rate_limited or attempt == max_retries:
                raise
            print(
                f"    rate limit hit, waiting {wait}s "
                f"(attempt {attempt}/{max_retries - 1})..."
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default=DEFAULT_START, help="First date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=None,
        help=f"Last date (YYYY-MM-DD). Defaults to today minus {ARCHIVE_LAG_DAYS} days.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=list(LOCATIONS),
        help="Locations to fetch. Defaults to all known locations.",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Data directory holding the cache."
    )
    parser.add_argument(
        "--chunk-years", type=int, default=5, help="Years of history per request."
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=None,
        help="Seconds between requests. Defaults to whatever keeps the run under the "
        "hourly call limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refetch chunks that are already cached.",
    )
    args = parser.parse_args(argv)

    end = args.end or default_end_date()
    cache_dir = os.path.join(args.data_dir, CACHE_SUBDIR)

    chunks = date_chunks(args.start, end, args.chunk_years)

    # Work is enumerated as (location, chunk) pairs so an interrupted or rate-limited run
    # resumes at the chunk it stopped on rather than restarting a whole location.
    jobs = [
        (location, lo, hi, block, complete)
        for location in args.locations
        for lo, hi, block, complete in chunks
    ]
    if not args.overwrite:
        n_all = len(jobs)

        def already_cached(job):
            location, lo, hi, block, complete = job
            path = chunk_path(cache_dir, location, block)
            # A complete chunk is skippable once its file exists -- it will never change.
            # A trailing (incomplete) chunk is only skippable if its file already reaches
            # the requested end date; otherwise there is genuinely new data to fetch.
            return chunk_is_up_to_date(path, hi) if not complete else os.path.exists(path)

        jobs = [job for job in jobs if not already_cached(job)]
        if n_all != len(jobs):
            print(
                f"{n_all - len(jobs)} of {n_all} chunk(s) already cached and will be "
                f"skipped (use --overwrite to refetch).\n"
            )

    if not jobs:
        print("Cache is already complete for the requested locations and period.")
        return 0

    est_calls = sum(
        api_call_weight(
            (pd.Timestamp(hi) - pd.Timestamp(lo)).days + 1, len(ARCHIVE_VARIABLES)
        )
        for _, lo, hi, _, _ in jobs
    )

    # Pace requests so the run stays under the hourly limit. The per-minute limit is looser
    # than the hourly one at these request sizes, so the hourly limit sets the pace.
    pace = args.pace
    if pace is None:
        heaviest = max(
            api_call_weight(
                (pd.Timestamp(hi) - pd.Timestamp(lo)).days + 1, len(ARCHIVE_VARIABLES)
            )
            for _, lo, hi, _, _ in jobs
        )
        pace = round(3600 * heaviest / HOURLY_CALL_LIMIT, 1)
    print(
        f"Building weather cache at {cache_dir}\n"
        f"  model      : {ARCHIVE_MODEL}\n"
        f"  period     : {args.start} to {end}\n"
        f"  locations  : {len(args.locations)} ({', '.join(args.locations)})\n"
        f"  variables  : {len(ARCHIVE_VARIABLES)}\n"
        f"  chunks     : {len(jobs)} still to fetch\n"
        f"  est. cost  : {est_calls:,.0f} weighted API calls "
        f"(free tier allows {DAILY_CALL_LIMIT:,}/day)\n"
        f"  pace       : {pace}s between requests\n"
    )
    if est_calls > 0.9 * DAILY_CALL_LIMIT:
        print(
            "This exceeds one day of free-tier budget. The run will stop when the daily\n"
            "limit is hit; re-run it tomorrow and it will resume at the first missing\n"
            "chunk. Narrow the work with --start / --locations to fit a single day.\n"
        )

    written = 0
    total_rows = 0
    for n, (location, lo, hi, block, _) in enumerate(jobs, start=1):
        print(f"[{n}/{len(jobs)}] {location} {lo}..{hi}")
        try:
            dataset = fetch_with_retry(
                locations=location,
                variables=ARCHIVE_VARIABLES,
                start_date=lo,
                end_date=hi,
                add_sun=False,
            )
        except Exception as exc:  # noqa: BLE001 - the SDK wraps every failure in one type
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print(
                f"\nStopped after {written} chunk(s). Re-run the script to resume at this "
                f"chunk; everything already written is kept."
            )
            return 1

        # Sun position is derived from coordinates and timestamp, so it is recomputed on
        # read rather than stored: it would grow the cache for no benefit.
        df = dataset.to_dataframe().reset_index()
        df["datetime"] = pd.to_datetime(df["date"]) + df["time"]
        df = df.drop(columns=["date", "time"])

        n_missing = int(df[ARCHIVE_VARIABLES].isna().sum().sum())
        path = write_chunk(chunk_path(cache_dir, location, block), df)
        written += 1
        total_rows += len(df)
        flag = f"  WARNING: {n_missing} missing values" if n_missing else ""
        print(
            f"  {len(df):>7,} rows  {os.path.getsize(path) / 1e6:>6.1f} MB{flag}"
        )

        if pace and n < len(jobs):
            time.sleep(pace)

    print(f"\nDone. {written} chunk(s), {total_rows:,} rows written to {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
