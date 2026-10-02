"""Record when each ECMWF IFS run becomes available on Open-Meteo.

Polls Open-Meteo's model metadata for the forecast model `predict.py` uses and appends one CSV row
per new run: its initialisation time, the time Open-Meteo made it available, and the delay between
the two. The metadata carries the exact availability time, so the polling interval only needs to be
shorter than the 6 h between runs; a missed poll (laptop asleep) loses nothing as long as the next
one comes before the following run.

Used to choose the times of the daily forecast runs (see `scripts/gce_trigger_forecast.sh`).

python scripts/measure_openmeteo_delay.py --days 5
"""

import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Same model as src.data.weather.FORECAST_MODEL; kept stdlib-only so it runs without the venv.
MODEL = "ecmwf_ifs025"
META_URL = f"https://api.open-meteo.com/data/{MODEL}/static/meta.json"
OUT = Path(__file__).resolve().parents[1] / "logs" / "openmeteo_delay" / f"{MODEL}.csv"
FIELDS = ["run_init_utc", "available_utc", "delay_h", "modified_utc", "first_seen_utc"]


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def poll(seen: set) -> None:
    with urllib.request.urlopen(META_URL, timeout=30) as r:
        meta = json.load(r)
    init = meta["last_run_initialisation_time"]
    if init in seen:
        return
    avail = meta["last_run_availability_time"]
    row = {
        "run_init_utc": _utc(init),
        "available_utc": _utc(avail),
        "delay_h": f"{(avail - init) / 3600:.2f}",
        "modified_utc": _utc(meta["last_run_modification_time"]),
        "first_seen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    new_file = not OUT.exists()
    with OUT.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    seen.add(init)
    print(row, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=5)
    parser.add_argument("--interval-min", type=float, default=15)
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if OUT.exists():
        with OUT.open() as f:
            for r in csv.DictReader(f):
                init = datetime.strptime(r["run_init_utc"], "%Y-%m-%d %H:%M:%S")
                seen.add(int(init.replace(tzinfo=timezone.utc).timestamp()))

    end = datetime.now(timezone.utc) + timedelta(days=args.days)
    while datetime.now(timezone.utc) < end:
        try:
            poll(seen)
        except Exception as e:  # network blips must not end a 5-day run
            print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} poll failed: {e}", flush=True)
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
