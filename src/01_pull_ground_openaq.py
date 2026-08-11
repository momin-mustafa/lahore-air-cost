"""01 — Pull ground-truth PM2.5 for Lahore from OpenAQ v3.

Finds all PM2.5 sensors within 25 km of central Lahore and pulls their
**daily** aggregates (server-side /sensors/{id}/days endpoint). Lahore has one
reference-grade monitor (US Diplomatic Post, since 2016) and a larger network
of low-cost sensors (AirGradient/Clarity, mostly 2023+). All are pulled; the
modeling target is the reference monitor — see the write-up.

The pull is **idempotent and resumable**: each sensor is checkpointed to
data/raw/openaq_parts/<sensor_id>.csv and skipped on re-run. Re-run until it
prints DONE, then it merges to data/raw/ground_pm25_daily.csv.

Logs station count, coverage span, and % missing days — these numbers drive
the honesty caveats in the write-up.
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import DATA_RAW, DATE_START, LAHORE_CENTER

API = "https://api.openaq.org/v3"
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
HEADERS = {"X-API-Key": os.environ["OPENAQ_API_KEY"]}
PARTS = DATA_RAW / "openaq_parts"


def get(url: str, **params) -> dict:
    """GET with simple retry/backoff (OpenAQ rate-limits aggressively)."""
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate-limited too many times: {url}")


def find_pm25_sensors() -> list[dict]:
    """All PM2.5 sensors on stations within 25 km of central Lahore (cached)."""
    cache = PARTS / "_sensors.json"
    if cache.exists():
        return json.load(open(cache))
    lat, lon = LAHORE_CENTER
    js = get(f"{API}/locations", coordinates=f"{lat},{lon}",
             radius=25_000, limit=1000)
    sensors = []
    for loc in js["results"]:
        for s in loc.get("sensors", []):
            if s["parameter"]["name"] == "pm25":
                sensors.append({
                    "sensor_id": s["id"],
                    "location_id": loc["id"],
                    "location": loc["name"],
                    "provider": loc["provider"]["name"],
                    "is_monitor": loc["isMonitor"],
                })
    json.dump(sensors, open(cache, "w"))
    return sensors


def pull_daily(sensor: dict, date_from: str, date_to: str) -> pd.DataFrame:
    """Daily aggregates for one sensor, paginated."""
    rows, page = [], 1
    while True:
        js = get(f"{API}/sensors/{sensor['sensor_id']}/days",
                 date_from=date_from, date_to=date_to, limit=1000, page=page)
        for rec in js["results"]:
            rows.append({
                "date": rec["period"]["datetimeFrom"]["local"][:10],
                "pm25": rec["value"],
                "n_hours": (rec.get("coverage") or {}).get("observedCount"),
                "sensor_id": sensor["sensor_id"],
                "location": sensor["location"],
                "provider": sensor["provider"],
                "is_monitor": sensor["is_monitor"],
            })
        if len(js["results"]) < 1000:
            break
        page += 1
    return pd.DataFrame(rows)


def merge() -> None:
    files = sorted(PARTS.glob("[0-9]*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # QC: physically implausible values out; reference monitor needs >=18 h/day
    n0 = len(df)
    df = df[(df.pm25 > 0) & (df.pm25 < 1500)]
    df = df[~(df.is_monitor & (df.n_hours.fillna(0) < 18))]
    print(f"QC: dropped {n0 - len(df)} of {n0} sensor-days")

    df = df.sort_values(["sensor_id", "date"])
    out = DATA_RAW / "ground_pm25_daily.csv"
    df.to_csv(out, index=False)

    ref = df[df.is_monitor]
    span = pd.to_datetime(ref.date)
    n_expected = (span.max() - span.min()).days + 1
    print(f"Reference monitor: {len(ref)} days, {ref.date.min()} → "
          f"{ref.date.max()}, {100 * (1 - len(ref) / n_expected):.1f}% of span missing")
    print(f"Low-cost sensors: {df.sensor_id.nunique() - ref.sensor_id.nunique()} "
          f"sensors, {len(df) - len(ref)} sensor-days")
    print(f"Saved {len(df)} rows → {out}")


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    date_to = dt.date.today().isoformat()
    sensors = find_pm25_sensors()
    todo = [s for s in sensors
            if not (PARTS / f"{s['sensor_id']}.csv").exists()]
    print(f"{len(sensors)} PM2.5 sensors; {len(todo)} still to pull")

    t0 = time.time()
    for s in todo:
        df = pull_daily(s, DATE_START, date_to)
        df.to_csv(PARTS / f"{s['sensor_id']}.csv", index=False)
        print(f"  sensor {s['sensor_id']} @ {s['location'][:40]}: {len(df)} days")
        if time.time() - t0 > 200:  # graceful stop; re-run to resume
            print("TIME BUDGET REACHED — re-run to resume")
            return

    print("DONE pulling — merging")
    merge()


if __name__ == "__main__":
    main()
