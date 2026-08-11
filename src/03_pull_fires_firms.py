"""03 — Pull daily active-fire counts around Lahore from NASA FIRMS.

Uses the VIIRS S-NPP *standard-processing* archive (full record; NRT only
covers recent weeks) over a wide Punjab box spanning both Pakistani Punjab
(W/SW of Lahore) and Indian Punjab (E/SE) — both major crop-residue-burning
regions. Aggregates to daily detection counts in west/east sectors relative
to Lahore, letting the model decide which matters.

The pull is **idempotent and resumable**: each 10-day chunk is checkpointed
to data/raw/firms_parts/<date>.csv and skipped on re-run. Re-run until it
prints DONE, then it merges to data/raw/fires_daily.csv.
"""

import datetime as dt
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import DATA_RAW, DATE_START, FIRE_BBOX, LAHORE_CENTER

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
KEY = os.environ["FIRMS_MAP_KEY"]
SOURCE = "VIIRS_SNPP_SP"  # standard processing = full archive
CHUNK_DAYS = 5            # FIRMS max per request for SP (archive) sources
PARTS = DATA_RAW / "firms_parts"


def sp_max_date() -> dt.date:
    """Last date covered by the SP archive (it lags ~2 months); NRT after."""
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/"
           f"{KEY}/{SOURCE}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return dt.date.fromisoformat(r.text.strip().splitlines()[-1].split(",")[-1])


def pull_chunk(start: dt.date, days: int, source: str = SOURCE) -> pd.DataFrame:
    bbox = ",".join(map(str, FIRE_BBOX))
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{KEY}/{source}/{bbox}/{days}/{start.isoformat()}")
    for attempt in range(4):
        r = requests.get(url, timeout=120)
        txt = r.text.strip()
        # FIRMS reports a temporary transaction-limit lockout as "Invalid MAP_KEY"
        if r.status_code == 200 and not txt.startswith("<"):
            if txt in ("", "No data found") or "\n" not in txt:
                return pd.DataFrame()  # header only / empty
            return pd.read_csv(io.StringIO(txt))
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"FIRMS failed for {start} ({r.status_code}): {r.text[:200]}")


def merge(start: dt.date, end: dt.date) -> None:
    lon_lahore = LAHORE_CENTER[1]
    frames = [pd.read_csv(f) for f in sorted(PARTS.glob("*.csv"))
              if f.stat().st_size > 10]  # skip empty chunks (no detections)
    fires = pd.concat([f for f in frames if len(f)], ignore_index=True)
    fires = fires[fires.confidence.isin(["n", "h"])]  # drop low confidence
    fires["date"] = pd.to_datetime(fires.acq_date).dt.date
    fires["sector"] = (fires.longitude < lon_lahore).map(
        {True: "fires_west", False: "fires_east"})

    daily = (fires.groupby(["date", "sector"]).size().unstack(fill_value=0)
             .reindex(pd.date_range(start, end).date, fill_value=0)
             .rename_axis("date").reset_index())
    for c in ("fires_west", "fires_east"):
        if c not in daily:
            daily[c] = 0
    daily["fires_total"] = daily.fires_west + daily.fires_east

    out = DATA_RAW / "fires_daily.csv"
    daily.to_csv(out, index=False)
    print(f"Saved {len(daily)} days ({daily.fires_total.sum():,} detections "
          f"after confidence filter) → {out}")


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    start = dt.date.fromisoformat(DATE_START)
    end = dt.date.today()

    sp_end = sp_max_date()
    print(f"SP archive available through {sp_end}; NRT used after")

    todo = []
    d = start
    while d <= end:
        days = min(CHUNK_DAYS, (end - d).days + 1)
        if not (PARTS / f"{d.isoformat()}.csv").exists():
            source = SOURCE if d + dt.timedelta(days=days - 1) <= sp_end else "VIIRS_SNPP_NRT"
            todo.append((d, days, source))
        d += dt.timedelta(days=days)
    print(f"{len(todo)} chunks to pull")

    def _one(job):
        day, days, source = job
        chunk = pull_chunk(day, days, source)
        chunk.to_csv(PARTS / f"{day.isoformat()}.csv", index=False)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_one, todo))

    print(f"DONE pulling ({len(todo)} new chunks) — merging")
    merge(start, end)


if __name__ == "__main__":
    main()
