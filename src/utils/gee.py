"""Google Earth Engine helpers: headless service-account auth + daily zonal means."""

import json
import os
from pathlib import Path

import ee
from dotenv import load_dotenv


def init_gee() -> None:
    """Authenticate to GEE with the service-account key referenced in .env."""
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    key_file = root / os.environ["GEE_SERVICE_ACCOUNT_JSON"]
    project = os.environ["GEE_PROJECT_ID"]
    email = json.load(open(key_file))["client_email"]
    creds = ee.ServiceAccountCredentials(email, str(key_file))
    ee.Initialize(creds, project=project)


def daily_zonal_mean(
    collection_id: str,
    band: str,
    region: "ee.Geometry",
    start: str,
    end: str,
    scale: int,
    out_name: str | None = None,
) -> list[dict]:
    """Daily mean of `band` over `region`, one request per calendar year.

    Days with no image (orbit gap, clouds) return None for the value — we
    keep them so downstream code sees explicit missingness. Returns a list of
    {"date": "YYYY-MM-DD", out_name: value-or-None} dicts.
    """
    out_name = out_name or band
    ic = ee.ImageCollection(collection_id).select(band)

    def _day_feature(day_start):
        day_start = ee.Date(day_start)
        imgs = ic.filterDate(day_start, day_start.advance(1, "day"))
        val = ee.Algorithms.If(
            imgs.size().gt(0),
            imgs.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
            ).get(band),
            None,
        )
        return ee.Feature(
            None, {"date": day_start.format("YYYY-MM-dd"), out_name: val}
        )

    rows: list[dict] = []
    import datetime as dt

    y0, y1 = int(start[:4]), int(end[:4])
    for year in range(y0, y1 + 1):
        a = max(start, f"{year}-01-01")
        b = min(end, f"{year + 1}-01-01")
        if a >= b:
            continue
        n_days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days
        days = ee.List.sequence(0, n_days - 1).map(
            lambda d: ee.Date(a).advance(d, "day")
        )
        fc = ee.FeatureCollection(days.map(_day_feature))
        rows += [f["properties"] for f in fc.getInfo()["features"]]
    return rows
