"""02 — Pull daily satellite + meteorology zonal means for Lahore from GEE.

For each product, computes the daily mean over the Lahore District polygon
(tiny exports — one number per day — to respect the EECU quota).

The pull is **idempotent and resumable**: each (product, year) is
checkpointed to data/raw/gee_parts/<name>_<year>.csv and skipped on re-run.
Re-run until it prints DONE, then it merges to one CSV per product.

Products:
  - Sentinel-5P OFFL L3: NO2, absorbing aerosol index, CO, SO2, HCHO
  - MODIS MCD19A2 MAIAC AOD @ 1 km (Optical_Depth_047, scaled ×0.001)
  - ERA5-Land daily aggregates: t2m, dewpoint, wind u/v, precip, pressure
  - ERA5 boundary-layer height if present in the GEE catalog (probed at
    runtime; skipped + logged if not)
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import BOUNDARY_SHP, DATA_RAW, DATE_START
from utils.gee import init_gee

PARTS = DATA_RAW / "gee_parts"

# (collection, band, scale_m, output name)
PRODUCTS = [
    ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", 7000, "no2"),
    ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index", 7000, "aer_ai"),
    ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density", 7000, "co"),
    ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", 7000, "so2"),
    ("COPERNICUS/S5P/OFFL/L3_HCHO", "tropospheric_HCHO_column_number_density", 7000, "hcho"),
    ("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_047", 1000, "aod"),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "temperature_2m", 11132, "t2m"),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "dewpoint_temperature_2m", 11132, "d2m"),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "u_component_of_wind_10m", 11132, "wind_u"),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "v_component_of_wind_10m", 11132, "wind_v"),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "total_precipitation_sum", 11132, "precip"),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "surface_pressure", 11132, "pressure"),
]
SCALE_FACTORS = {"aod": 0.001}  # MODIS MAIAC integer scaling


def region_from_shapefile() -> ee.Geometry:
    gdf = gpd.read_file(BOUNDARY_SHP)
    return ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)


def year_zonal_means(coll: str, band: str, scale: int, name: str,
                     region: ee.Geometry, a: str, b: str,
                     hour_window: tuple[int, int] | None = None) -> pd.DataFrame:
    """Daily mean of band over region for [a, b) — one getInfo round-trip.

    hour_window (UTC) restricts to part of the day, e.g. (5, 9) ≈ 10:00–14:00
    Lahore time — used for boundary-layer height at satellite overpass.
    """
    ic = ee.ImageCollection(coll).select(band)

    def _day(day0):
        day0 = ee.Date(day0)
        if hour_window:
            imgs = ic.filterDate(day0.advance(hour_window[0], "hour"),
                                 day0.advance(hour_window[1], "hour"))
        else:
            imgs = ic.filterDate(day0, day0.advance(1, "day"))
        val = ee.Algorithms.If(
            imgs.size().gt(0),
            imgs.mean().reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=scale,
                maxPixels=1e9, bestEffort=True,
            ).get(band),
            None,
        )
        return ee.Feature(None, {"date": day0.format("YYYY-MM-dd"), name: val})

    n_days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days
    days = ee.List.sequence(0, n_days - 1).map(lambda d: ee.Date(a).advance(d, "day"))
    fc = ee.FeatureCollection(days.map(_day))
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows)
    if name in SCALE_FACTORS:
        df[name] = df[name] * SCALE_FACTORS[name]
    return df


def blh_product() -> tuple | None:
    """ERA5 boundary-layer height isn't in ERA5-Land; probe the catalog."""
    try:
        bands = ee.ImageCollection("ECMWF/ERA5/HOURLY").first().bandNames().getInfo()
        b = "boundary_layer_height" if "boundary_layer_height" in bands else None
        if b:
            return ("ECMWF/ERA5/HOURLY", b, 27830, "blh")
    except Exception as e:  # noqa: BLE001
        print(f"BLH probe failed ({e})")
    return None


def merge(products: list[tuple]) -> None:
    for _, _, _, name in products:
        parts = sorted(PARTS.glob(f"{name}_[0-9]*.csv"))
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values("date")
        out = DATA_RAW / f"gee_{name}_daily.csv"
        df.to_csv(out, index=False)
        print(f"  {name}: {len(df)} days, {df[name].notna().mean():.0%} non-missing → {out.name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=DATE_START)
    p.add_argument("--end", default=dt.date.today().isoformat())
    args = p.parse_args()

    PARTS.mkdir(parents=True, exist_ok=True)
    init_gee()
    region = region_from_shapefile()

    products = list(PRODUCTS)
    blh = blh_product()
    if blh:
        products.append(blh)
        # daytime BLH at satellite-overpass hours (05–09 UTC ≈ 10:00–14:00 PKT):
        # the physically right normaliser for column AOD → surface PM2.5
        products.append((blh[0], blh[1], blh[2], "blh_day"))
    else:
        print("NOTE: no ERA5 boundary-layer height in GEE — skipping "
              "(documented in Limitations)")

    y0, y1 = int(args.start[:4]), int(args.end[:4])
    todo_done = 0
    for coll, band, scale, name in products:
        for year in range(y0, y1 + 1):
            a = max(args.start, f"{year}-01-01")
            b = min(args.end, f"{year + 1}-01-01")
            if a >= b:
                continue
            part = PARTS / f"{name}_{year}.csv"
            if part.exists():
                continue
            hw = (5, 9) if name == "blh_day" else None
            df = year_zonal_means(coll, band, scale, name, region, a, b,
                                  hour_window=hw)
            df.to_csv(part, index=False)
            todo_done += 1
            print(f"  pulled {name} {year}: {df[name].notna().sum()}/{len(df)} days")

    print("DONE pulling — merging")
    merge(products)


if __name__ == "__main__":
    main()
