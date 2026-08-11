"""Central configuration: paths, study area, date range, constants."""

from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

BOUNDARY_SHP = DATA_RAW / "boundary" / "lahore_district_gadm41.shp"
POP_RASTER = DATA_RAW / "population" / "pak_pd_2020_1km.tif"
GBD_CSV = (
    DATA_RAW / "mortality" / "IHME-GBD_2023_DATA-3236439e-1"
    / "IHME-GBD_2023_DATA-3236439e-1.csv"
)

# ---------------------------------------------------------- study area -----
# Lahore District (GADM 4.1 level 3, GID_3 = PAK.7.5.2_1).
# Bounds: 74.003–74.642 E, 31.250–31.741 N. Bbox padded ~0.05°.
LAHORE_CENTER = (31.5497, 74.3436)  # (lat, lon)
LAHORE_BBOX = (73.95, 31.20, 74.70, 31.80)  # (min_lon, min_lat, max_lon, max_lat)

# Wider Punjab box for fire counts — covers Pakistani Punjab (W/SW of Lahore)
# and Indian Punjab (E/SE), both major crop-residue-burning source regions.
FIRE_BBOX = (72.5, 30.0, 76.5, 33.0)

# ------------------------------------------------------------- period ------
DATE_START = "2018-01-01"   # OpenAQ ground record; S5P coverage begins mid-2018
DATE_END = None             # None → today (scripts resolve at runtime)

# Smog season: October–January
SMOG_MONTHS = {10, 11, 12, 1}

# ------------------------------------------------------------ standards ----
WHO_GUIDELINE_ANNUAL = 5.0    # µg/m³, WHO 2021 AQG
NEQS_ANNUAL = 15.0            # µg/m³, Pakistan NEQS (effective 1 Jan 2012)
NEQS_24H = 35.0               # µg/m³
