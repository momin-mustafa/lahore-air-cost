"""08 — Export gridded Sentinel-5P seasonal maps over Lahore/Punjab from GEE.

These are legitimate gridded products (no ground truth needed) — used only for
spatial *context* visuals, never presented as a gridded PM2.5 surface.

Exports mean NO2 and absorbing-aerosol-index rasters for two seasons over a
Lahore-centred box, as GeoTIFFs in figures/maps/ (via getDownloadURL for small
regions — no Drive/Tasks needed). Idempotent: skips existing files.
"""

import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import FIGURES
from utils.gee import init_gee

# Lahore-centred context box (wider than the district, for the map)
BOX = [73.6, 30.8, 75.1, 32.2]  # W, S, E, N
SEASONS = {
    "smog_2023_OctJan": ("2023-10-01", "2024-02-01"),
    "summer_2023": ("2023-05-01", "2023-08-01"),
}
LAYERS = {
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2",
            "tropospheric_NO2_column_number_density", 7000),
    "aai": ("COPERNICUS/S5P/OFFL/L3_AER_AI",
            "absorbing_aerosol_index", 7000),
}
OUT = FIGURES / "maps"


def export(coll: str, band: str, scale: int, a: str, b: str,
           region: ee.Geometry, dest: Path) -> None:
    img = (ee.ImageCollection(coll).select(band).filterDate(a, b).mean()
           .clip(region))
    url = img.getDownloadURL({"scale": scale, "region": region,
                              "format": "GEO_TIFF", "crs": "EPSG:4326"})
    data = urllib.request.urlopen(url, timeout=120).read()
    if data[:2] == b"PK":  # zipped
        z = zipfile.ZipFile(BytesIO(data))
        tif = [n for n in z.namelist() if n.endswith(".tif")][0]
        dest.write_bytes(z.read(tif))
    else:
        dest.write_bytes(data)
    print(f"  saved {dest.name} ({len(data)//1024} KB)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    init_gee()
    region = ee.Geometry.Rectangle(BOX)
    for season, (a, b) in SEASONS.items():
        for layer, (coll, band, scale) in LAYERS.items():
            dest = OUT / f"s5p_{layer}_{season}.tif"
            if dest.exists():
                print(f"  skip {dest.name}")
                continue
            export(coll, band, scale, a, b, region, dest)


if __name__ == "__main__":
    main()
