"""04 — Merge all daily sources into one model frame + engineer features.

Target: daily PM2.5 from the US Consulate reference monitor (the only
reference-grade instrument; low-cost sensors are kept in a separate column
`pm25_lcs_median` for comparison/validation, never as training target).

Features engineered:
  - calendar: day-of-year (sin/cos), month, weekday, smog-season flag
  - lags: AOD, NO2, aer_ai (1, 2, 3 days); fire counts (1–3 days)
  - rolling: 3-day means of AOD / NO2 / fires
  - met: RH from t2m+d2m (Magnus), wind speed/direction, BLH × AOD interaction
    (vertical-column AOD translates to surface PM2.5 more strongly when the
    boundary layer is shallow)

Output: data/processed/model_frame.parquet  (+ CSV twin for inspection)

Usage: python src/04_build_dataset.py [--data-dir data/raw] [--out data/processed]
(--data-dir data/sample lets the repo run end-to-end without API keys)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import DATA_PROCESSED, DATA_RAW, SMOG_MONTHS

GEE_SERIES = ["no2", "aer_ai", "co", "so2", "hcho", "aod",
              "t2m", "d2m", "wind_u", "wind_v", "precip", "pressure",
              "blh", "blh_day"]


def load_all(data_dir: Path) -> pd.DataFrame:
    # Ground truth
    g = pd.read_csv(data_dir / "ground_pm25_daily.csv", parse_dates=["date"])
    ref = (g[g.is_monitor].groupby("date").pm25.mean()
           .rename("pm25").to_frame())
    lcs = (g[~g.is_monitor].groupby("date").pm25.median()
           .rename("pm25_lcs_median").to_frame())
    n_lcs = (g[~g.is_monitor].groupby("date").pm25.size()
             .rename("n_lcs_sensors").to_frame())

    # Satellite + met
    sat = None
    for name in GEE_SERIES:
        s = pd.read_csv(data_dir / f"gee_{name}_daily.csv", parse_dates=["date"])
        sat = s if sat is None else sat.merge(s, on="date", how="outer")

    # Fires
    fires = pd.read_csv(data_dir / "fires_daily.csv", parse_dates=["date"])

    df = (sat.merge(fires, on="date", how="left")
             .merge(ref, on="date", how="left")
             .merge(lcs, on="date", how="left")
             .merge(n_lcs, on="date", how="left")
             .sort_values("date").reset_index(drop=True))
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # --- calendar ---
    doy = d.date.dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d.date.dt.month
    d["weekday"] = d.date.dt.weekday
    d["smog_season"] = d.month.isin(SMOG_MONTHS).astype(int)

    # --- met derived ---
    t, td = d.t2m - 273.15, d.d2m - 273.15  # K → °C
    es = 6.112 * np.exp(17.62 * t / (243.12 + t))       # Magnus
    e = 6.112 * np.exp(17.62 * td / (243.12 + td))
    d["rh"] = (100 * e / es).clip(0, 100)
    d["wind_speed"] = np.hypot(d.wind_u, d.wind_v)
    d["wind_dir"] = (np.degrees(np.arctan2(d.wind_u, d.wind_v)) + 360) % 360

    # AOD is a column measure; a shallow boundary layer concentrates it at the
    # surface → interaction term (AOD per km of mixing depth). Uses *daytime*
    # BLH (10:00–14:00 local ≈ satellite overpass), not the 24-h mean, which
    # is dragged down by the shallow nocturnal layer.
    d["aod_over_blh"] = d.aod / (d.blh_day / 1000.0)

    # --- lags & rolling means ---
    for col in ["aod", "no2", "aer_ai"]:
        for lag in (1, 2, 3):
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
        d[f"{col}_roll3"] = d[col].rolling(3, min_periods=1).mean()
    for col in ["fires_west", "fires_east", "fires_total"]:
        for lag in (1, 2, 3):
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
        d[f"{col}_roll3"] = d[col].rolling(3, min_periods=1).mean()

    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_RAW))
    p.add_argument("--out", default=str(DATA_PROCESSED))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(Path(args.data_dir))
    df = engineer(df)

    out = out_dir / "model_frame.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(out_dir / "model_frame.csv", index=False)

    labeled = df.pm25.notna().sum()
    print(f"model frame: {len(df)} days {df.date.min():%Y-%m-%d} → "
          f"{df.date.max():%Y-%m-%d}, {df.shape[1]} columns")
    print(f"labeled days (reference monitor): {labeled}")
    print(f"AOD availability on labeled days: "
          f"{df.loc[df.pm25.notna(), 'aod'].notna().mean():.0%}")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
