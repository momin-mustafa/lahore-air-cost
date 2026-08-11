"""06 — Reconstruct the continuous daily PM2.5 series for Lahore.

For every day 2018-01 → present, predict PM2.5 with the trained model and
attach CQR-calibrated 90% intervals. The final series prefers observation
when the reference monitor reported, model prediction otherwise:

  pm25_final = pm25_obs (reference monitor)  if available
             = pm25_pred (model)             otherwise

Includes an **independent out-of-sample check**: after Feb 2025 the reference
monitor is dark, but the low-cost sensor network median exists — we report
agreement between the model and that series, which the model never trained on.

Output: data/processed/pm25_reconstructed.parquet (+ CSV twin)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import DATA_PROCESSED, MODELS


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "model_frame.parquet")
    bundle = pickle.load(open(MODELS / "best_model.pkl", "rb"))
    qbundle = pickle.load(open(MODELS / "quantile_models.pkl", "rb"))
    conf = json.load(open(MODELS / "conformal.json"))
    adj = conf["conformal_adjustment_ugm3"]

    X = df[bundle["features"]]
    pred = bundle["model"].predict(X)
    lo = qbundle["models"][0.05].predict(X) - adj
    hi = qbundle["models"][0.95].predict(X) + adj

    out = pd.DataFrame({
        "date": df.date,
        "pm25_obs": df.pm25,
        "pm25_pred": np.clip(pred, 0, None),
        "pm25_lo": np.clip(lo, 0, None),
        "pm25_hi": np.clip(hi, 0, None),
        "pm25_lcs_median": df.pm25_lcs_median,
        "smog_season": df.smog_season,
    })
    out["pm25_final"] = out.pm25_obs.fillna(out.pm25_pred)
    out["source"] = np.where(out.pm25_obs.notna(), "monitor", "model")

    path = DATA_PROCESSED / "pm25_reconstructed.parquet"
    out.to_parquet(path, index=False)
    out.to_csv(DATA_PROCESSED / "pm25_reconstructed.csv", index=False)

    # --- honesty checks -----------------------------------------------------
    n_obs = (out.source == "monitor").sum()
    print(f"{len(out)} days: {n_obs} observed, {len(out) - n_obs} model-filled")

    tail = out[(out.date > "2025-02-18") & out.pm25_lcs_median.notna()]
    if len(tail):
        r = tail.pm25_pred.corr(tail.pm25_lcs_median)
        bias = (tail.pm25_pred - tail.pm25_lcs_median).mean()
        print(f"post-monitor check vs low-cost network median "
              f"({len(tail)} days, never used in training): r={r:.3f}, "
              f"mean bias={bias:+.1f} µg/m³")

    ann = (out.set_index("date").pm25_final
           .resample("YE").agg(["mean", "count"]).round(1))
    ann.index = ann.index.year
    print("annual means (µg/m³):")
    print(ann.to_string())


if __name__ == "__main__":
    main()
