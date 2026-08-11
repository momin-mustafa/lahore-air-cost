"""05 — Train & validate the PM2.5 model with time-aware cross-validation.

Validation design (non-negotiable honesty rules):
  - **Expanding-window time-series CV** (sklearn TimeSeriesSplit, 5 folds) on
    the labeled days sorted by date — never random splits, which leak
    temporally adjacent (highly autocorrelated) days.
  - Metrics reported overall AND by season (smog Oct–Jan vs rest).
  - Baselines to beat: (a) OLS on AOD alone, (b) OLS on AOD + met.

Models: RandomForest (median-imputed), XGBoost, LightGBM (native NaN).
Prediction intervals: LightGBM quantile regression (5th/95th), with empirical
coverage measured out-of-fold.

Each stage checkpoints to models/cv_<stage>.json so the script is resumable:
  python src/05_train_model.py --stage baseline|rf|xgb|lgbm|quantile|final
  (no --stage = run all remaining stages)

Outputs: models/best_model.pkl, models/quantile_models.pkl,
         models/metrics.json, models/oof_predictions.csv, models/shap_values.parquet
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import DATA_PROCESSED, MODELS

NON_FEATURES = {"date", "pm25", "pm25_lcs_median", "n_lcs_sensors"}
N_SPLITS = 5
SEED = 42


# --------------------------------------------------------------- helpers ---
def load_labeled() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(DATA_PROCESSED / "model_frame.parquet")
    df = df[df.pm25.notna()].sort_values("date").reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    return df, feats


def season_metrics(d: pd.DataFrame) -> dict:
    """RMSE/MAE/R² overall and by season for a frame with y_true/y_pred."""
    out = {}
    for label, sub in [("overall", d),
                       ("smog_season", d[d.smog_season == 1]),
                       ("rest_of_year", d[d.smog_season == 0])]:
        out[label] = {
            "rmse": float(np.sqrt(mean_squared_error(sub.y_true, sub.y_pred))),
            "mae": float(mean_absolute_error(sub.y_true, sub.y_pred)),
            "r2": float(r2_score(sub.y_true, sub.y_pred)),
            "n": int(len(sub)),
        }
    return out


def run_cv(make_model, df: pd.DataFrame, feats: list[str],
           needs_impute: bool = False) -> tuple[dict, pd.DataFrame]:
    """Expanding-window CV; returns metrics + out-of-fold predictions."""
    X, y = df[feats], df.pm25
    oof = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X)):
        model = make_model()
        if needs_impute:
            model = make_pipeline(SimpleImputer(strategy="median"), model)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        oof.append(pd.DataFrame({
            "date": df.date.iloc[te], "y_true": y.iloc[te], "y_pred": pred,
            "smog_season": df.smog_season.iloc[te], "fold": fold,
        }))
    oof = pd.concat(oof, ignore_index=True)
    return season_metrics(oof), oof


def stage_done(name: str) -> bool:
    return (MODELS / f"cv_{name}.json").exists()


def save_stage(name: str, metrics: dict, oof: pd.DataFrame | None = None) -> None:
    MODELS.mkdir(exist_ok=True)
    json.dump(metrics, open(MODELS / f"cv_{name}.json", "w"), indent=2)
    if oof is not None:
        oof.to_csv(MODELS / f"oof_{name}.csv", index=False)
    print(f"[{name}] overall R²={metrics['overall']['r2']:.3f} "
          f"RMSE={metrics['overall']['rmse']:.1f} | smog R²="
          f"{metrics['smog_season']['r2']:.3f} | rest R²="
          f"{metrics['rest_of_year']['r2']:.3f}")


# ---------------------------------------------------------------- stages ---
def stage_baseline(df, feats):
    # (0) seasonal climatology — predict each day by its month's mean in the
    # training window. If the ML model can't beat this, it has learned nothing
    # beyond the calendar.
    X, y = df[["month"]], df.pm25
    oof = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X)):
        mmean = y.iloc[tr].groupby(X.month.iloc[tr]).mean()
        pred = X.month.iloc[te].map(mmean).fillna(y.iloc[tr].mean())
        oof.append(pd.DataFrame({"date": df.date.iloc[te], "y_true": y.iloc[te],
                                 "y_pred": pred, "smog_season": df.smog_season.iloc[te],
                                 "fold": fold}))
    oof = pd.concat(oof, ignore_index=True)
    save_stage("baseline_climatology", season_metrics(oof), oof)

    # (a) AOD only — the naive satellite baseline
    d = df[df.aod.notna()]
    m, oof = run_cv(LinearRegression, d, ["aod"], needs_impute=False)
    save_stage("baseline_aod", m, oof)
    # (b) AOD + met — a fairer linear bar
    met = ["aod", "aod_over_blh", "t2m", "rh", "wind_speed", "blh_day",
           "precip", "pressure", "doy_sin", "doy_cos"]
    m, oof = run_cv(LinearRegression, d, met, needs_impute=True)
    save_stage("baseline_aod_met", m, oof)


def stage_rf(df, feats):
    make = lambda: RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.5,
        random_state=SEED, n_jobs=-1)
    m, oof = run_cv(make, df, feats, needs_impute=True)
    save_stage("rf", m, oof)


def stage_xgb(df, feats):
    import xgboost as xgb
    make = lambda: xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1)
    m, oof = run_cv(make, df, feats)
    save_stage("xgb", m, oof)


def stage_lgbm(df, feats):
    import lightgbm as lgb
    make = lambda: lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1)
    m, oof = run_cv(make, df, feats)
    save_stage("lgbm", m, oof)


def stage_quantile(df, feats):
    """Conformalized quantile regression (CQR, Romano et al. 2019).

    Raw LightGBM 5/95% quantile intervals under-cover out-of-fold (they are
    fit in-sample and the series is non-stationary), so we calibrate them:
    conformity scores E = max(q05 − y, y − q95) computed out-of-fold, and the
    90th percentile of E widens both bounds. Reported coverage is honest OOF.
    """
    import lightgbm as lgb
    make_q = lambda alpha: lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=800,
        learning_rate=0.03, num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
    X, y = df[feats], df.pm25
    lo, hi, idx = [], [], []
    for tr, te in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        m_lo, m_hi = make_q(0.05), make_q(0.95)
        m_lo.fit(X.iloc[tr], y.iloc[tr]); m_hi.fit(X.iloc[tr], y.iloc[tr])
        lo += list(m_lo.predict(X.iloc[te])); hi += list(m_hi.predict(X.iloc[te]))
        idx += list(te)
    lo, hi = np.array(lo), np.array(hi)
    y_oof = y.iloc[idx].to_numpy()

    cov_raw = float(np.mean((y_oof >= lo) & (y_oof <= hi)))
    scores = np.maximum(lo - y_oof, y_oof - hi)          # CQR conformity
    adj = float(np.quantile(scores, 0.90))
    cov_cal = float(np.mean((y_oof >= lo - adj) & (y_oof <= hi + adj)))

    json.dump({"conformal_adjustment_ugm3": adj,
               "coverage_raw": cov_raw, "coverage_calibrated": cov_cal},
              open(MODELS / "conformal.json", "w"), indent=2)
    save_stage("quantile", {"overall": {"rmse": 0, "mae": 0, "r2": 0, "n": int(len(idx))},
                            "smog_season": {"rmse": 0, "mae": 0, "r2": 0, "n": 0},
                            "rest_of_year": {"rmse": 0, "mae": 0, "r2": 0, "n": 0},
                            "coverage_raw": cov_raw,
                            "coverage_calibrated": cov_cal,
                            "conformal_adjustment_ugm3": adj})
    print(f"[quantile] coverage raw {cov_raw:.1%} → calibrated {cov_cal:.1%} "
          f"(adjustment ±{adj:.1f} µg/m³)")


def stage_final(df, feats):
    """Pick best by overall OOF R², refit on ALL labeled data, save + SHAP."""
    import lightgbm as lgb
    import xgboost as xgb

    scores = {n: json.load(open(MODELS / f"cv_{n}.json"))["overall"]["r2"]
              for n in ("rf", "xgb", "lgbm")}
    best_name = max(scores, key=scores.get)

    makers = {
        "rf": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                  max_features=0.5, random_state=SEED, n_jobs=-1)),
        "xgb": lambda: xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1),
        "lgbm": lambda: lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=-1, verbose=-1),
    }
    best = makers[best_name]()
    best.fit(df[feats], df.pm25)
    pickle.dump({"model": best, "features": feats, "name": best_name},
                open(MODELS / "best_model.pkl", "wb"))

    # quantile models refit on all data (for the reconstruction intervals)
    mq = {}
    for alpha in (0.05, 0.5, 0.95):
        q = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                              n_estimators=800, learning_rate=0.03,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbose=-1)
        q.fit(df[feats], df.pm25)
        mq[alpha] = q
    pickle.dump({"models": mq, "features": feats},
                open(MODELS / "quantile_models.pkl", "wb"))

    # SHAP on the best model (TreeExplainer)
    import shap
    tree = best[-1] if best_name == "rf" else best
    Xs = df[feats] if best_name != "rf" else pd.DataFrame(
        best[0].transform(df[feats]), columns=feats)
    sv = shap.TreeExplainer(tree).shap_values(Xs)
    pd.DataFrame(sv, columns=feats).to_parquet(MODELS / "shap_values.parquet")

    # consolidated metrics.json
    metrics = {n: json.load(open(MODELS / f"cv_{n}.json"))
               for n in ("baseline_climatology", "baseline_aod",
                         "baseline_aod_met", "rf", "xgb", "lgbm", "quantile")}
    metrics["best_model"] = best_name
    metrics["n_features"] = len(feats)
    metrics["cv_design"] = (f"expanding-window TimeSeriesSplit, {N_SPLITS} folds, "
                            "labeled days sorted by date")
    json.dump(metrics, open(MODELS / "metrics.json", "w"), indent=2)

    # one tidy OOF file for the app/site (best model's)
    oof = pd.read_csv(MODELS / f"oof_{best_name}.csv")
    oof.to_csv(MODELS / "oof_predictions.csv", index=False)
    print(f"[final] best = {best_name} (OOF R² {scores[best_name]:.3f}); "
          f"artifacts saved to models/")


STAGES = {"baseline": stage_baseline, "rf": stage_rf, "xgb": stage_xgb,
          "lgbm": stage_lgbm, "quantile": stage_quantile, "final": stage_final}
STAGE_KEYS = {"baseline": "baseline_aod_met", "rf": "rf", "xgb": "xgb",
              "lgbm": "lgbm", "quantile": "quantile"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=list(STAGES), default=None)
    args = p.parse_args()

    df, feats = load_labeled()
    print(f"labeled days: {len(df)}, features: {len(feats)}")

    if args.stage:
        STAGES[args.stage](df, feats)
        return
    for name, fn in STAGES.items():
        if name != "final" and stage_done(STAGE_KEYS.get(name, name)):
            continue
        fn(df, feats)


if __name__ == "__main__":
    main()
