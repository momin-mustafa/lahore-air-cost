# QA Report — The Price of Lahore's Air

Verification performed at the end of the build. All checks reproducible from the
scripts in `src/` and the committed artifacts.

## 1. Independent re-derivation of the cost arithmetic — PASS

The GEMM mortality + VSL cost was re-computed from scratch in a standalone
script that deliberately does **not** import `src/utils/health.py` (formula
re-typed from Burnett et al. 2018 to catch coding errors). Results matched the
pipeline to the last significant figure:

| Quantity | Independent | Pipeline |
|---|---|---|
| Attributable deaths / yr (vs WHO 5) | 19,710 | 19,709 |
| Economic cost (PKR bn) | 408.3 | 408.3 |
| VSL Pakistan (USD) | 74,793 | 74,793 |
| Baseline NCD+LRI deaths (25+) | 73,841 | 73,841 |

## 2. Cross-validation design — PASS (no leakage)

Expanding-window `TimeSeriesSplit` (5 folds) audited fold by fold: every test
fold begins **strictly after** the last training date, and no calendar date
appears in more than one test fold. The out-of-fold prediction file has 1,380
rows, all unique dates.

```
fold 0: train ≤ 2020-03-10   test ≥ 2020-03-11   strictly-future=True
fold 1: train ≤ 2021-01-15   test ≥ 2021-01-16   strictly-future=True
fold 2: train ≤ 2022-02-23   test ≥ 2022-02-24   strictly-future=True
fold 3: train ≤ 2023-06-11   test ≥ 2023-06-12   strictly-future=True
fold 4: train ≤ 2024-04-13   test ≥ 2024-04-14   strictly-future=True
```

## 3. Feature-leakage audit — PASS

The 50 model features contain **no** target-derived columns: the reference-
monitor PM2.5 target and the low-cost-sensor median (`pm25_lcs_median`) are both
excluded from the feature set. The low-cost median is used only as an
*independent* post-2025 validation series.

## 4. End-to-end run on committed sample data — PASS

`python src/04_build_dataset.py --data-dir data/sample --out /tmp/qa` builds a
valid 92-day model frame (all engineered features present, incl.
`aod_over_blh`) with no API keys. The repo therefore runs for a reviewer who has
no credentials.

## 5. Literature reconciliation — PASS (defensible neighbourhood)

- Lahore attributable deaths (WHO counterfactual): **~19,700/yr**.
- As a share of Pakistan's national GBD-2021 PM2.5 death total (~157,800):
  **12.5%**, for a district holding ~4.5% of the national population —
  consistent with Lahore being one of the world's most polluted cities.
- The older World Bank ~22,000 outdoor-air figure for all Pakistan is now
  widely regarded as a large underestimate versus GBD, so it is reported as
  historical context rather than the primary anchor.
- Model skill (R² 0.72 overall / 0.55 smog season, blocked CV) sits at the lower
  edge of the published 0.72–0.80 range for satellite-PM2.5 models — honest for
  a single-station target.

## 6. Uncertainty is reported, not hidden — PASS

- Prediction intervals: raw quantile-GBM covered 61% out-of-fold; conformal
  (CQR) calibration widened them to an honest **90.0%** empirical coverage.
- Cost: reported as a **range** across CR-function × VSL base × income
  elasticity × baseline year (19,000–36,600 deaths; PKR 258–2,060 bn), not a
  single false-precision point estimate.

## 7. Artifact & secret hygiene — PASS

- All required model and data artifacts present (12/12).
- 17 presentation figures + 4 Sentinel-5P map GeoTIFFs.
- `.env` and the GEE service-account JSON are gitignored (verified via
  `git check-ignore`); `data/sample/` and `app/data/` are tracked so the repo
  and app are self-contained.

## Limitations named in the write-up (confirmed present)

Single-/few-station ground truth (no fabricated gridded PM2.5 surface); largest
model error during the extreme smog days that dominate exposure; coarse and
cloud-gapped satellite inputs; contestable GEMM-vs-log-linear CR and VSL
assumptions; Punjab-level (not Lahore-specific) baseline rates; fires treated as
predictive context, not causal attribution. This is an association-based burden
estimate, not a causal experiment.

## Notes / deviations from the original spec

- **Boundary corrected:** the supplied shapefile was Lahore *Division*
  (~16,100 km²); swapped to the GADM level-3 Lahore *District* (~1,840 km²) per
  confirmation, with the Division archived alongside.
- **Ground truth is richer than assumed:** 82 PM2.5 sensors were found, not one.
  The single US-Consulate reference monitor is the training target; the 81
  low-cost sensors provide an independent post-Feb-2025 check (r ≈ 0.90).
- **Boundary-layer height added** at satellite-overpass hours — the physically
  correct normaliser for column AOD → surface PM2.5 (lifts AOD–PM2.5 r from
  0.23 to 0.55).
- **GEMM parameters:** the PNAS SI / PMC were unreachable (captcha / fetch
  timeouts), so the GEMM formula, counterfactual (2.4 µg/m³) and age structure
  were verified against a peer-reviewed open-source implementation
  (github.com/lukeconibear/health_impact_assessment); the age-specific θ table
  is hard-coded as a citable constant and the whole estimate is reconciled
  against national GBD figures as the external check.
- **WAQI** deliberately unused for the core model (free tier is
  current-conditions only, no history), per the project brief.
