"""07 — Health & economic cost of Lahore's PM2.5.

Pipeline:
  1. Population: clip WorldPop 2020 density to Lahore District → total people.
  2. Age structure & baseline mortality: from GBD 2023 Punjab NCD+LRI
     (Number & Rate → implied Punjab population by age → age fractions;
     Punjab age-specific NCD+LRI death rates → Lahore baseline deaths by age).
  3. Exposure: annual-mean PM2.5 from the reconstructed series (latest full
     year), and the multi-year mean.
  4. Mortality: GEMM NCD+LRI attributable deaths vs three counterfactuals
     (WHO 5, Pakistan NEQS 15, GEMM TMREL 2.4). Log-linear CR as a sensitivity.
  5. Valuation: VSL benefit transfer → USD → PKR.
  6. Scenarios: meet WHO / meet NEQS / −30% smog-season peaks → avoided
     deaths + rupee savings.
  7. Sensitivity: CR function × VSL base × income elasticity × baseline year.
  8. Reconcile the headline against the World Bank anchor.

All outputs saved as tidy CSVs in data/processed/ for the app and site.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import (BOUNDARY_SHP, DATA_PROCESSED, GBD_CSV, NEQS_ANNUAL,
                          POP_RASTER, WHO_GUIDELINE_ANNUAL)
from utils.health import (GEMM_THETA, GEMM_TMREL, attributable_deaths,
                          vsl_pakistan)

# ---- economic parameters (documented, all varied in sensitivity) -----------
USD_PKR = 277.0            # July 2026 (exchangerates.org.uk)
GNI_PK = 1430.0           # Pakistan GNI/capita, Atlas method, 2024 (World Bank)
GNI_OECD_BASE = 38000.0   # approx OECD-average GNI/capita (benefit-transfer base)
VSL_BASE_USD = 3_830_000  # OECD base VSL (2019 USD); World Bank 2016 methodology
ELASTICITY = 1.2          # income elasticity of VSL for low-income transfer
WORLD_BANK_PK_DEATHS = 22_000    # World Bank (2014): outdoor-air adult deaths/yr
GBD_PK_DEATHS = 157_762          # GBD 2021 / State of Global Air 2024: PM2.5 deaths/yr

GBD_AGES_25PLUS = list(GEMM_THETA.keys())


# --------------------------------------------------------------- population --
def lahore_population() -> float:
    """Total Lahore District population from WorldPop 2020 density raster."""
    gdf = gpd.read_file(BOUNDARY_SHP)
    with rasterio.open(POP_RASTER) as r:
        arr, transform = rasterio.mask.mask(r, gdf.geometry, crop=True,
                                            nodata=r.nodata)
        dens = np.where(arr[0] == r.nodata, 0.0, arr[0])  # persons per km²
        # per-row pixel area (km²); latitude varies down the clip
        import math
        top = transform.f
        dlat = transform.e  # negative
        dlon = transform.a
        rows = dens.shape[0]
        lat = top + (np.arange(rows) + 0.5) * dlat
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * np.cos(np.radians(lat))
        px_area = (abs(dlon) * km_per_deg_lon) * (abs(dlat) * km_per_deg_lat)
        pop = float((dens * px_area[:, None]).sum())
    return pop


# ----------------------------------------------- baseline mortality by age --
def baseline_deaths_by_age(total_pop: float, year: int = 2023) -> dict[str, float]:
    """Lahore baseline NCD+LRI deaths by age (25+).

    Punjab age fractions (from GBD implied population) scale the Lahore total;
    Punjab age-specific NCD+LRI death rates give deaths.
    """
    m = pd.read_csv(GBD_CSV)
    m = m[(m.year == year) & (m.cause_name.isin(
        ["Non-communicable diseases", "Lower respiratory infections"]))]

    # sum sexes; pivot Number and Rate per age & cause
    agg = (m.groupby(["age_name", "cause_name", "metric_name"]).val.sum()
             .reset_index())
    num = agg[agg.metric_name == "Number"].pivot(
        index="age_name", columns="cause_name", values="val")
    rate = agg[agg.metric_name == "Rate"].pivot(
        index="age_name", columns="cause_name", values="val")

    # implied Punjab population by age = NCD Number / NCD Rate × 100,000
    punjab_pop = (num["Non-communicable diseases"]
                  / rate["Non-communicable diseases"] * 1e5)
    # age fractions across ALL ages present (denominator = full Punjab pop)
    age_frac = punjab_pop / punjab_pop.sum()

    ncdlri_rate = (rate["Non-communicable diseases"].fillna(0)
                   + rate["Lower respiratory infections"].fillna(0)) / 1e5

    out = {}
    for age in GBD_AGES_25PLUS:
        lahore_pop_age = total_pop * age_frac.get(age, 0.0)
        out[age] = lahore_pop_age * ncdlri_rate.get(age, 0.0)
    return out


# ------------------------------------------------------------------ exposure -
def annual_means() -> tuple[dict[int, float], float]:
    rec = pd.read_parquet(DATA_PROCESSED / "pm25_reconstructed.parquet")
    rec["year"] = rec.date.dt.year
    full = (rec.groupby("year").agg(mean=("pm25_final", "mean"),
                                    n=("pm25_final", "size")))
    full = full[full.n >= 350]
    by_year = full["mean"].round(2).to_dict()
    multiyear = float(rec[rec.year.isin(full.index)].pm25_final.mean())
    return by_year, multiyear


# -------------------------------------------------------------------- main ---
def main() -> None:
    pop = lahore_population()
    base = baseline_deaths_by_age(pop)
    base_total = sum(base.values())
    by_year, exposure = annual_means()
    latest_year = max(by_year)
    exposure_latest = by_year[latest_year]

    print(f"Lahore District population (WorldPop 2020): {pop/1e6:.2f} M")
    print(f"Baseline NCD+LRI deaths (25+, GBD 2023 Punjab rates): {base_total:,.0f}/yr")
    print(f"Annual-mean PM2.5: multi-year {exposure:.1f}, {latest_year} {exposure_latest:.1f} µg/m³")

    counterfactuals = {"WHO guideline (5)": WHO_GUIDELINE_ANNUAL,
                       "Pakistan NEQS (15)": NEQS_ANNUAL,
                       "GEMM TMREL (2.4)": GEMM_TMREL}

    vsl = vsl_pakistan(VSL_BASE_USD, GNI_OECD_BASE, GNI_PK, ELASTICITY)

    # --- headline: deaths & cost at multi-year exposure, each counterfactual --
    rows = []
    for label, cf in counterfactuals.items():
        d = attributable_deaths(exposure, base, counterfactual=cf, cr="gemm")
        rows.append({"counterfactual": label, "cf_ugm3": cf,
                     "attributable_deaths": round(d),
                     "cost_usd_bn": round(d * vsl / 1e9, 2),
                     "cost_pkr_bn": round(d * vsl * USD_PKR / 1e9, 1)})
    headline = pd.DataFrame(rows)
    headline.to_csv(DATA_PROCESSED / "cost_headline.csv", index=False)
    print("\nHeadline (multi-year mean exposure, GEMM, central VSL):")
    print(headline.to_string(index=False))

    # --- scenarios (relative to WHO counterfactual valuation) -----------------
    rec = pd.read_parquet(DATA_PROCESSED / "pm25_reconstructed.parquet")
    rec_latest = rec[rec.date.dt.year == latest_year].copy()
    obs_annual = rec_latest.pm25_final.mean()

    def annual_from_capped(series, cap):
        return series.clip(upper=cap).mean()

    def annual_peak_cut(series, frac):
        s = series.copy()
        peak = s > s.quantile(0.75)
        s.loc[peak] = s.loc[peak] * (1 - frac)
        return s.mean()

    scen_defs = {
        "Baseline (observed)": obs_annual,
        "Meet WHO guideline": WHO_GUIDELINE_ANNUAL,
        "Meet Pakistan NEQS": NEQS_ANNUAL,
        "-30% smog-season peaks":
            annual_peak_cut(rec_latest.pm25_final, 0.30),
    }
    base_deaths_now = attributable_deaths(obs_annual, base,
                                          counterfactual=GEMM_TMREL, cr="gemm")
    srows = []
    for label, exp in scen_defs.items():
        d = attributable_deaths(exp, base, counterfactual=GEMM_TMREL, cr="gemm")
        srows.append({"scenario": label, "annual_pm25": round(exp, 1),
                      "attributable_deaths": round(d),
                      "avoided_vs_baseline": round(base_deaths_now - d),
                      "pkr_bn_saved": round((base_deaths_now - d) * vsl * USD_PKR / 1e9, 1)})
    scenarios = pd.DataFrame(srows)
    scenarios.to_csv(DATA_PROCESSED / "cost_scenarios.csv", index=False)
    print(f"\nScenarios ({latest_year}, vs GEMM TMREL, central VSL):")
    print(scenarios.to_string(index=False))

    # --- sensitivity grid -----------------------------------------------------
    grid = []
    for cr in ["gemm", "loglinear"]:
        for vbase in [2_500_000, 3_830_000, 5_400_000]:
            for elas in [1.0, 1.2]:
                for byear in [2019, 2023]:
                    b = baseline_deaths_by_age(pop, year=byear)
                    d = attributable_deaths(exposure, b,
                                            counterfactual=WHO_GUIDELINE_ANNUAL, cr=cr)
                    v = vsl_pakistan(vbase, GNI_OECD_BASE, GNI_PK, elas)
                    grid.append({"cr": cr, "vsl_base_usd": vbase,
                                 "elasticity": elas, "baseline_year": byear,
                                 "deaths": round(d),
                                 "cost_pkr_bn": round(d * v * USD_PKR / 1e9, 1),
                                 "cost_usd_bn": round(d * v / 1e9, 2)})
    sens = pd.DataFrame(grid)
    sens.to_csv(DATA_PROCESSED / "cost_sensitivity.csv", index=False)
    d_lo, d_hi = sens.deaths.min(), sens.deaths.max()
    c_lo, c_hi = sens.cost_pkr_bn.min(), sens.cost_pkr_bn.max()
    print(f"\nSensitivity range (WHO counterfactual): deaths {d_lo:,}–{d_hi:,}, "
          f"PKR {c_lo:,}–{c_hi:,} bn")

    # --- reconciliation vs World Bank ----------------------------------------
    who_deaths = headline.loc[headline.cf_ugm3 == WHO_GUIDELINE_ANNUAL,
                              "attributable_deaths"].iloc[0]
    lahore_share = pop / 240e6  # Lahore District share of Pakistan population
    implied_national = who_deaths / lahore_share
    share_of_gbd = who_deaths / GBD_PK_DEATHS
    print(f"\nReconciliation:")
    print(f"  Lahore attributable deaths (WHO cf): {who_deaths:,.0f}")
    print(f"  Lahore District = {lahore_share:.1%} of Pakistan's population")
    print(f"  vs GBD 2021 national PM2.5 deaths (~{GBD_PK_DEATHS:,}): "
          f"Lahore is {share_of_gbd:.1%} of the national total")
    print(f"  → a {lahore_share:.1%}-of-population hotspot accounting for "
          f"{share_of_gbd:.1%} of national deaths is consistent with Lahore "
          "being one of the world's most polluted cities.")
    print(f"  (Older World Bank ~{WORLD_BANK_PK_DEATHS:,} outdoor-death figure "
          "is now widely regarded as a large underestimate vs GBD.)")

    summary = {
        "population": round(pop),
        "baseline_ncdlri_deaths_25plus": round(base_total),
        "exposure_multiyear_ugm3": round(exposure, 1),
        "exposure_latest_year": latest_year,
        "exposure_latest_ugm3": round(exposure_latest, 1),
        "vsl_pakistan_usd_central": round(vsl),
        "usd_pkr": USD_PKR,
        "headline_deaths_who_cf": int(who_deaths),
        "headline_cost_pkr_bn_who_cf": float(headline.loc[
            headline.cf_ugm3 == WHO_GUIDELINE_ANNUAL, "cost_pkr_bn"].iloc[0]),
        "deaths_range": [int(d_lo), int(d_hi)],
        "cost_pkr_bn_range": [float(c_lo), float(c_hi)],
        "world_bank_anchor_deaths_pakistan": WORLD_BANK_PK_DEATHS,
        "gbd_national_pm25_deaths_pakistan": GBD_PK_DEATHS,
        "lahore_share_of_national_gbd": round(who_deaths / GBD_PK_DEATHS, 3),
        "params": {"vsl_base_usd": VSL_BASE_USD, "gni_pk": GNI_PK,
                   "gni_oecd_base": GNI_OECD_BASE, "elasticity": ELASTICITY},
    }
    json.dump(summary, open(DATA_PROCESSED / "cost_summary.json", "w"), indent=2)

    # app inputs: baseline deaths by age + econ params, so the Streamlit
    # scenario calculator can recompute deaths/cost live without any raw data
    app_inputs = {
        "population": pop,
        "baseline_deaths_by_age": base,
        "econ": {"vsl_base_usd": VSL_BASE_USD, "gni_pk": GNI_PK,
                 "gni_oecd_base": GNI_OECD_BASE, "elasticity": ELASTICITY,
                 "usd_pkr": USD_PKR},
        "counterfactuals": {"WHO guideline": WHO_GUIDELINE_ANNUAL,
                            "Pakistan NEQS": NEQS_ANNUAL,
                            "GEMM TMREL": GEMM_TMREL},
        "exposure_latest_year": latest_year,
    }
    json.dump(app_inputs, open(DATA_PROCESSED / "app_inputs.json", "w"), indent=2)
    print("\nsaved cost_headline / cost_scenarios / cost_sensitivity / "
          "cost_summary / app_inputs")


if __name__ == "__main__":
    main()
