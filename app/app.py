"""The Price of Lahore's Air — interactive scenario app.

Reads ONLY precomputed artifacts committed under app/data/ (parquet + JSON +
PNG). It never calls Google Earth Engine or any API at runtime — all satellite
work is done offline in the pipeline. Deploy notes in app/README.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

# --- GEMM (self-contained copy so the app has no src/ dependency) ------------
GEMM_ALPHA, GEMM_MU, GEMM_NU, GEMM_TMREL = 1.6, 15.5, 36.8, 2.4
GEMM_THETA = {
    "25-29 years": 0.1430, "30-34 years": 0.1362, "35-39 years": 0.1310,
    "40-44 years": 0.1222, "45-49 years": 0.1151, "50-54 years": 0.1069,
    "55-59 years": 0.1002, "60-64 years": 0.0935, "65-69 years": 0.0850,
    "70-74 years": 0.0759, "75-79 years": 0.0673, "80+ years": 0.0591,
}


def gemm_af(pm25, theta):
    z = max(0.0, pm25 - GEMM_TMREL)
    rr = np.exp(theta * np.log(1 + z / GEMM_ALPHA)
                / (1 + np.exp((GEMM_MU - z) / GEMM_NU)))
    return 1 - 1 / rr


def attributable_deaths(pm25, base_by_age, counterfactual=GEMM_TMREL):
    return sum(base * (gemm_af(pm25, GEMM_THETA[age]) - gemm_af(counterfactual, GEMM_THETA[age]))
               for age, base in base_by_age.items())


def vsl_pk(base, gni_base, gni_pk, elas):
    return base * (gni_pk / gni_base) ** elas


@st.cache_data
def load():
    rec = pd.read_parquet(DATA / "pm25_reconstructed.parquet")
    inp = json.load(open(DATA / "app_inputs.json"))
    summary = json.load(open(DATA / "cost_summary.json"))
    return rec, inp, summary


rec, inp, summary = load()
base_by_age = inp["baseline_deaths_by_age"]
econ = inp["econ"]

st.set_page_config(page_title="The Price of Lahore's Air", layout="wide")
st.title("The Price of Lahore's Air")
st.caption("Satellite-reconstructed daily PM2.5 for Lahore, and what it costs "
           "in lives and rupees. Reads precomputed artifacts only — no live "
           "satellite calls.")

c1, c2, c3 = st.columns(3)
c1.metric("Annual-mean PM2.5", f"{summary['exposure_multiyear_ugm3']:.0f} µg/m³",
          f"{summary['exposure_multiyear_ugm3']/5:.0f}× the WHO guideline",
          delta_color="inverse")
c2.metric("Attributable deaths / yr", f"~{summary['headline_deaths_who_cf']:,}",
          "vs WHO guideline")
c3.metric("Economic cost / yr", f"~PKR {summary['headline_cost_pkr_bn_who_cf']:.0f} bn",
          f"~USD {summary['headline_cost_pkr_bn_who_cf']/econ['usd_pkr']*1000/1000:.1f} bn")

tab1, tab2, tab3 = st.tabs(["📈 Explore the series",
                            "🧮 Scenario calculator", "🗺️ Seasonal maps"])

# ---------------------------------------------------------------- explorer ---
with tab1:
    st.subheader("Daily PM2.5 — observed & reconstructed")
    years = sorted(rec.date.dt.year.unique())
    yr = st.select_slider("Year", years, value=years[-2])
    d = rec[rec.date.dt.year == yr].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.date, y=d.pm25_hi, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.date, y=d.pm25_lo, fill="tonexty",
                             fillcolor="rgba(241,196,15,0.25)", line=dict(width=0),
                             name="90% interval", hoverinfo="skip"))
    obs = d[d.source == "monitor"]
    fig.add_trace(go.Scatter(x=obs.date, y=obs.pm25_obs, mode="markers",
                             marker=dict(size=4, color="#c0392b"), name="observed"))
    fig.add_trace(go.Scatter(x=d.date, y=d.pm25_pred, line=dict(width=1.2, color="#2c3e50"),
                             name="model"))
    fig.add_hline(y=5, line_dash="dash", line_color="green",
                  annotation_text="WHO 5")
    fig.add_hline(y=15, line_dash="dash", line_color="blue",
                  annotation_text="NEQS 15")
    fig.update_layout(height=430, yaxis_title="PM2.5 (µg/m³)",
                      legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"{yr}: annual mean "
               f"{d.pm25_final.mean():.0f} µg/m³ · "
               f"{(d.pm25_final>35).mean():.0%} of days above the NEQS 24-h limit (35).")

# ------------------------------------------------------------- calculator ----
with tab2:
    st.subheader("What would cleaner air buy?")
    left, right = st.columns([1, 1.3])
    with left:
        mode = st.radio("Set the counterfactual annual PM2.5",
                        ["Meet WHO guideline (5)", "Meet Pakistan NEQS (15)",
                         "Custom target", "% cut to smog-season peaks"])
        latest = summary["exposure_latest_ugm3"]
        rec_latest = rec[rec.date.dt.year == inp["exposure_latest_year"]]
        if mode == "Meet WHO guideline (5)":
            target = 5.0
        elif mode == "Meet Pakistan NEQS (15)":
            target = 15.0
        elif mode == "Custom target":
            target = st.slider("Target annual-mean PM2.5 (µg/m³)", 2.4, float(latest), 25.0)
        else:
            cut = st.slider("Cut to days above the 75th percentile (%)", 0, 100, 30)
            s = rec_latest.pm25_final.copy()
            peak = s > s.quantile(0.75)
            s.loc[peak] = s.loc[peak] * (1 - cut / 100)
            target = float(s.mean())
            st.caption(f"→ implied annual mean: {target:.1f} µg/m³")

        st.markdown("**Valuation assumptions**")
        vbase = st.select_slider("VSL base (USD)",
                                 [2_500_000, 3_830_000, 5_400_000], 3_830_000,
                                 format_func=lambda x: f"${x/1e6:.1f}M")
        elas = st.slider("Income elasticity of VSL", 0.8, 1.4, float(econ["elasticity"]), 0.1)

    with right:
        d_now = attributable_deaths(latest, base_by_age, GEMM_TMREL)
        d_target = attributable_deaths(target, base_by_age, GEMM_TMREL)
        avoided = d_now - d_target
        vsl = vsl_pk(vbase, econ["gni_oecd_base"], econ["gni_pk"], elas)
        pkr_saved = avoided * vsl * econ["usd_pkr"] / 1e9

        m1, m2 = st.columns(2)
        m1.metric("Lives saved / yr", f"~{avoided:,.0f}")
        m2.metric("Saved / yr", f"~PKR {pkr_saved:,.0f} bn",
                  f"~USD {avoided*vsl/1e9:,.1f} bn")

        fig = go.Figure(go.Bar(
            x=["Current burden", "After scenario", "Avoided"],
            y=[d_now, d_target, avoided],
            marker_color=["#c0392b", "#7f8c8d", "#27ae60"],
            text=[f"{v:,.0f}" for v in [d_now, d_target, avoided]],
            textposition="outside"))
        fig.update_layout(height=340, yaxis_title="attributable deaths / yr",
                          margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Current exposure {latest:.0f} µg/m³ → target {target:.1f} µg/m³. "
                   "GEMM NCD+LRI, deaths vs the GEMM TMREL (2.4). Valuation is a "
                   "benefit transfer; see the article's Limitations.")

# --------------------------------------------------------------- maps --------
with tab3:
    st.subheader("Sentinel-5P — smog season vs summer")
    st.image(str(DATA / "maps_s5p_seasonal.png"), width='stretch')
    st.caption("Gridded satellite columns (NO₂, absorbing aerosol index) — "
               "spatial context. These are *not* a ground-level PM2.5 map; the "
               "validated PM2.5 model is city-level (single ground station).")
    st.image(str(DATA / "calendar_heatmap.png"), width='stretch')

st.divider()
st.caption("Sources: OpenAQ · Sentinel-5P/MODIS/ERA5 via Google Earth Engine · "
           "NASA FIRMS · WorldPop · GBD 2023 (IHME) · Pakistan NEQS. "
           "Method & limitations in the accompanying article.")
