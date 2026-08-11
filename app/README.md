# Streamlit app — deploy notes

Interactive companion to *The Price of Lahore's Air*: a date explorer for the
reconstructed PM2.5 series and a live scenario cost calculator.

## Runs on precomputed artifacts only

The app **never calls Google Earth Engine or any API at runtime** (GEE needs
service-account auth and has compute quotas — unsuitable for a public app). It
reads a handful of small committed files in `app/data/`:

- `pm25_reconstructed.parquet` — the daily series with prediction intervals
- `app_inputs.json` — population, baseline deaths by age, econ parameters
- `cost_summary.json` — headline numbers
- `maps_s5p_seasonal.png`, `calendar_heatmap.png` — static figures

Regenerate them by running the pipeline (`src/04`–`src/08`); the health-cost
script writes `app_inputs.json`, and copies land in `app/data/`.

## Local run

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (public).
2. On share.streamlit.io → **New app**, point at this repo, main file
   `app/app.py`.
3. Streamlit auto-installs `app/requirements.txt`. No secrets needed — the app
   uses no API keys.
4. The `app/data/` artifacts are committed, so the deployed app is fully
   self-contained.
