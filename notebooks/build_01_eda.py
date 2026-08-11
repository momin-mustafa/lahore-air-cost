"""Builds and executes notebooks/01_eda.ipynb (helper, not part of pipeline)."""

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

HERE = Path(__file__).resolve().parent

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = [
    md("""# 01 — Exploratory Data Analysis
*The Price of Lahore's Air — satellite reconstruction of PM2.5.*

This notebook inspects the merged daily model frame before any modeling:
ground-truth coverage, missingness of satellite inputs, the seasonal cycle,
and the AOD–PM2.5 relationship that the model will exploit.

**Ground truth honesty note:** Lahore has exactly one reference-grade public
monitor (US Consulate, via OpenAQ), covering May 2019 – Feb 2025 with ~21%
of days missing. 81 low-cost sensors (AirGradient/Clarity) exist from late
2023 onward; their daily median is kept as an *independent comparison series*,
never as a training target."""),
    code("""import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
FIG = '../figures'

df = pd.read_parquet('../data/processed/model_frame.parquet')
df['year'] = df.date.dt.year
print(df.shape)
df[['date','pm25','pm25_lcs_median','aod','no2','aer_ai','t2m','blh',
    'fires_east','fires_west']].describe().T.round(2)"""),

    md("## 1. Ground-truth coverage — what we actually have"),
    code("""fig, ax = plt.subplots(figsize=(11, 3.2))
ax.scatter(df.date, df.pm25, s=3, color='#c0392b', label='US Consulate reference monitor')
ax.scatter(df.date, df.pm25_lcs_median, s=3, color='#2980b9', alpha=.5,
           label='Median of low-cost sensor network (n≤81)')
ax.set(ylabel='PM2.5 (µg/m³)', title='Daily PM2.5 ground truth, Lahore — two very different records')
ax.legend(loc='upper left', markerscale=3)
ax.annotate('reference monitor\\nends Feb 2025', xy=(pd.Timestamp('2025-02-18'), 400),
            fontsize=8, color='#c0392b')
fig.text(.01,-.04,'Source: OpenAQ v3. Reference monitor = BAM-1020 (StateAir); low-cost = AirGradient/Clarity.',
         fontsize=7, color='gray')
plt.tight_layout(); plt.savefig(f'{FIG}/eda_ground_truth_coverage.png', bbox_inches='tight')
plt.show()

lab = df.pm25.notna()
print(f"labeled days: {lab.sum()} ({df.loc[lab,'date'].min():%Y-%m-%d} → {df.loc[lab,'date'].max():%Y-%m-%d})")
both = df.dropna(subset=['pm25','pm25_lcs_median'])
print(f"overlap days (ref & low-cost median): {len(both)}, r = {both.pm25.corr(both.pm25_lcs_median):.3f}")"""),

    md("""The two records agree well where they overlap, which is what lets us use
the low-cost network median later as an *out-of-sample* sanity check for
reconstructed 2025–26 values (after the reference monitor went dark)."""),

    md("## 2. Missingness of satellite inputs"),
    code("""sat_cols = ['pm25','aod','no2','aer_ai','co','so2','hcho','t2m','blh']
miss = df.set_index('date')[sat_cols].notna().astype(int)
fig, ax = plt.subplots(figsize=(11, 3))
ax.imshow(miss.T.values, aspect='auto', cmap='Greys_r', interpolation='none',
          extent=[mdates.date2num(df.date.min()), mdates.date2num(df.date.max()), len(sat_cols)-.5, -.5])
ax.xaxis_date(); ax.set_yticks(range(len(sat_cols))); ax.set_yticklabels(sat_cols)
ax.set_title('Data availability by day (white = present, black = missing)')
ax.grid(False)
fig.text(.01,-.02,'S5P gaps = clouds/orbit; AOD gaps = clouds. ERA5 met is complete.', fontsize=7, color='gray')
plt.tight_layout(); plt.savefig(f'{FIG}/eda_missingness.png', bbox_inches='tight'); plt.show()
print(df[sat_cols].notna().mean().round(2).to_string())"""),

    md("## 3. The seasonal cycle — Lahore's smog season"),
    code("""m = df.dropna(subset=['pm25']).groupby(df.date.dt.month).pm25.agg(['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
m.columns = ['median','q25','q75']
fig, ax = plt.subplots(figsize=(8,3.5))
ax.plot(m.index, m['median'], 'o-', color='#c0392b')
ax.fill_between(m.index, m.q25, m.q75, alpha=.25, color='#c0392b', label='IQR')
ax.axhline(15, ls='--', lw=1, color='green');  ax.text(12.1, 15, 'Pakistan NEQS (annual)', fontsize=7, color='green', va='center')
ax.axhline(5, ls='--', lw=1, color='blue');    ax.text(12.1, 5, 'WHO guideline', fontsize=7, color='blue', va='center')
ax.set(xticks=range(1,13), xlabel='month', ylabel='PM2.5 (µg/m³)',
       title='Monthly PM2.5 distribution, Lahore (reference monitor, 2019–2025)')
ax.legend()
fig.text(.01,-.04,'Source: OpenAQ. Oct–Jan shaded = smog season.', fontsize=7, color='gray')
ax.axvspan(9.5,12.5,alpha=.08,color='gray'); ax.axvspan(.5,1.5,alpha=.08,color='gray')
plt.tight_layout(); plt.savefig(f'{FIG}/eda_seasonal_cycle.png', bbox_inches='tight'); plt.show()
print('smog-season median:', df[df.smog_season==1].pm25.median(), '| rest:', df[df.smog_season==0].pm25.median())"""),

    md("## 4. AOD ↔ PM2.5 — the core relationship"),
    code("""d = df.dropna(subset=['pm25','aod','blh'])
fig, axes = plt.subplots(1, 2, figsize=(11,4))
axes[0].scatter(d.aod, d.pm25, s=6, alpha=.35, c=d.smog_season, cmap='coolwarm')
axes[0].set(xlabel='MAIAC AOD (0.47 µm)', ylabel='PM2.5 (µg/m³)',
            title=f'AOD vs PM2.5 (r = {d.aod.corr(d.pm25):.2f}) — colour = smog season')
sc = axes[1].scatter(d.aod_over_blh, d.pm25, s=6, alpha=.35, c=d.smog_season, cmap='coolwarm')
axes[1].set(xlabel='AOD ÷ boundary-layer height (km)', ylabel='PM2.5 (µg/m³)',
            title=f'BLH-normalised AOD (r = {d.aod_over_blh.corr(d.pm25):.2f})')
fig.text(.01,-.03,'Source: MODIS MCD19A2, ERA5, OpenAQ. Column AOD maps to surface PM2.5 more strongly when the mixed layer is shallow.',
         fontsize=7, color='gray')
plt.tight_layout(); plt.savefig(f'{FIG}/eda_aod_pm25.png', bbox_inches='tight'); plt.show()"""),

    md("## 5. Fires and smog-season PM2.5"),
    code("""d = df.dropna(subset=['pm25'])
smog = d[d.smog_season==1]
fig, ax = plt.subplots(figsize=(8,3.5))
ax.scatter(smog.fires_east_roll3, smog.pm25, s=8, alpha=.4, label='east of Lahore (Indian Punjab)', color='#8e44ad')
ax.scatter(smog.fires_west_roll3, smog.pm25, s=8, alpha=.4, label='west of Lahore (Pakistani Punjab)', color='#16a085')
ax.set(xscale='symlog', xlabel='3-day mean VIIRS fire detections (wider Punjab box)',
       ylabel='PM2.5 (µg/m³)', title='Smog-season PM2.5 vs upwind crop-fire activity')
ax.legend(fontsize=8)
fig.text(.01,-.04,'Source: NASA FIRMS VIIRS S-NPP (n/h confidence). Symlog x-axis.', fontsize=7, color='gray')
plt.tight_layout(); plt.savefig(f'{FIG}/eda_fires.png', bbox_inches='tight'); plt.show()
print('corr (smog season): east_roll3', smog.pm25.corr(smog.fires_east_roll3).round(2),
      '| west_roll3', smog.pm25.corr(smog.fires_west_roll3).round(2))"""),

    md("## 6. Predictor correlation matrix"),
    code("""feats = ['pm25','aod','aod_over_blh','no2','aer_ai','co','so2','hcho',
         't2m','rh','wind_speed','blh','precip','pressure',
         'fires_east_roll3','fires_west_roll3']
corr = df[feats].corr()
fig, ax = plt.subplots(figsize=(9,7.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            annot_kws={'size':7}, cbar_kws={'shrink':.8}, ax=ax)
ax.set_title('Correlation matrix — daily series, all available days')
plt.tight_layout(); plt.savefig(f'{FIG}/eda_corr_matrix.png', bbox_inches='tight'); plt.show()"""),

    md("""## Takeaways

1. **One reference monitor** with a hard stop in Feb 2025 and ~21% gaps — the
   reconstruction is genuinely useful (it fills those gaps), and the low-cost
   network gives an independent check for the post-Feb-2025 tail.
2. **Strong seasonality**: smog-season median PM2.5 is ~3–4× the summer level;
   every month's median exceeds the WHO guideline by an order of magnitude.
3. **AOD is the workhorse predictor** and BLH-normalising it visibly tightens
   the relationship — the interaction feature earns its place.
4. **Contemporaneous fire counts correlate only weakly with smog-season
   PM2.5** (r ≈ 0.03 east, −0.10 west): within the season PM2.5 is high
   regardless, and transport depends on wind and timing. Fires stay in the
   feature set (with lags), but any "crop-burning share" claim would be
   unsupported — a caveat the write-up keeps.
5. S5P gas columns are 78–94% complete; the model must tolerate missing
   satellite days (tree models handle NaN natively / via imputation flags)."""),
]

nb = nbf.v4.new_notebook(cells=cells,
                         metadata={"kernelspec": {"name": "python3",
                                                  "display_name": "Python 3",
                                                  "language": "python"}})
path = HERE / "01_eda.ipynb"
nbf.write(nb, path)
NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(HERE)}}).execute()
nbf.write(nb, path)
print("executed →", path)
