"""06 — Provincial RDD Analysis: 24 provinces x 2 thresholds (48 estimates),
provincial table, coefficient choropleth map, and stacked coefficient plot.

Transcribed from notebooks/06_provincial.ipynb (all code cells: 1, 2, 4, 6, 8,
10). The only structural change is the data source for the province loop: the
notebook reads each province's own processed parquet
(`PROCESSED_DIR/province_XX.parquet`) directly, which isn't part of the
replication package; here the same rows are obtained by filtering
`load_window(['voted'], thresholds=(18, 70), bw=BW_MAX)` (the T18 union T70
+-3y window, matching 03_gradient.py/05_triple.py) down to `province_id ==
prov_id`, then re-applying the notebook's own per-threshold +-BW_MAX filter —
identical rows, one read instead of 24. Neither `gender` nor any SES column is
touched anywhere in this notebook, so only `voted` is loaded.

`ELECTORAL_TO_INDEC` is imported from src/analysis.py (already used there by
plot_provincial_turnout for the same province_id <-> codprov_censo mapping)
rather than re-declaring the notebook's inline copy — the two dicts are
identical, this just avoids a second source of truth.

Resilience: the province loop (the expensive step, ~30-60 min for 48
rdrobust calls) is checkpointed to a pickle after every province so a
relaunch resumes mid-loop instead of restarting; the final `rdd_provincial.csv`
guards the whole loop, and each downstream artifact (LaTeX table, map,
coefplot) is independently resume-guarded on its own output file.
"""
import os
import pickle
from datetime import datetime

from _common import setup, load_window, REP_FIGURES, REP_TABLES  # noqa: F401
setup()


def _log(msg):
    """Timestamped, flushed progress line — localizes the failure if a run dies."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
from rdrobust import rdrobust

from config import GEO_PROVINCIAS
from src.analysis import ELECTORAL_TO_INDEC

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})

# Fail fast: the geojson lives outside the repo (see CLAUDE.md) — better to
# error before the ~30-60 min province loop than after it.
if not os.path.exists(GEO_PROVINCIAS):
    raise FileNotFoundError(
        f"Province boundaries GeoJSON not found at GEO_PROVINCIAS={GEO_PROVINCIAS}. "
        "Set GEO_PROVINCIAS in config.py to the local path of "
        "provincias_simplified.geojson (see config_template.py)."
    )

# ── Province names (electoral ID -> display name; cell 2, verbatim) ────────
PROVINCE_NAMES = {
    1:  'CABA',         2:  'Buenos Aires',  3:  'Catamarca',
    4:  'Córdoba',      5:  'Corrientes',    6:  'Chaco',
    7:  'Chubut',       8:  'Entre Ríos',    9:  'Formosa',
    10: 'Jujuy',        11: 'La Pampa',      12: 'La Rioja',
    13: 'Mendoza',      14: 'Misiones',      15: 'Neuquén',
    16: 'Río Negro',    17: 'Salta',         18: 'San Juan',
    19: 'San Luis',     20: 'Santa Cruz',    21: 'Santa Fe',
    22: 'Stgo. del Estero', 23: 'Tucumán',  24: 'Tierra del Fuego',
}


def extract_rdd_prov(rdd_obj, province_id, province_name, threshold):
    """Extract key scalars from an rdrobust result for provincial estimates."""
    return {
        'province_id':   province_id,
        'province':      province_name,
        'threshold':     threshold,
        'coef':          round(float(rdd_obj.coef.iloc[0, 0]), 4),
        'se':            round(float(rdd_obj.se.iloc[0, 0]),   4),
        'p_value':       round(float(rdd_obj.pv.iloc[0, 0]),   6),
        'robust_ci_low': round(float(rdd_obj.ci.iloc[2, 0]),   4),
        'robust_ci_high':round(float(rdd_obj.ci.iloc[2, 1]),   4),
        'bw':            round(float(rdd_obj.bws.iloc[0, 0]),  1),
        'n_eff_left':    int(rdd_obj.N_h[0]),
        'n_eff_right':   int(rdd_obj.N_h[1]),
    }


# ── 2. Province Loop — RDD Estimation (cell 4) ──────────────────────────────
BW_MAX = 365 * 3
MIN_OBS = 200  # minimum effective N per side to attempt rdrobust

prov_csv = f"{REP_TABLES}/rdd_provincial.csv"
partial_pkl = f"{REP_TABLES}/_partial_rdd_provincial.pkl"

if os.path.exists(prov_csv):
    _log(f"resume guard: {os.path.basename(prov_csv)} exists, skipping province loop")
    df_prov = pd.read_csv(prov_csv)
else:
    if os.path.exists(partial_pkl):
        with open(partial_pkl, 'rb') as fh:
            _ck = pickle.load(fh)
        results, skipped, done_provinces = _ck['results'], _ck['skipped'], _ck['done_provinces']
        _log(f"resume: loaded checkpoint — {len(results)} estimates, "
             f"{len(done_provinces)}/{len(PROVINCE_NAMES)} provinces already attempted")
    else:
        results, skipped, done_provinces = [], [], set()

    remaining = [p for p in PROVINCE_NAMES if p not in done_provinces]
    if remaining:
        _log(f"starting province loop ({len(remaining)} provinces remaining)")
        df = load_window(['voted'], thresholds=(18, 70), bw=BW_MAX)
        _log(f"Records (T18 union T70 +-3y window): {len(df):,}")
    else:
        df = None

    for prov_id, prov_name in PROVINCE_NAMES.items():
        if prov_id in done_provinces:
            continue

        d = df[df['province_id'] == prov_id]
        if d.empty:
            print(f"  [{prov_id:02d}] {prov_name}: no records in window, skipping")
            skipped.append(prov_id)
            done_provinces.add(prov_id)
            with open(partial_pkl, 'wb') as fh:
                pickle.dump({'results': results, 'skipped': skipped,
                             'done_provinces': done_provinces}, fh)
            continue

        d18 = d[d['days_from_18'].between(-BW_MAX, BW_MAX)]
        d70 = d[d['days_from_70'].between(-BW_MAX, BW_MAX)]

        for threshold, dd in [(18, d18), (70, d70)]:
            rv = f'days_from_{threshold}'
            n_left  = (dd[rv] < 0).sum()
            n_right = (dd[rv] >= 0).sum()

            if n_left < MIN_OBS or n_right < MIN_OBS:
                print(f"  [{prov_id:02d}] {prov_name} T{threshold}: too few obs ({n_left}|{n_right}), skipping")
                skipped.append((prov_id, threshold))
                continue

            try:
                r = rdrobust(y=dd['voted'].values, x=dd[rv].values, c=0, masspoints='rd')
                res = extract_rdd_prov(r, prov_id, prov_name, threshold)
                results.append(res)
                print(f"  [{prov_id:02d}] {prov_name} T{threshold}: coef={res['coef']*100:+.2f}pp  se={res['se']*100:.2f}  bw={res['bw']:.0f}d")
            except Exception as e:
                print(f"  [{prov_id:02d}] {prov_name} T{threshold}: ERROR — {e}")
                skipped.append((prov_id, threshold))

        done_provinces.add(prov_id)
        with open(partial_pkl, 'wb') as fh:
            pickle.dump({'results': results, 'skipped': skipped,
                         'done_provinces': done_provinces}, fh)
        _log(f"checkpoint: [{prov_id:02d}] {prov_name} done "
             f"({len(results)} estimates so far, {len(done_provinces)}/{len(PROVINCE_NAMES)} provinces)")

    print(f"\nCompleted: {len(results)} estimates across {len(PROVINCE_NAMES)} provinces x 2 thresholds")
    if skipped:
        print(f"Skipped:   {skipped}")

    df_prov = pd.DataFrame(results)
    df_prov.to_csv(prov_csv, index=False)
    _log(f"Saved: {os.path.basename(prov_csv)}")
    if os.path.exists(partial_pkl):
        os.remove(partial_pkl)
        _log("removed province-loop checkpoint (run completed successfully)")

# ── 3. Provincial Estimates Table (cell 6) ──────────────────────────────────
prov_tex = f"{REP_TABLES}/rdd_provincial_latex.tex"
if os.path.exists(prov_tex):
    _log(f"resume guard: {os.path.basename(prov_tex)} exists, skipping table export")
else:
    _log("starting provincial estimates table export")
    disp = df_prov.copy()
    disp['coef_pp']   = disp['coef'].mul(100).apply(lambda x: f"{x:+.2f}")
    disp['se_pp']     = disp['se'].mul(100).apply(lambda x: f"{x:.2f}")
    ci_lo = disp['robust_ci_low'].mul(100).apply(lambda x: f"{x:.2f}")
    ci_hi = disp['robust_ci_high'].mul(100).apply(lambda x: f"{x:.2f}")
    disp['ci']        = '[' + ci_lo + ', ' + ci_hi + ']'
    disp['sig']       = disp['p_value'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

    cols = ['province', 'threshold', 'coef_pp', 'se_pp', 'ci', 'sig', 'bw', 'n_eff_left', 'n_eff_right']
    print(disp[cols].to_string(index=False))

    latex = disp[cols].rename(columns={
        'province': 'Province', 'threshold': 'Threshold',
        'coef_pp': 'Coef. (pp)', 'se_pp': 'SE (pp)', 'ci': 'Robust 95% CI',
        'sig': '', 'bw': 'BW', 'n_eff_left': 'N left', 'n_eff_right': 'N right'
    }).to_latex(index=False, escape=True,
        caption='Province-level RDD Estimates of the Effect of Compulsory Voting on Turnout',
        label='tab:rdd_provincial',
        column_format='lrrrrlrrr'
    )
    # Keep the manual float/overflow fixes ([ht], centering, resizebox) across
    # regenerations — same rationale as rdd_results_latex.tex in 02_main_rdd.py
    # (the curated outputs/tables/ artifact carries them; the appendix \input's
    # this table, so a re-run must not clobber the working form).
    latex = latex.replace("\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n", 1)
    latex = latex.replace("\\begin{tabular}{lrrrrlrrr}",
                          "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{lrrrrlrrr}", 1)
    latex = latex.replace("\\end{tabular}\n", "\\end{tabular}%\n}\n", 1)
    with open(prov_tex, 'w') as fout:
        fout.write(latex)
    _log(f"Saved: {os.path.basename(prov_tex)}")

# ── 4. Coefficient Maps T18/T70 (cell 8) ────────────────────────────────────
map_png = f"{REP_FIGURES}/provincial_rdd_map.png"
if os.path.exists(map_png):
    _log(f"resume guard: {os.path.basename(map_png)} exists, skipping map")
else:
    _log("starting provincial coefficient choropleth (T18, T70)")
    prov_gdf = gpd.read_file(GEO_PROVINCIAS)
    # Add electoral province_id via inverse mapping
    indec_to_electoral = {v: k for k, v in ELECTORAL_TO_INDEC.items()}
    prov_gdf['province_id'] = prov_gdf['codprov_censo'].map(indec_to_electoral)
    prov_gdf['province']    = prov_gdf['province_id'].map(PROVINCE_NAMES)
    print(f"Province polygons loaded: {len(prov_gdf)}")

    # Side-by-side choropleth for T18 and T70
    fig, axes = plt.subplots(1, 2, figsize=(14, 16))

    for ax, threshold, label in [
        (axes[0], 18, 'Threshold 18\n(voting becomes compulsory)'),
        (axes[1], 70, 'Threshold 70\n(voting becomes voluntary)'),
    ]:
        df_t = df_prov[df_prov['threshold'] == threshold][['province_id', 'coef']].copy()
        df_t['coef_pp'] = df_t['coef'] * 100
        map_t = prov_gdf.merge(df_t, on='province_id', how='left')

        map_t.plot(
            column='coef_pp', ax=ax,
            cmap='RdYlGn', legend=True,
            missing_kwds={'color': 'lightgrey', 'label': 'No estimate'},
            legend_kwds={'label': 'RDD Estimate (pp)', 'shrink': 0.55, 'pad': 0.01},
            edgecolor='white', linewidth=0.4,
        )
        ax.set_xlim(-74, -52)
        ax.set_ylim(-56, -21)
        ax.set_axis_off()
        ax.set_title(label, fontsize=12)

    fig.suptitle(
        'Effect of Compulsory Voting on Turnout by Province\nArgentina 2025 (pp)',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(map_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    _log(f"Saved: {os.path.basename(map_png)}")

# ── 5. Coefficient Plots T18/T70 (cell 10) ──────────────────────────────────
coefplot_png = f"{REP_FIGURES}/provincial_rdd_coefplot.png"
if os.path.exists(coefplot_png):
    _log(f"resume guard: {os.path.basename(coefplot_png)} exists, skipping coefplot")
else:
    _log("starting provincial coefficient plot (T18, T70)")
    fig, axes = plt.subplots(2, 1, figsize=(8, 20))

    for ax, threshold, subtitle in [
        (axes[0], 18, 'Threshold 18 — voting becomes compulsory'),
        (axes[1], 70, 'Threshold 70 — voting becomes voluntary'),
    ]:
        df_t = (
            df_prov[df_prov['threshold'] == threshold]
            .sort_values('coef', ascending=True)
            .copy()
        )
        df_t['coef_pp']  = df_t['coef'] * 100
        df_t['ci_lo_pp'] = df_t['robust_ci_low'] * 100
        df_t['ci_hi_pp'] = df_t['robust_ci_high'] * 100
        df_t['sig']      = df_t['p_value'] < 0.05

        colors = ['#2980b9' if s else '#bdc3c7' for s in df_t['sig']]
        ax.barh(df_t['province'], df_t['coef_pp'], color=colors, alpha=0.85)
        ax.errorbar(
            df_t['coef_pp'], df_t['province'],
            xerr=[
                df_t['coef_pp'] - df_t['ci_lo_pp'],
                df_t['ci_hi_pp'] - df_t['coef_pp'],
            ],
            fmt='none', color='black', linewidth=0.8, capsize=3
        )
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('RDD Estimate (percentage points)')
        ax.set_title(subtitle, fontsize=11)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle(
        'Effect of Compulsory Voting on Turnout by Province — Argentina 2025',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(coefplot_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    _log(f"Saved: {os.path.basename(coefplot_png)}")

print("\n06_provincial: DONE")
