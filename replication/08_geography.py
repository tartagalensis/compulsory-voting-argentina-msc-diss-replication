"""08 — Is the socioeconomic gradient an artifact of where deprived circuits are?

The RD identifies tau *within* each level of circuit deprivation, but the
*difference* in tau across deprivation — the manuscript's firmest heterogeneity
result (Sec. 6.2) — compares circuits that also differ in province, rurality and
population density. Deprivation is not randomly assigned across places, so the
gradient could in principle be a between-province composition effect, or a
rurality effect operating through the cost of complying (distance to the polling
place), which Sec. 6.2 already concedes this design cannot separate from the
sanctions channel.

This script bounds that. It re-fits the published continuous specification
(src.analysis.run_ses_continuous_interaction, the 16-column
`voted ~ x * T * ses * female` design, CRV1 by circuit, inside the same
MSE-optimal window) five ways and compares Delta-tau, the change in the cutoff
effect from the 10th to the 90th deprivation percentile:

  1. unconditional                     — reproduces rdd_gradient_summary.csv
  2. + province FE x (1, x, T, Tx)     — gradient identified within province
  3. + log voter density x (1,x,T,Tx)  — holding urbanicity fixed
  4. urban half of voters only         — non-parametric version of (3)
  5. rural half of voters only         — the complement, reported for contrast

Step 1 is a hard check, not decoration: if it does not reproduce the published
Delta-tau to three decimals the augmented rows are not comparable, and the
script asserts on it.

STAGES. Building circuit_geography.csv needs the electoral-circuit GeoJSON
layer (data/raw/circuitos_2025/, reached by symlink; the layer itself is public
— Galeano's published circuit boundaries — but it is not vendored here). The
CSV it produces IS committed to replication/outputs/tables/, so the estimation
stage runs from national.parquet alone. The geometry stage is therefore guarded
on the CSV's existence: present (the normal case) it is skipped, and this script
is safe in the default Stage-B ORDER. Missing and the GeoJSONs unavailable, it
fails with an explicit message rather than silently producing nothing.

Outputs
-------
circuit_geography.csv       circuit area (km2), registered voters, density
gradient_geography_T18.csv  Delta-tau per sex x SES measure x specification,
                            plus theta1 under each specification
gradient_geography_T18.tex  the appendix table
"""
import os
import warnings
from datetime import datetime

from _common import setup, REP_TABLES  # noqa: F401
setup()


def _log(msg):
    """Timestamped, flushed progress line — localizes the failure if a run dies."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import FINAL_DIR, GEO_DIR

warnings.filterwarnings('ignore')

geo_csv    = f"{REP_TABLES}/circuit_geography.csv"
out_csv    = f"{REP_TABLES}/gradient_geography_T18.csv"
out_tex    = f"{REP_TABLES}/gradient_geography_T18.tex"
triple_csv = f"{REP_TABLES}/triple_interaction_T18.csv"
grad_csv   = f"{REP_TABLES}/rdd_gradient_summary.csv"

SES_MEASURES = ['pca_index', 'nbi']

# The window is read off the published artifact rather than re-selected, so the
# unconditional row is the same estimand as Sec. 6.2's by construction.
if not os.path.exists(triple_csv):
    raise SystemExit(f"missing {os.path.basename(triple_csv)} — run 05_triple.py first")
BW = float(pd.read_csv(triple_csv)['bw_days'].iloc[0])
_log(f"T18 window: h = {BW:.5f}d (from {os.path.basename(triple_csv)})")


# ── Stage 1 — circuit geography ──────────────────────────────────────────────
def build_circuit_geography():
    """Circuit area and registered-voter density from the circuit GeoJSONs.

    Areas are summed over a circuit's polygon parts rather than dissolved:
    `GeoDataFrame.dissolve` unions the parts and GEOS raises a side-location
    conflict on the invalid polygons present in the layer. Summing part areas
    is equivalent here because circuit parts do not overlap.
    """
    import geopandas as gpd
    from src.pipeline import PROVINCE_CONFIGS, load_circuits

    parts = []
    for pid in PROVINCE_CONFIGS:
        g = load_circuits(pid).to_crs('EPSG:5346')       # POSGAR 2007 / Arg 3, metric
        g['area_km2'] = g.geometry.area / 1e6
        a = g.groupby('circuito_key', as_index=False)['area_km2'].sum()
        a['province_id'] = pid
        parts.append(a)
    geo = pd.concat(parts, ignore_index=True)
    _log(f"circuit polygons: {len(geo):,} across {geo.province_id.nunique()} provinces")

    vp = pd.read_parquet(f"{FINAL_DIR}/national.parquet",
                         columns=['province_id', 'circuit_id'])
    vp['circuito_key'] = _circuito_key(vp['circuit_id'])
    cnt = (vp.groupby(['province_id', 'circuito_key']).size()
             .rename('n_voters').reset_index())

    m = cnt.merge(geo, on=['province_id', 'circuito_key'], how='left')
    m['density'] = m['n_voters'] / m['area_km2']
    ok = m['area_km2'].notna() & (m['area_km2'] > 0)
    _log(f"circuits in roll: {len(m):,} | matched to a polygon: {ok.sum():,} "
         f"({100 * ok.mean():.1f}%) | voters covered: "
         f"{100 * m.loc[ok, 'n_voters'].sum() / m['n_voters'].sum():.2f}%")
    m.to_csv(geo_csv, index=False)
    _log(f"Saved: {os.path.basename(geo_csv)}")


def _circuito_key(s):
    """The circuit merge key used throughout the pipeline (src/pipeline.py:108,345).

    circuit_id strings repeat across provinces, so every merge on this key must
    also carry province_id.
    """
    return s.astype(str).str.strip().str.lstrip('0').str.upper()


if os.path.exists(geo_csv):
    _log(f"resume guard: {os.path.basename(geo_csv)} exists, skipping geometry stage")
elif os.path.isdir(GEO_DIR):
    _log("building circuit geography from the circuit GeoJSON layer")
    build_circuit_geography()
else:
    raise SystemExit(
        f"missing {os.path.basename(geo_csv)} and the circuit GeoJSON layer "
        f"({GEO_DIR}) is not available. The CSV is committed to the repository; "
        "restore it, or point config.GEO_DIR at the published circuit layer.")


# ── Stage 2 — the five specifications ────────────────────────────────────────
_log("loading the T18 estimation window")
df = pd.read_parquet(
    f"{FINAL_DIR}/national.parquet",
    columns=['province_id', 'circuit_id', 'gender', 'voted',
             'days_from_18', 'nbi', 'pca_index'],
    filters=[[('days_from_18', '>=', -BW), ('days_from_18', '<=', BW)]])
d = df[df['gender'].isin(['F', 'M'])].copy()
del df

d['circuito_key'] = _circuito_key(d['circuit_id'])
geo = pd.read_csv(geo_csv)
d = d.merge(geo[['province_id', 'circuito_key', 'density']],
            on=['province_id', 'circuito_key'], how='left')
d['cluster'] = d.groupby(['province_id', 'circuito_key']).ngroup()

# area_km2 == 0 for a handful of degenerate polygons, so density is inf there.
d['logden'] = np.log(d['density'])
has_geo = np.isfinite(d['logden'])
_log(f"window rows: {len(d):,} | with usable density: {has_geo.sum():,} "
     f"({100 * has_geo.mean():.2f}%) | circuits: {d['cluster'].nunique():,} | "
     f"provinces: {d['province_id'].nunique()}")

dg = d[has_geo].copy()
dg['urban'] = (dg['logden'] > dg['logden'].median()).astype(int)   # voter-weighted
_log(f"urban share of voters in window: {dg['urban'].mean():.3f}")
for v in SES_MEASURES:
    _log(f"  corr(log density, {v}) = {np.corrcoef(dg['logden'], dg[v])[0, 1]:+.3f}")


def _design(sub, ses_var):
    """The published 16-column design: voted ~ x * T * ses * female.

    Column index is (female<<3)|(ses<<2)|(T<<1)|x_pow, so col 6 is T*ses and
    col 14 is T*ses*female — the two the contrasts below read.
    """
    x = sub['days_from_18'].values.astype(float)
    T = (x >= 0).astype(float)
    f = (sub['gender'] == 'F').astype(float).values
    s = sub[ses_var].values.astype(float)
    cols = []
    for fem in [None, f]:
        for ses in [None, s]:
            for trt in [None, T]:
                for xt in [None, x]:
                    col = np.ones_like(x)
                    for arr in (xt, trt, ses, fem):
                        if arr is not None:
                            col = col * arr
                    cols.append(col)
    return np.column_stack(cols), x, T, s


def _augment(X, x, T, sub, kind):
    """Append the conditioning block, each control fully interacted with
    (1, x, T, T*x) so the RD itself is free to differ across its levels."""
    if kind == 'province':
        provs = np.sort(sub['province_id'].unique())[1:]   # first absorbed by the base
        Z = np.column_stack([(sub['province_id'].values == p).astype(float) * blk
                             for p in provs
                             for blk in (np.ones_like(x), x, T, T * x)])
    elif kind == 'density':
        z = sub['logden'].values.astype(float)
        Z = np.column_stack([z, z * x, z * T, z * T * x])
    else:
        return X
    return np.hstack([X, Z])


def run_spec(sub, ses_var, label, kind=None):
    X, x, T, s = _design(sub, ses_var)
    X = _augment(X, x, T, sub, kind)
    p10, p90 = np.percentile(s, 10), np.percentile(s, 90)
    rng = p90 - p10
    m = sm.OLS(sub['voted'].values.astype(float), X).fit(
        cov_type='cluster', cov_kwds={'groups': sub['cluster'].values})

    rows = []
    for sex, fem in [('Men', 0.0), ('Women', 1.0)]:
        c = np.zeros(X.shape[1]); c[6] = rng; c[14] = fem * rng
        tt = m.t_test(c)
        rows.append(dict(ses_var=ses_var, spec=label, quantity='dtau_p10_p90',
                         sex=sex, coef_pp=float(np.ravel(tt.effect)[0]) * 100,
                         se_pp=float(np.ravel(tt.sd)[0]) * 100,
                         p=float(np.ravel(tt.pvalue)[0]),
                         n=int(len(sub)), n_circuits=int(sub['cluster'].nunique())))
    c = np.zeros(X.shape[1]); c[14] = 1.0
    tt = m.t_test(c)
    rows.append(dict(ses_var=ses_var, spec=label, quantity='theta1_per_unit',
                     sex='', coef_pp=float(np.ravel(tt.effect)[0]) * 100,
                     se_pp=float(np.ravel(tt.sd)[0]) * 100,
                     p=float(np.ravel(tt.pvalue)[0]),
                     n=int(len(sub)), n_circuits=int(sub['cluster'].nunique())))
    return rows


# Rows 1-2 use the full window, so row 1 reproduces the published estimate
# exactly. Rows 4-6 can only be fit where density exists (99.85% of the window),
# and row 3 is their own baseline: it prices the 319 dropped voters, which move
# NBI/Men by 0.06pp, so the geography rows are read against a like-for-like
# comparison rather than against a slightly different sample.
SPECS = [
    ('Unconditional',                     d,                 None),
    ('+ province fixed effects',          d,                 'province'),
    ('Unconditional, density subsample',  dg,                None),
    ('+ log voter density',               dg,                'density'),
    ('Urban half of voters',              dg[dg.urban == 1], None),
    ('Rural half of voters',              dg[dg.urban == 0], None),
]

rows = []
for ses_var in SES_MEASURES:
    for label, sub, kind in SPECS:
        _log(f"fitting: {ses_var} | {label}")
        rows += run_spec(sub, ses_var, label, kind)
res = pd.DataFrame(rows)

# ── Hard check: the unconditional row must reproduce the published gradient ──
pub = pd.read_csv(grad_csv)
pub = pub[pub['threshold'] == 18].set_index('ses_var')
for ses_var in SES_MEASURES:
    for sex, col in [('Men', 'cont_dtau_M_pp'), ('Women', 'cont_dtau_F_pp')]:
        got = res.query("ses_var == @ses_var and spec == 'Unconditional' "
                        "and sex == @sex")['coef_pp'].iloc[0]
        want = float(pub.loc[ses_var, col])
        assert abs(got - want) < 0.01, (
            f"unconditional {ses_var}/{sex} = {got:.4f} does not reproduce "
            f"{os.path.basename(grad_csv)}'s {want:.4f}; the augmented rows are "
            "not comparable to the published estimate")
_log("check passed: unconditional row reproduces rdd_gradient_summary.csv")

res.to_csv(out_csv, index=False)
print("\nDelta-tau, 10th -> 90th deprivation percentile (pp), by specification — T18")
print(res[res.quantity == 'dtau_p10_p90']
      .pivot_table(index='spec', columns=['ses_var', 'sex'], values='coef_pp', sort=False)
      .round(2).to_string())
_log(f"Saved: {os.path.basename(out_csv)}")


# ── LaTeX table ──────────────────────────────────────────────────────────────
def _cell(r):
    star = '' if r['p'] >= .05 else (r'^{**}' if r['p'] < .01 else r'^{*}')
    return f"${r['coef_pp']:.2f}{star}$ & ({r['se_pp']:.2f})"


dt = res[res.quantity == 'dtau_p10_p90']
lines = []
for label, _, _ in SPECS:
    cells = []
    for ses_var in SES_MEASURES:
        for sex in ('Men', 'Women'):
            r = dt.query("spec == @label and ses_var == @ses_var and sex == @sex").iloc[0]
            cells.append(_cell(r))
    n = dt.query("spec == @label").iloc[0]
    lines.append(f"{label} & " + " & ".join(cells)
                 + f" & {n['n']:,} & {n['n_circuits']:,} \\\\")

tex = r"""\begin{table}[ht]
\centering
\caption{The socioeconomic gradient in the compulsory-voting effect,
conditioning on geography (age-18 cutoff).}
\label{tab:gradient_geography}
\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%
\begin{tabular}{lrlrlrlrlrr}
\toprule
& \multicolumn{4}{c}{PCA index} & \multicolumn{4}{c}{NBI} & & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Specification & \multicolumn{2}{c}{Men} & \multicolumn{2}{c}{Women}
              & \multicolumn{2}{c}{Men} & \multicolumn{2}{c}{Women}
              & $N$ & Circuits \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}%
}
\par\vspace{0.6em}
{\footnotesize\begin{minipage}{\linewidth}\emph{Note:} Each cell is
$\Delta\tau$, the change in the estimated cutoff effect from the 10th to the
90th percentile of the deprivation measure, in percentage points, with its
standard error in parentheses. Every row is the same continuous model as
Table~\ref{tab:triple_cells_T18}, fit in the same MSE-optimal window
($\approx$66 days), with standard errors clustered by electoral circuit (CRV1).
Each conditioning variable is interacted with $(1, x, T, Tx)$, so the
discontinuity itself is free to differ across provinces and across levels of
density. Urban and rural halves split at the median log voter density, so each
holds half the voters in the window; the urban half holds them in far fewer
circuits. Density is unavailable for $0.15\%$ of the window, so the last three
rows are read against the subsample row above them rather than against the
first. The first row reproduces the estimate reported in
Section~\ref{sec:res_ses}. $^{*}p<.05$, $^{**}p<.01$.
\end{minipage}}
\end{table}
"""
with open(out_tex, 'w') as fh:
    fh.write(tex)
_log(f"Saved: {os.path.basename(out_tex)}")
