"""03 — SES Gradient RDD: per-decile table, summary, per-sex/gap/marginal
figures, and cluster-robustness appendix table (Table 5).

Transcribed from notebooks/07_gradient.ipynb (all code cells in scope, per
the task map): parameters (cell 2), the §3 orchestration loop (cell 6:
run_ses_decile_rdd + run_ses_continuous_interaction + ses_gradient_test per
SES measure x threshold), the combined-table CSV/summary/LaTeX export (cell
8), the per-sex / gap / marginal-effect figures (cells 10/12/14), and §8e
clustering robustness for the SES gradient (cell 22, LaTeX float writer
copied verbatim). Out of scope (Task 5): §8/§8c/§8d — the gender-gap RD
visual, its robustness forest plot, and the canonical global F-M gap number.

Data load: cell 4 of the notebook reads the full national.parquet for a
national SES describe() (exploratory — skipped here) and then narrows to the
T18 union T70 +-3y window before any RD loop. load_window(['voted', 'gender',
'nbi', 'pca_index']) performs the identical narrowing via parquet predicate
pushdown (see _common.py) and is memory-safe by construction, so it replaces
cell 4 in full.

Resilience: the §3 orchestration loop is the expensive part (40 decile RDs +
4 continuous interactions across 4 SES x threshold combinations) — each
combination is checkpointed to a pickle on completion and reloaded on
restart. §8e (gradient_clustering_robustness) re-runs a comparable amount of
work internally (HC1 vs cluster-by-circuit, x2 SES measures) as a single
function call from src/analysis.py (not modified here, called verbatim), so
it is guarded at the block level: skipped entirely if its final CSV/TEX
already exist. Figure cells are guarded per output file.
"""
import gc
import os
import pickle
from datetime import datetime

from _common import setup, load_window, REP_FIGURES, REP_TABLES  # noqa: F401
setup()


def _log(msg):
    """Timestamped, flushed progress line — localizes the failure if a run dies."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.analysis import (
    run_ses_decile_rdd,
    run_ses_continuous_interaction,
    ses_gradient_test,
    gradient_clustering_robustness,
    plot_decile_rdd_per_sex,
    plot_decile_rdd_gap,
    plot_marginal_cutoff_continuous,
)

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})

# ── PARAMETERS (cell 2, verbatim) ───────────────────────────────────────────
N_DECILES    = 10              # population-weighted SES quantiles
BANDWIDTH    = 365 * 3         # pre-filter window in days (3 years, matches 04/06)
POLY_DEGREE  = 1               # polynomial degree (rdrobust default = 1; matches 04)
ALPHA_LEVEL  = 0.10            # 90% CI for post-cutoff level estimates

SES_MEASURES = ['nbi', 'pca_index']
THRESHOLDS   = [18, 70]

_log(f"Parameters: N_DECILES={N_DECILES}, BANDWIDTH={BANDWIDTH}d, POLY_DEGREE={POLY_DEGREE}")

warnings.filterwarnings('ignore')

# ── Data load ────────────────────────────────────────────────────────────
df = load_window(['voted', 'gender', 'nbi', 'pca_index'])
_log(f"Records (windowed union of T18/T70 +-3y): {len(df):,}")

# ── §3 Orchestration loop — NBI x {T18, T70} and PCA x {T18, T70} ─────────
# (RESUME GUARD: each (ses, threshold) combination is checkpointed to a
# pickle after it completes; a relaunch reloads finished combinations
# instead of recomputing 10 decile rdrobust calls + 1 continuous-interaction
# OLS per combination.)
decile_results     = {}   # {(ses, threshold): df_decile}
continuous_results = {}   # {(ses, threshold): dict}
gradient_results    = {}  # {(ses, threshold): dict}
boundaries_store    = {}

for ses in SES_MEASURES:
    for t in THRESHOLDS:
        partial_path = f"{REP_TABLES}/_partial_gradient_{ses}_T{t}.pkl"
        if os.path.exists(partial_path):
            with open(partial_path, 'rb') as fh:
                cached = pickle.load(fh)
            decile_results[(ses, t)]     = cached['decile_df']
            boundaries_store[(ses, t)]   = cached['bounds']
            continuous_results[(ses, t)] = cached['cont']
            gradient_results[(ses, t)]   = cached['grad']
            _log(f"resume guard: [{ses.upper()} x T{t}] loaded from checkpoint")
            continue

        _log(f"[{ses.upper()} x T{t}] starting decile RDD ({N_DECILES} deciles)")
        df_dec, bounds = run_ses_decile_rdd(
            df, ses_var=ses, threshold=t,
            n_deciles=N_DECILES, bandwidth=BANDWIDTH,
            alpha_level=ALPHA_LEVEL,
        )
        decile_results[(ses, t)]   = df_dec
        boundaries_store[(ses, t)] = bounds
        _log(f"[{ses.upper()} x T{t}] decile RDD done")

        cont = run_ses_continuous_interaction(
            df, ses_var=ses, threshold=t,
            bandwidth=None,                 # MSE-optimal from unconditional rdrobust
            poly_degree=POLY_DEGREE,
            bandwidth_max=BANDWIDTH,
        )
        continuous_results[(ses, t)] = cont
        _log(f"[{ses.upper()} x T{t}] continuous interaction done")

        grad = ses_gradient_test(df_dec)
        gradient_results[(ses, t)] = grad

        with open(partial_path, 'wb') as fh:
            pickle.dump({'decile_df': df_dec, 'bounds': bounds,
                        'cont': cont, 'grad': grad}, fh)
        _log(f"checkpoint: [{ses.upper()} x T{t}] -> {os.path.basename(partial_path)}")

_log(f"Completed: {len(decile_results)} (SES x threshold) settings x {N_DECILES} deciles "
     f"= {len(decile_results) * N_DECILES} decile RDs")

# ── §4 Combined results table ──────────────────────────────────────────────
# ── Per-decile table (wide format) ──────────────────────────────────────────
df_gradient = pd.concat(decile_results.values(), ignore_index=True)
front = ['ses_var', 'threshold', 'decile', 'n_voters']
cols = front + [c for c in df_gradient.columns if c not in front]
df_gradient = df_gradient[cols]
df_gradient.to_csv(f"{REP_TABLES}/rdd_gradient.csv", index=False)
_log(f"Saved: rdd_gradient.csv ({len(df_gradient)} rows)")

# ── Gradient + continuous summary ───────────────────────────────────────────
summary_rows = []
for (ses, t), grad in gradient_results.items():
    cont = continuous_results[(ses, t)]
    summary_rows.append({
        'ses_var':  ses,
        'threshold': t,
        # Gradient (per decile, IV-WLS)
        'grad_M_slope_pp':    grad['tau_M_slope']  * 100,
        'grad_M_slope_se_pp': grad['tau_M_slope_se'] * 100,
        'grad_M_total_pp':    grad['tau_M_total_pp'] * 100,
        'grad_M_p':           grad['tau_M_p'],
        'grad_F_slope_pp':    grad['tau_F_slope']  * 100,
        'grad_F_slope_se_pp': grad['tau_F_slope_se'] * 100,
        'grad_F_total_pp':    grad['tau_F_total_pp'] * 100,
        'grad_F_p':           grad['tau_F_p'],
        'grad_gap_slope_pp':    grad['gap_MF_slope']  * 100,
        'grad_gap_slope_se_pp': grad['gap_MF_slope_se'] * 100,
        'grad_gap_total_pp':    grad['gap_MF_total_pp'] * 100,
        'grad_gap_p':           grad['gap_MF_p'],
        # Continuous interaction (p10 → p90)
        'cont_bw_d':         cont['bandwidth'],
        'cont_p10':          cont['p10'],
        'cont_p90':          cont['p90'],
        'cont_dtau_M_pp':    cont['dtau_M']    * 100,
        'cont_dtau_M_se_pp': cont['se_dtau_M'] * 100,
        'cont_dtau_F_pp':    cont['dtau_F']    * 100,
        'cont_dtau_F_se_pp': cont['se_dtau_F'] * 100,
        'cont_dgap_pp':      cont['dgap_MF']   * 100,
        'cont_dgap_se_pp':   cont['se_dgap_MF'] * 100,
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{REP_TABLES}/rdd_gradient_summary.csv", index=False)
_log(f"Saved: rdd_gradient_summary.csv ({len(df_summary)} rows)")

# ── LaTeX (compact per-decile presentation) ─────────────────────────────────
def _fmt_pp(v, se=None):
    if pd.isna(v):
        return '--'
    s = f"{v * 100:+.2f}"
    if se is not None and pd.notna(se):
        s += f"\\,({se * 100:.2f})"
    return s


tex_rows = []
for _, r in df_gradient.iterrows():
    tex_rows.append({
        'SES':       r['ses_var'].upper(),
        'Threshold': f"T{int(r['threshold'])}",
        'Decile':    int(r['decile']),
        'N':         f"{int(r['n_voters']):,}",
        'tau_F (pp)':  _fmt_pp(r['tau_F'], r['se_F']),
        'tau_M (pp)':  _fmt_pp(r['tau_M'], r['se_M']),
        'M-F gap':     _fmt_pp(r['gap_MF'], r['se_gap']),
        'p (M-F)':     f"{r['p_gap']:.3f}" if pd.notna(r['p_gap']) else '--',
        'Post F (pp)': _fmt_pp(r['post_F']),
        'Post M (pp)': _fmt_pp(r['post_M']),
    })
df_tex = pd.DataFrame(tex_rows)
latex_str = df_tex.to_latex(
    index=False, escape=False,
    caption='SES Gradient RDD Estimates by Decile, Sex, and Threshold --- Argentina 2025',
    label='tab:rdd_gradient',
    column_format='llrrrrrrrr'
)
# Keep the manual float/overflow fixes ([ht], centering, resizebox) that the
# curated outputs/ artifact carries — same rationale as rdd_results_latex.tex
# in 02_main_rdd.py (they were once lost by re-running the notebook cell).
latex_str = latex_str.replace("\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n", 1)
latex_str = latex_str.replace("\\begin{tabular}{llrrrrrrrr}",
                              "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{llrrrrrrrr}", 1)
latex_str = latex_str.replace("\\end{tabular}\n", "\\end{tabular}%\n}\n", 1)
with open(f"{REP_TABLES}/rdd_gradient_latex.tex", 'w') as fout:
    fout.write(latex_str)
_log("Saved: rdd_gradient_latex.tex")

print('\n--- Gradient + continuous summary ---')
summary_pretty = df_summary.copy()
for c in summary_pretty.columns:
    if summary_pretty[c].dtype == float:
        summary_pretty[c] = summary_pretty[c].round(3)
print(summary_pretty.to_string(index=False))

# The orchestration-loop checkpoints are kept on disk past this point (cheap
# to keep) — useful if this script is re-run after a partial failure further
# down (figures / §8e), and removed only once the whole run has succeeded.

# ── §5 Per-sex RD effect across deciles (with 95% CIs) ──────────────────────
for ses in SES_MEASURES:
    fname = f'rdd_gradient_per_sex_{ses}.png'
    if os.path.exists(f"{REP_FIGURES}/{fname}"):
        _log(f"resume guard: {fname} exists, skipping")
        continue
    plot_decile_rdd_per_sex(
        {18: decile_results[(ses, 18)], 70: decile_results[(ses, 70)]},
        ses_var=ses, figures_dir=REP_FIGURES,
    )
    _log(f"{fname} written")

# ── §6 M−F Gap in RD effect across deciles ──────────────────────────────────
for ses in SES_MEASURES:
    fname = f'rdd_gradient_gap_{ses}.png'
    if os.path.exists(f"{REP_FIGURES}/{fname}"):
        _log(f"resume guard: {fname} exists, skipping")
        continue
    plot_decile_rdd_gap(
        {18: decile_results[(ses, 18)], 70: decile_results[(ses, 70)]},
        ses_var=ses, figures_dir=REP_FIGURES,
    )
    _log(f"{fname} written")

# ── §7 Marginal Cutoff Effect across Continuous SES ─────────────────────────
for ses in SES_MEASURES:
    fname = f'rdd_gradient_marginal_{ses}.png'
    if os.path.exists(f"{REP_FIGURES}/{fname}"):
        _log(f"resume guard: {fname} exists, skipping")
        continue
    plot_marginal_cutoff_continuous(
        {18: continuous_results[(ses, 18)], 70: continuous_results[(ses, 70)]},
        ses_var=ses, figures_dir=REP_FIGURES,
    )
    _log(f"{fname} written")

# ── §8e Clustering robustness for the SES gradient (Moulton) ───────────────
# SES is constant within a circuit, so default RD SEs on the gradient terms
# can be understated. Compare HC1/unclustered vs cluster-by-circuit (CRV1).
# Point estimates are invariant; only SEs / p-values change. Exports CSV + a
# LaTeX float for the appendix (Table 5).
# (RESUME GUARD: skip the whole block — a single function call internally
# re-running a comparable amount of rdrobust/OLS work — if both final
# artifacts already exist.)
if (os.path.exists(f"{REP_TABLES}/gradient_cluster_robustness_T18.csv")
        and os.path.exists(f"{REP_TABLES}/gradient_cluster_robustness_T18.tex")):
    _log("resume guard: gradient_cluster_robustness_T18.{csv,tex} exist, skipping §8e")
else:
    _log("starting §8e cluster robustness (Moulton) for the SES gradient")
    cluster_rob = gradient_clustering_robustness(
        df, ses_measures=SES_MEASURES, threshold=18,
        n_deciles=N_DECILES, bandwidth=BANDWIDTH)
    print(cluster_rob.round(4).to_string(index=False))
    cluster_rob.to_csv(f"{REP_TABLES}/gradient_cluster_robustness_T18.csv", index=False)
    _log("cluster robustness estimation done")

    # Compact LaTeX float for the appendix: baseline vs clustered SE/p per quantity.
    _lab = {'dtau_M(p10->p90)':  r'$\Delta\tau_M$ (p10$\to$p90)',
            'dtau_F(p10->p90)':  r'$\Delta\tau_F$ (p10$\to$p90)',
            'dgap_MF(p10->p90)': r'$\Delta$(M$-$F) gap (p10$\to$p90)',
            'tau_M_slope':       r'Decile slope, men',
            'tau_F_slope':       r'Decile slope, women',
            'gap_MF_slope':      r'Decile slope, M$-$F gap'}
    _ses = {'nbi': 'NBI', 'pca_index': 'PCA'}
    _order = ['dtau_M(p10->p90)', 'dtau_F(p10->p90)', 'dgap_MF(p10->p90)',
              'tau_M_slope', 'tau_F_slope', 'gap_MF_slope']
    _rows = []
    for _s in SES_MEASURES:
        for _q in _order:
            _sub = cluster_rob[(cluster_rob.ses == _s) & (cluster_rob.quantity == _q)]
            _b = _sub[_sub.estimator.isin(['HC1', 'unclustered'])].iloc[0]
            _c = _sub[_sub.estimator == 'cluster'].iloc[0]
            _rows.append({'SES': _ses[_s], 'Quantity': _lab[_q],
                          'Est. (pp)': f"{_b.estimate_pp:+.2f}",
                          'SE base': f"{_b.se_pp:.2f}", '$p$ base': f"{_b.p:.3f}",
                          'SE clust.': f"{_c.se_pp:.2f}", '$p$ clust.': f"{_c.p:.3f}"})
    _texdf = pd.DataFrame(_rows)
    _tabular = _texdf.to_latex(index=False, escape=False, column_format='llrrrrr')
    # Caption matches the compiled thesis Table 5 (the outputs/tables/
    # artifact), whose wording was polished after the notebook cell's export —
    # the manuscript, not the notebook cell, is the authority for float text.
    _cap = ("Robustness of the SES gradient to clustering standard errors by electoral "
            "circuit (the Moulton concern, since deprivation is constant within a "
            "circuit), age-18 threshold. Point estimates are invariant to the variance "
            "estimator; only standard errors and $p$-values change. `Base' denotes HC1 "
            "for the continuous $\\Delta\\tau$(p10$\\to$p90) terms and unclustered for "
            "the inverse-variance decile slope test; `clust.' is cluster-robust (CRV1) "
            "by circuit. Clustering moves every standard error by at most a few per "
            "cent and overturns no inference.")
    with open(f"{REP_TABLES}/gradient_cluster_robustness_T18.tex", "w") as _fh:
        _fh.write("\\begin{table}[ht]\n\\centering\n\\caption{" + _cap + "}\n"
                  "\\label{tab:cluster_robustness}\n" + _tabular + "\\end{table}\n")
    _log("Saved: gradient_cluster_robustness_T18.csv / .tex")

# Orchestration-loop checkpoints are no longer needed now that every
# downstream artifact (tables + figures + §8e) has been produced.
for ses in SES_MEASURES:
    for t in THRESHOLDS:
        p = f"{REP_TABLES}/_partial_gradient_{ses}_T{t}.pkl"
        if os.path.exists(p):
            os.remove(p)
_log("removed orchestration-loop checkpoints (run completed successfully)")

print("\n03_gradient: DONE")
