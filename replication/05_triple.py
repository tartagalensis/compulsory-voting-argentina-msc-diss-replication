"""05 — CV x sex x SES triple interaction: unpacking, cell-level marginal
effects, decile/means cross-checks, and figures.

Transcribed from notebooks/08_triple_interaction.ipynb, all code cells (1, 3,
5, 7, 9, 11, 13, 15, 17, 19, 21):
  - cell 1/3: setup + data load -> replaced by _common.load_window
    (['voted','gender','nbi','pca_index'], both thresholds, +-3y windows,
    identical to the notebook's own memory-safe union-mask pattern).
  - cell 5/19: unpack_triple_interaction over SES_MEASURES at T18 and T70
    (the triple-interaction coefficient T*female*ses, cluster-robust by
    circuit) -> triple_interaction_T18.csv / triple_interaction_T70.csv.
  - cell 7: theta1 (T*female*ses alone) and the F-M gap change at the cutoff
    at p10/p90, both read off the SAME clustered ols object fit in cell 5 (no
    re-fit) -> gender_gap_ses_theta_T18.csv / gender_gap_ses_test_T18.csv.
    This is the source of the manuscript's theta1 p-values (NBI p~=.18, PCA
    p~=.11) named in the Task-6 key-number gate, so it is transcribed even
    though it is not itself in the brief's file list.
  - cell 9: per-cell (sex x SES level) marginal tau -> triple_cells_T18.csv,
    plus a decile-stratified (D1/D10) cross-check of the same taus via
    run_ses_decile_rdd -> triple_cells_decile_xcheck_T18.csv.
  - cell 11: print-only comparison (low-deprivation-cell winner, D1
    cross-check) -- no new artifact, reproduced for log fidelity.
  - cell 13: baseline (voluntary-side) turnout by cell, plus a raw-means
    cross-check at extreme deciles via gender_gap_means_by_decile ->
    triple_baseline_decile_xcheck_T18.csv.
  - cell 15: plot_gender_gap_baseline_endpoint for pca_index and nbi (the
    slope-chart convergence figures) -> gender_gap_baseline_endpoint_
    {pca_index,nbi}_T18.png.
  - cell 17: plot_triple_cells_grouped (tau, baseline) at T18 ->
    triple_tau_cells_T18.png, triple_baseline_cells_T18.png.
  - cell 19 (cont.): plot_triple_cells_grouped (tau) at T70 ->
    triple_tau_cells_T70.png.
  - cell 21: LaTeX exports for triple_interaction_T18/T70 and
    triple_cells_T18 (§7).

Resilience: the expensive block (cell 5+7+9's triple loop, theta/gap-test,
and the per-cell table) is guarded as one unit -- on resume its two
DataFrames (`triple_df`, `cells_by_ses`) are reconstructed straight from
their own CSV exports (triple_interaction_T18.csv, triple_cells_T18.csv),
since both are saved in the exact units `unpack_triple_interaction` returns
them in (probability units, not pp) and losslessly round-trip. Each
remaining step (decile cross-check, means cross-check, each figure, the T70
block, the .tex exports) is guarded independently on its own output file(s).
"""
import os
import warnings
from datetime import datetime

from _common import setup, load_window, REP_FIGURES, REP_TABLES  # noqa: F401
setup()


def _log(msg):
    """Timestamped, flushed progress line — localizes the failure if a run dies."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.analysis import (
    unpack_triple_interaction,
    plot_triple_cells_grouped,
    plot_gender_gap_baseline_endpoint,
    run_ses_decile_rdd,
    gender_gap_means_by_decile,
)

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})

# ── PARAMETERS (cell 1, verbatim) ───────────────────────────────────────────
SES_MEASURES  = ['nbi', 'pca_index']
BANDWIDTH_MAX = 365 * 3   # pre-filter window (matches 04/06/07)
N_DECILES     = 10

_log(f"Parameters: SES_MEASURES={SES_MEASURES}, BANDWIDTH_MAX={BANDWIDTH_MAX}d, "
     f"N_DECILES={N_DECILES}")

warnings.filterwarnings('ignore')

# ── Data load (cell 3) ───────────────────────────────────────────────────
# T18 union T70, +-3y: identical to the notebook's own memory-safe pattern
# (union mask on days_from_18/days_from_70, then a circuit cluster_id).
df = load_window(['voted', 'gender', 'nbi', 'pca_index'])
_log(f"Rows in T18 U T70 windows: {len(df):,}")

TRIPLE_COLS = ['ses_var', 'threshold', 'coef', 'coef_pp',
               'se', 'z', 'p', 'bw_days', 'p10', 'p90']

# ── Step 1 / cells 5+7+9a — T18 triple loop, theta1/gap-test, per-cell table ─
triple_csv_18 = f"{REP_TABLES}/triple_interaction_T18.csv"
cells_csv_18  = f"{REP_TABLES}/triple_cells_T18.csv"
theta_csv     = f"{REP_TABLES}/gender_gap_ses_theta_T18.csv"
gaptest_csv   = f"{REP_TABLES}/gender_gap_ses_test_T18.csv"

if all(os.path.exists(p) for p in (triple_csv_18, cells_csv_18, theta_csv, gaptest_csv)):
    _log("resume guard: T18 triple/theta/gap-test/cells outputs exist, "
         "reloading (cells 5, 7, 9a)")
    triple_df = pd.read_csv(triple_csv_18)
    cells_all = pd.read_csv(cells_csv_18)
    cells_by_ses = {ses: cells_all[cells_all['ses_var'] == ses].reset_index(drop=True)
                    for ses in SES_MEASURES}
else:
    _log("starting T18 triple-interaction loop (cell 5)")
    triple_rows = []
    cells_by_ses = {}
    for ses in SES_MEASURES:
        triple, cells = unpack_triple_interaction(
            df, ses_var=ses, threshold=18, bandwidth_max=BANDWIDTH_MAX,
            cluster_var='cluster_id')
        triple_rows.append(triple)
        cells_by_ses[ses] = cells
        _log(f"  {ses}: triple coef={triple['coef']*100:+.3f}pp, "
             f"se={triple['se']*100:.3f}, p={triple['p']:.3f}, bw={triple['bw_days']:.1f}d")

    triple_df = pd.DataFrame(triple_rows)
    triple_df['coef_pp'] = triple_df['coef'] * 100
    triple_df = triple_df[TRIPLE_COLS]
    triple_df.to_csv(triple_csv_18, index=False)
    print("Triple interaction  T x female x SES  (cluster-robust by circuit, "
          "continuous spec) — T18")
    print(triple_df.round(5).to_string(index=False))
    _log(f"Saved: {os.path.basename(triple_csv_18)}")

    # cell 7 — theta1 alone, and the F-M gap change at p10/p90, off the SAME
    # clustered ols fitted above (no re-fit; joint clustered covariance).
    _log("starting theta1 / F-M gap-change test (cell 7)")

    def _scalar(a):
        a = np.ravel(a)
        return float(a[0])

    theta_rows, gap_test_rows = [], []
    for tr in triple_rows:                      # T18, clustered (from cell 5)
        ols = tr['ols']; ses = tr['ses_var']
        p10, p90 = tr['p10'], tr['p90']
        nb = len(ols.params)
        assert ols.cov_type == 'cluster', ols.cov_type   # guard: clustered, not HC1

        # theta1 alone (col 14): does the F-M gap depend on SES?
        c = np.zeros(nb); c[14] = 1.0
        tt = ols.t_test(c)
        theta_rows.append({'ses_var': ses,
                           'theta1_pp': _scalar(tt.effect) * 100,
                           'se_pp': _scalar(tt.sd) * 100,
                           'z': _scalar(tt.tvalue), 'p': _scalar(tt.pvalue)})

        # F-M gap change at the cutoff = delta1 (col 10) + theta1 (col 14) * ses,
        # at p10 (low) and p90 (high), SE from the JOINT clustered vcov.
        for lvl, sv in [('Low (p10)', p10), ('High (p90)', p90)]:
            c = np.zeros(nb); c[10] = 1.0; c[14] = sv
            tt = ols.t_test(c)
            ci = tt.conf_int()
            gap_test_rows.append({'ses_var': ses, 'ses_level': lvl, 'ses_val': sv,
                                  'gap_change_pp': _scalar(tt.effect) * 100,
                                  'se_pp': _scalar(tt.sd) * 100,
                                  'z': _scalar(tt.tvalue), 'p': _scalar(tt.pvalue),
                                  'ci_lo_pp': ci[0, 0] * 100, 'ci_hi_pp': ci[0, 1] * 100})

    theta_df = pd.DataFrame(theta_rows)
    gap_test_df = pd.DataFrame(gap_test_rows)
    theta_df.to_csv(theta_csv, index=False)
    gap_test_df.to_csv(gaptest_csv, index=False)

    print("theta1 = T x female x SES (clustered) — does the F-M gap depend on SES? — T18")
    print(theta_df.round(5).to_string(index=False))
    print("\nF-M gender-gap change at cutoff = delta1 + theta1*ses (joint clustered vcov, pp) — T18")
    print(gap_test_df.round(4).to_string(index=False))
    _log(f"Saved: {os.path.basename(theta_csv)}, {os.path.basename(gaptest_csv)}")

    # cell 9a — per-cell (sex x SES level) marginal tau table.
    _log("starting per-cell marginal tau table (cell 9a)")
    cells_all = pd.concat(cells_by_ses.values(), ignore_index=True)
    cells_tau = cells_all[['ses_var', 'sex', 'ses_level', 'ses_val',
                           'tau', 'tau_se', 'tau_p', 'tau_lo', 'tau_hi']].copy()
    for c in ['tau', 'tau_se', 'tau_lo', 'tau_hi']:
        cells_tau[c] = cells_tau[c] * 100
    cells_all.to_csv(cells_csv_18, index=False)
    print("Marginal CV effect tau by sex x SES cell (continuous, pp) — T18")
    print(cells_tau.round(3).to_string(index=False))
    _log(f"Saved: {os.path.basename(cells_csv_18)}")

# ── Step 2 / cell 9b — decile-stratified cross-check at D1/D10 ─────────────
xcheck_csv = f"{REP_TABLES}/triple_cells_decile_xcheck_T18.csv"
if os.path.exists(xcheck_csv):
    _log(f"resume guard: {os.path.basename(xcheck_csv)} exists, skipping cell 9b")
    xcheck_tau = pd.read_csv(xcheck_csv)
else:
    _log("starting decile-stratified cross-check at D1/D10 (cell 9b)")
    xrows = []
    for ses in SES_MEASURES:
        dec, _ = run_ses_decile_rdd(df, ses_var=ses, threshold=18,
                                    n_deciles=N_DECILES, bandwidth=BANDWIDTH_MAX,
                                    verbose=False)
        for d in (1, 10):
            row = dec[dec['decile'] == d].iloc[0]
            tag = 'D1 (least deprived)' if d == 1 else 'D10 (most deprived)'
            for sx, name in [('F', 'Women'), ('M', 'Men')]:
                xrows.append({'ses_var': ses, 'decile': tag, 'sex': name,
                              'tau_pp': row[f'tau_{sx}'] * 100,
                              'se_pp': row[f'se_{sx}'] * 100})
        _log(f"  {ses}: decile cross-check done")
    xcheck_tau = pd.DataFrame(xrows)
    xcheck_tau.to_csv(xcheck_csv, index=False)
    print("\nDecile-stratified per-sex tau at extreme deciles (cross-check, pp) — T18")
    print(xcheck_tau.round(3).to_string(index=False))
    _log(f"Saved: {os.path.basename(xcheck_csv)}")

# ── Step 3 / cell 11 — print-only comparison (no new artifact) ─────────────
_log("cell 11: low-deprivation-cell winner comparison (print only)")
print("Low-deprivation (p10) cell with the largest CV effect (continuous):")
for ses in SES_MEASURES:
    low = cells_by_ses[ses]
    low = low[low['ses_level'] == 'Low (p10)'].set_index('sex')['tau'] * 100
    winner = low.idxmax()
    print(f"  {ses:>9}: Women={low['Women']:.2f}pp  Men={low['Men']:.2f}pp"
          f"  -> max = {winner} (gap {low['Women']-low['Men']:+.2f}pp)")

print("\nCross-check at the least-deprived decile (D1):")
print(xcheck_tau[xcheck_tau['decile'].str.startswith('D1')]
      .round(2).to_string(index=False))

# ── Step 4 / cell 13 — baseline table + raw-means cross-check ──────────────
base_xcheck_csv = f"{REP_TABLES}/triple_baseline_decile_xcheck_T18.csv"
cells_base = cells_all[['ses_var', 'sex', 'ses_level', 'ses_val',
                        'baseline', 'baseline_se',
                        'baseline_lo', 'baseline_hi']].copy()
for c in ['baseline', 'baseline_se', 'baseline_lo', 'baseline_hi']:
    cells_base[c] = cells_base[c] * 100
print("Baseline turnout, voluntary side (x=0-), continuous spec (pp) — T18")
print(cells_base.round(3).to_string(index=False))

if os.path.exists(base_xcheck_csv):
    _log(f"resume guard: {os.path.basename(base_xcheck_csv)} exists, skipping cell 13 xcheck")
    base_xcheck = pd.read_csv(base_xcheck_csv)
else:
    _log("starting raw left-side means cross-check at extreme deciles (cell 13)")
    brows = []
    for ses in SES_MEASURES:
        lm = gender_gap_means_by_decile(df, ses_var=ses, threshold=18,
                                        n_deciles=N_DECILES, bandwidth=BANDWIDTH_MAX,
                                        side='left', means_window=90)
        for d in (1, 10):
            row = lm[lm['decile'] == d].iloc[0]
            tag = 'D1 (least deprived)' if d == 1 else 'D10 (most deprived)'
            brows.append({'ses_var': ses, 'decile': tag,
                          'turnout_F_pp': row['turnout_F'] * 100,
                          'turnout_M_pp': row['turnout_M'] * 100,
                          'F_minus_M_pp': row['diff_FM'] * 100,
                          'n_F': int(row['n_F']), 'n_M': int(row['n_M'])})
    base_xcheck = pd.DataFrame(brows)
    base_xcheck.to_csv(base_xcheck_csv, index=False)
    print("\nRaw left-side means 90d pre-cutoff at extreme deciles (cross-check, pp) — T18")
    print(base_xcheck.round(3).to_string(index=False))
    _log(f"Saved: {os.path.basename(base_xcheck_csv)}")

# ── Step 5 / cell 15 — slope charts: voluntary baseline -> compulsory endpoint ─
gap_be_pca_png = f"{REP_FIGURES}/gender_gap_baseline_endpoint_pca_index_T18.png"
gap_be_nbi_png = f"{REP_FIGURES}/gender_gap_baseline_endpoint_nbi_T18.png"
if os.path.exists(gap_be_pca_png) and os.path.exists(gap_be_nbi_png):
    _log("resume guard: gender_gap_baseline_endpoint_{pca_index,nbi}_T18.png exist, "
         "skipping cell 15")
else:
    _log("starting slope charts (cell 15): pca_index primary, nbi robustness")
    gap_be_pca = plot_gender_gap_baseline_endpoint(
        cells_by_ses['pca_index'], 'pca_index', REP_FIGURES, threshold=18)
    gap_be_nbi = plot_gender_gap_baseline_endpoint(
        cells_by_ses['nbi'], 'nbi', REP_FIGURES, threshold=18)
    _log("Saved: gender_gap_baseline_endpoint_pca_index_T18.png, "
         "gender_gap_baseline_endpoint_nbi_T18.png")

# ── Step 6 / cell 17 — grouped bar charts: tau and baseline by cell, T18 ───
tau_cells_png = f"{REP_FIGURES}/triple_tau_cells_T18.png"
base_cells_png = f"{REP_FIGURES}/triple_baseline_cells_T18.png"
if os.path.exists(tau_cells_png) and os.path.exists(base_cells_png):
    _log("resume guard: triple_{tau,baseline}_cells_T18.png exist, skipping cell 17")
else:
    _log("starting grouped bar charts (cell 17): tau and baseline, T18")
    plot_triple_cells_grouped(cells_by_ses, REP_FIGURES, value='tau', threshold=18)
    plot_triple_cells_grouped(cells_by_ses, REP_FIGURES, value='baseline', threshold=18)
    _log("Saved: triple_tau_cells_T18.png, triple_baseline_cells_T18.png")

# ── Step 7 / cell 19 — T70 triple loop (secondary threshold) ───────────────
triple_csv_70 = f"{REP_TABLES}/triple_interaction_T70.csv"
cells_csv_70  = f"{REP_TABLES}/triple_cells_T70.csv"
tau_cells_70_png = f"{REP_FIGURES}/triple_tau_cells_T70.png"

if (os.path.exists(triple_csv_70) and os.path.exists(cells_csv_70)
        and os.path.exists(tau_cells_70_png)):
    _log("resume guard: T70 triple outputs exist, reloading (cell 19)")
    triple_df_70 = pd.read_csv(triple_csv_70)
    cells_all_70 = pd.read_csv(cells_csv_70)
else:
    _log("starting T70 triple-interaction loop (cell 19)")
    triple_rows_70, cells_by_ses_70 = [], {}
    for ses in SES_MEASURES:
        triple, cells = unpack_triple_interaction(
            df, ses_var=ses, threshold=70, bandwidth_max=BANDWIDTH_MAX,
            cluster_var='cluster_id')
        triple_rows_70.append(triple)
        cells_by_ses_70[ses] = cells
        _log(f"  {ses}: T70 triple coef={triple['coef']*100:+.3f}pp, "
             f"se={triple['se']*100:.3f}, p={triple['p']:.3f}, bw={triple['bw_days']:.1f}d")

    triple_df_70 = pd.DataFrame(triple_rows_70)
    triple_df_70['coef_pp'] = triple_df_70['coef'] * 100
    triple_df_70 = triple_df_70[TRIPLE_COLS]
    triple_df_70.to_csv(triple_csv_70, index=False)
    cells_all_70 = pd.concat(cells_by_ses_70.values(), ignore_index=True)
    cells_all_70.to_csv(cells_csv_70, index=False)

    print("Triple interaction — T70 (secondary, cluster-robust by circuit)")
    print(triple_df_70.round(5).to_string(index=False))
    show = cells_all_70[['ses_var', 'sex', 'ses_level',
                         'tau', 'tau_se', 'tau_p', 'baseline', 'baseline_se']].copy()
    for c in ['tau', 'tau_se', 'baseline', 'baseline_se']:
        show[c] = show[c] * 100
    print("\nMarginal tau and baseline by cell — T70 (pp).")
    print("NOTE: at T70, T=1 is the VOLUNTARY side (>=70); 'baseline' = COMPULSORY side (<70).")
    print(show.round(3).to_string(index=False))
    _log(f"Saved: {os.path.basename(triple_csv_70)}, {os.path.basename(cells_csv_70)}")

    plot_triple_cells_grouped(cells_by_ses_70, REP_FIGURES, value='tau', threshold=70)
    _log("Saved: triple_tau_cells_T70.png")

# ── Step 8 / cell 21 — LaTeX exports (Sec. 7) ───────────────────────────────
tex_18 = f"{REP_TABLES}/triple_interaction_T18.tex"
tex_70 = f"{REP_TABLES}/triple_interaction_T70.tex"
tex_cells_18 = f"{REP_TABLES}/triple_cells_T18.tex"

if os.path.exists(tex_18) and os.path.exists(tex_70) and os.path.exists(tex_cells_18):
    _log("resume guard: triple_interaction_{T18,T70}.tex, triple_cells_T18.tex exist, "
         "skipping cell 21")
else:
    _log("starting LaTeX exports (cell 21)")

    def _save_tex(df_, path, caption, label, ff='%.4f', note=None, escape=True):
        # escape=False is needed where a column header carries maths ($\tau$);
        # the cell contents of these tables are numbers and plain words, so
        # disabling the escape costs nothing.
        tex = df_.to_latex(index=False, float_format=ff, escape=escape,
                           caption=caption, label=label)
        if note:
            tex = tex.replace(
                "\\end{table}",
                "\\par\\vspace{0.6em}\n{\\footnotesize\\begin{minipage}{\\linewidth}"
                f"\\emph{{Note:}} {note}\\end{{minipage}}}}\n\\end{{table}}",
                1)
        with open(path, "w") as fout:
            fout.write(tex)
        print('Saved:', path)

    # Pretty names for the SES column: the raw keys read as unfinished work in a
    # manuscript table.
    _SES_TEX = {'nbi': 'NBI', 'pca_index': 'PCA index'}

    def _pretty_triple(df_):
        """Manuscript-ready view of the triple-interaction row set.

        Drops the duplicated probability-unit coefficient and the threshold column
        (the caption already names the cutoff), puts the SE in percentage points so
        it is on the coefficient's scale, and keeps p10/p90 so the reader can scale
        the coefficient onto the observed range (see _NOTE_TRIPLE).
        """
        out = pd.DataFrame({
            'SES measure': df_['ses_var'].map(_SES_TEX).fillna(df_['ses_var']),
            'Coef. (pp)':  df_['coef_pp'],
            'SE (pp)':     df_['se'] * 100,
            'z':           df_['z'],
            'p':           df_['p'],
            # As a string so the table-wide float_format does not print a
            # bandwidth to three decimals.
            'BW (days)':   df_['bw_days'].map('{:.1f}'.format),
            'p10':         df_['p10'],
            'p90':         df_['p90'],
        })
        return out

    # The NBI coefficient (-26.25) is uninterpretable without its scale: NBI is a
    # proportion on 0-1, so a "one-unit" change is the whole range. Spelling this
    # out in the note is what makes the two rows comparable to each other.
    _NOTE_TRIPLE = (
        "Coefficient on $T \\times$ female $\\times$ SES from "
        "Equation~\\ref{eq:triple}, in percentage points per one-unit change in "
        "the SES measure. The two measures are on different scales --- NBI is a "
        "proportion $(0,1)$, the PCA index is in index units --- so the columns "
        "p10 and p90 give the range each actually spans: the implied change in "
        "the gender gap across the observed distribution is "
        "Coef.\\,$\\times$\\,(p90 $-$ p10). Standard errors clustered by "
        "electoral circuit (CRV1)."
    )
    # The cell table is routinely misread against the stratified per-cell rows of
    # Table~\ref{tab:rdd_results}, which are fit at their own bandwidths and so
    # need not agree with these.
    _NOTE_CELLS = (
        "Cells are marginal effects from the continuous triple-interaction model "
        "(Section~\\ref{sec:strat_hetero}, Equation~\\ref{eq:triple}), evaluated at "
        "the 10th and 90th percentiles of the SES measure within the pooled "
        "MSE-optimal window ($\\approx$47 days). They are not the stratified "
        "per-cell RD estimates of Table~\\ref{tab:rdd_results}, each of which is "
        "fit at its own bandwidth, and the two need not coincide. Standard errors "
        "clustered by electoral circuit (CRV1)."
    )

    _save_tex(_pretty_triple(triple_df), tex_18,
              'CV $\\times$ sex $\\times$ SES triple-interaction coefficient (age-18 cutoff).',
              'tab:triple_T18', ff='%.3f', note=_NOTE_TRIPLE)
    _save_tex(_pretty_triple(triple_df_70), tex_70,
              'CV $\\times$ sex $\\times$ SES triple-interaction coefficient (age-70 cutoff).',
              'tab:triple_T70', ff='%.3f', note=_NOTE_TRIPLE)

    # tau_p is dropped: every cell effect is significant at any conventional level,
    # so the column printed 0.00 in all eight rows and carried no information. The
    # standard errors are what a reader needs, and they stay.
    _cells_tex = pd.DataFrame({
        'SES measure':   cells_all['ses_var'].map(_SES_TEX).fillna(cells_all['ses_var']),
        'Sex':           cells_all['sex'],
        'Deprivation':   cells_all['ses_level'],
        '$\\tau$ (pp)':      cells_all['tau'] * 100,
        'SE ($\\tau$)':      cells_all['tau_se'] * 100,
        'Baseline (pp)':     cells_all['baseline'] * 100,
        'SE (baseline)':     cells_all['baseline_se'] * 100,
    })
    _save_tex(_cells_tex, tex_cells_18,
              ('Marginal CV effect ($\\tau$) and voluntary-side baseline by '
               'sex $\\times$ SES cell (age-18 cutoff, pp).'),
              'tab:triple_cells_T18', ff='%.2f', note=_NOTE_CELLS, escape=False)
    _log(f"Saved: {os.path.basename(tex_18)}, {os.path.basename(tex_70)}, "
         f"{os.path.basename(tex_cells_18)}")

print("\n05_triple: DONE")
