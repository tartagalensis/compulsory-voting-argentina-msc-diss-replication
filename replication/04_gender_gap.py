"""04 — Gender-Gap RD at Age 18: canonical F-M gap, decile re-run, Figure 4
(gender-gap RD vs per-sex/pooled decile panel), and the §8c robustness forest
plot (gap_jump_robustness).

Transcribed from notebooks/07_gradient.ipynb, cells in scope per the task map
(§8, §8c, §8d — everything else in `07_gradient.ipynb` is Task 3's / already
in `03_gradient.py`): the §8d canonical global F-M gap cell (cell 20,
transcribed verbatim except FIGURES_DIR/TABLES_DIR -> REP_FIGURES/REP_TABLES),
a decile re-run standing in for §3's `decile_results[('pca_index', 18)]`
(`run_ses_decile_rdd`, identical call signature to 03_gradient.py), the §8
Figure-4 call (cell 16), and the §8c forest-plot cell (cell 18) with its
MDE/CSV/LaTeX export lines.

ORDER MATTERS: §8d must run and write `gender_gap_global_rd_T18.csv` to
REP_TABLES *before* the Figure-4 call, which reads that CSV via
`rd_result_csv=` so the figure annotation and the thesis text share one
source of truth (see the notebook's own cell-16 comment).

Data load: none of §8/§8c/§8d ever touch `days_from_70` or `nbi` — only the
age-18 window, `voted`, `gender`, and `pca_index` (for the decile re-run) are
needed, so `load_window(['voted', 'gender', 'pca_index'], thresholds=(18,))`
is used (narrower than 03_gradient.py's T18-union-T70 load, since this script
never needs the T70 side).

Resilience: each of the four steps guards on its own final artifact(s) and is
independently resumable; the decile re-run (the only "expensive" step besides
§8d's rdrobust-on-the-full-window calls) is additionally checkpointed to a
pickle so a relaunch after Figure-4/§8c failure doesn't redo it.
"""
import os
import pickle
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
import statsmodels.api as sm
from scipy.stats import norm
from rdrobust import rdrobust

from src.analysis import (
    run_ses_decile_rdd,
    plot_gender_gap_rd_and_decile,
    plot_gap_jump_robustness,
)

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})

# ── PARAMETERS (cell 2, verbatim subset used by this script) ───────────────
N_DECILES   = 10              # population-weighted SES quantiles
BANDWIDTH   = 365 * 3         # pre-filter window in days (3 years, matches 04/06)
ALPHA_LEVEL = 0.10            # 90% CI for post-cutoff level estimates
THRESHOLD   = 18              # only the age-18 cutoff is in scope for §8/§8c/§8d

_log(f"Parameters: N_DECILES={N_DECILES}, BANDWIDTH={BANDWIDTH}d, THRESHOLD={THRESHOLD}")

warnings.filterwarnings('ignore')

# ── Data load ────────────────────────────────────────────────────────────
# T18-only window: none of §8/§8c/§8d use days_from_70 or nbi.
df = load_window(['voted', 'gender', 'pca_index'], thresholds=(THRESHOLD,))
_log(f"Records (windowed T{THRESHOLD} +-3y): {len(df):,}")

gap_csv = f"{REP_TABLES}/gender_gap_global_rd_T18.csv"

# ── Step 1 / §8d — Canonical global F-M gap change at the age-18 cutoff ────
# (R4b). rdrobust at a COMMON bandwidth, consistent with the rest of the
# paper (MSE-optimal, bias-corrected/robust). Reuses the existing per-sex
# rdrobust pattern (masspoints 'rd'; point = coef row 0, robust SE = se row
# 2); the ONLY change is fixing h. Transcribed verbatim from cell 20 (only
# TABLES_DIR -> REP_TABLES).
if os.path.exists(gap_csv):
    _log(f"resume guard: {os.path.basename(gap_csv)} exists, skipping §8d")
    gap_versions = pd.read_csv(gap_csv)
    _h_row = gap_versions[gap_versions['version'].str.contains('CANONICAL')].iloc[0]
    _log(f"loaded canonical gap: {_h_row['gap_pp']:+.3f}pp, "
         f"SE {_h_row['se_pp']:.3f}, p={_h_row['p']:.3f}, bw={_h_row['bandwidth']}")
else:
    _log("starting §8d canonical global F-M gap (rdrobust, common bandwidth)")

    _tv = 'days_from_18'; _BWMAX = 365 * 3
    _base = df[df['gender'].isin(['F', 'M']) & df[_tv].between(-_BWMAX, _BWMAX)].copy()
    _Z = norm.ppf(0.975)
    _MDEk = 1.959963984540054 + 0.8416212335729143       # z_.975 + z_.80 ~ 2.80

    def _rd(_d, _h=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _r = rdrobust(y=_d['voted'].values, x=_d[_tv].values, c=0,
                          masspoints='rd', **({'h': _h} if _h is not None else {}))
        return float(_r.coef.iloc[0, 0]) * 100, float(_r.se.iloc[2, 0]) * 100, float(_r.bws.iloc[0, 0])

    # (A) CANONICAL: pooled MSE-optimal h -> common h fixed for both sexes.
    _h = _rd(_base)[2]
    _F = _rd(_base[_base['gender'] == 'F'], _h)
    _M = _rd(_base[_base['gender'] == 'M'], _h)
    _gapA, _seA = _F[0] - _M[0], np.sqrt(_F[1]**2 + _M[1]**2)

    # (B) DISCARDED: each sex at its OWN MSE-optimal h (different windows).
    _bF = _rd(_base[_base['gender'] == 'F']); _bM = _rd(_base[_base['gender'] == 'M'])
    _gapB, _seB = _bF[0] - _bM[0], np.sqrt(_bF[1]**2 + _bM[1]**2)

    # (C) ROBUSTNESS: OLS linear +/-365 (the female*right coef = col 6; HC1).
    _s = _base[_base[_tv].between(-365, 365)]
    _x = _s[_tv].to_numpy(float); _f = (_s['gender'] == 'F').to_numpy(float)
    _r = (_s[_tv] >= 0).to_numpy(float)
    _X = np.column_stack([np.ones_like(_x), _x, _r, _r*_x, _f, _f*_x, _f*_r, _f*_r*_x])
    _mC = sm.OLS(_s['voted'].to_numpy(float), _X).fit(cov_type='HC1')
    _gapC, _seC = _mC.params[6] * 100, _mC.bse[6] * 100

    def _row(label, gap, se, bw, quantity='gap (F-M)'):
        # `quantity` documents what `gap_pp` holds: a difference for the gap rows,
        # a per-sex level for the tau rows added below. MDE is a property of a
        # contrast, so it is left blank on the level rows.
        z = gap / se
        return {'version': label, 'quantity': quantity, 'gap_pp': gap, 'se_pp': se,
                'z': z, 'p': 2 * (1 - norm.cdf(abs(z))), 'ci_lo_pp': gap - _Z*se,
                'ci_hi_pp': gap + _Z*se,
                'mde80_pp': _MDEk * se if quantity == 'gap (F-M)' else np.nan,
                'bandwidth': bw}

    # The per-sex taus at the COMMON h are persisted alongside the difference:
    # the thesis needs them to show that the sex-specific estimates BRACKET the
    # pooled Total of Table 2 (17.9 < 18.25 < 18.5), which the own-h pair of that
    # table does not (both sit above it). Labels must not contain "CANONICAL" —
    # the figure annotation and the resume guard both select on that string.
    gap_versions = pd.DataFrame([
        _row('rdrobust common h (CANONICAL)', _gapA, _seA, f'common {_h:.1f}d'),
        _row('tau_F at common h', _F[0], _F[1], f'common {_h:.1f}d', quantity='level (tau_F)'),
        _row('tau_M at common h', _M[0], _M[1], f'common {_h:.1f}d', quantity='level (tau_M)'),
        _row('rdrobust differenced (own h, defective)', _gapB, _seB,
             f'F={_bF[2]:.1f}d, M={_bM[2]:.1f}d'),
        _row('OLS linear +/-365 (robustness, not canonical)', _gapC, _seC, '+/-365d linear'),
    ])
    gap_versions.to_csv(gap_csv, index=False)
    print("Canonical global F-M gap change at age-18 cutoff (rdrobust, common bandwidth):")
    print(gap_versions.round(4).to_string(index=False))
    print(f"\nPer-sex at common h={_h:.1f}d: tau_F={_F[0]:+.3f}pp (SE {_F[1]:.3f}), "
          f"tau_M={_M[0]:+.3f}pp (SE {_M[1]:.3f})")
    _log(f"Saved: {os.path.basename(gap_csv)}")

# ── Step 1b — Same contrast at the age-70 cutoff (defensive, see PLAN.md D-B) ──
# Table 2 reports T70 tau_F=-4.70 and tau_M=-3.14 at their OWN bandwidths (167 vs
# 225 days), i.e. the larger response is women's — the reverse of T18. Because the
# two are fit in different windows that contrast is not a test, exactly as at T18.
# This step applies the same common-h fix so the thesis can state what the T70
# ordering is instead of leaving an unaddressed flank in Table 2 and Figure 3.
# Self-contained (its own T70 load, freed immediately) so the T18 path above is
# untouched.
gap_csv_70 = f"{REP_TABLES}/gender_gap_global_rd_T70.csv"
if os.path.exists(gap_csv_70):
    _log(f"resume guard: {os.path.basename(gap_csv_70)} exists, skipping T70 gap")
    _g70 = pd.read_csv(gap_csv_70)
    _r70 = _g70[_g70['version'].str.contains('CANONICAL')].iloc[0]
    _log(f"loaded T70 gap: {_r70['gap_pp']:+.3f}pp, SE {_r70['se_pp']:.3f}, "
         f"p={_r70['p']:.3f}, bw={_r70['bandwidth']}")
else:
    _log("starting T70 canonical global F-M gap (rdrobust, common bandwidth)")
    df70 = load_window(['voted', 'gender'], thresholds=(70,))
    _log(f"Records (windowed T70 +-3y): {len(df70):,}")

    _tv70 = 'days_from_70'; _BWMAX = 365 * 3
    _Z = norm.ppf(0.975)
    _MDEk = 1.959963984540054 + 0.8416212335729143
    _b70 = df70[df70['gender'].isin(['F', 'M'])
                & df70[_tv70].between(-_BWMAX, _BWMAX)].copy()

    def _rd70(_d, _h=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _r = rdrobust(y=_d['voted'].values, x=_d[_tv70].values, c=0,
                          masspoints='rd', **({'h': _h} if _h is not None else {}))
        return (float(_r.coef.iloc[0, 0]) * 100, float(_r.se.iloc[2, 0]) * 100,
                float(_r.bws.iloc[0, 0]))

    _h70 = _rd70(_b70)[2]
    _F70 = _rd70(_b70[_b70['gender'] == 'F'], _h70)
    _M70 = _rd70(_b70[_b70['gender'] == 'M'], _h70)
    _gap70 = _F70[0] - _M70[0]
    _se70 = np.sqrt(_F70[1]**2 + _M70[1]**2)

    def _row70(label, val, se, quantity):
        z = val / se
        return {'version': label, 'quantity': quantity, 'gap_pp': val, 'se_pp': se,
                'z': z, 'p': 2 * (1 - norm.cdf(abs(z))), 'ci_lo_pp': val - _Z*se,
                'ci_hi_pp': val + _Z*se,
                'mde80_pp': _MDEk * se if quantity == 'gap (F-M)' else np.nan,
                'bandwidth': f'common {_h70:.1f}d'}

    gap70 = pd.DataFrame([
        _row70('rdrobust common h (CANONICAL, T70)', _gap70, _se70, 'gap (F-M)'),
        _row70('tau_F at common h (T70)', _F70[0], _F70[1], 'level (tau_F)'),
        _row70('tau_M at common h (T70)', _M70[0], _M70[1], 'level (tau_M)'),
    ])
    gap70.to_csv(gap_csv_70, index=False)
    print("\nF-M gap change at the age-70 cutoff (rdrobust, common bandwidth):")
    print(gap70.round(4).to_string(index=False))
    print(f"\nNOTE: at T70 tau is the effect of REMOVING compulsion, so both taus are "
          f"negative and the LARGER |tau| is the sex compulsion was holding up more.")
    print(f"Per-sex at common h={_h70:.1f}d: tau_F={_F70[0]:+.3f}pp (SE {_F70[1]:.3f}), "
          f"tau_M={_M70[0]:+.3f}pp (SE {_M70[1]:.3f})")
    _log(f"Saved: {os.path.basename(gap_csv_70)}")
    del df70, _b70


# ── Step 2 — Decile re-run standing in for §3's decile_results[('pca_index', 18)] ──
# (RESUME GUARD: checkpointed to a pickle — 10 decile rdrobust calls x 2 sexes.)
decile_partial = f"{REP_TABLES}/_partial_gendergap_decile_pca_T18.pkl"
if os.path.exists(decile_partial):
    with open(decile_partial, 'rb') as fh:
        decile_df = pickle.load(fh)
    _log("resume guard: decile re-run loaded from checkpoint")
else:
    _log(f"starting decile re-run (pca_index x T{THRESHOLD}, {N_DECILES} deciles)")
    decile_df, _bounds = run_ses_decile_rdd(
        df, ses_var='pca_index', threshold=THRESHOLD,
        n_deciles=N_DECILES, bandwidth=BANDWIDTH, alpha_level=ALPHA_LEVEL,
    )
    with open(decile_partial, 'wb') as fh:
        pickle.dump(decile_df, fh)
    _log(f"checkpoint: decile re-run -> {os.path.basename(decile_partial)}")

# ── Step 3 / §8 Figure 4 — Gender-gap RD vs per-sex/pooled RD by decile ────
fig4_name = f'rdd_gradient_gaprd_vs_decile_pca_index_T{THRESHOLD}.png'
if os.path.exists(f"{REP_FIGURES}/{fig4_name}"):
    _log(f"resume guard: {fig4_name} exists, skipping Figure 4")
else:
    _log("starting §8 Figure 4 (gender-gap RD vs per-sex/pooled decile panel)")
    _ = plot_gender_gap_rd_and_decile(
        df, decile_df,
        ses_var='pca_index', figures_dir=REP_FIGURES,
        threshold=THRESHOLD, n_deciles=N_DECILES, bandwidth=BANDWIDTH,
        bw_days=365, bin_days=30, lowess_frac=0.6,
        rd_result_csv=gap_csv,
    )
    _log(f"Saved: {fig4_name}")

# ── Step 4 / §8c — Is the equalizing effect real? Robustness forest plot ───
jump_csv = f"{REP_TABLES}/gap_jump_robustness_T18.csv"
jump_tex = f"{REP_TABLES}/gap_jump_robustness_T18.tex"
jump_png = f"gap_jump_robustness_T{THRESHOLD}.png"
if (os.path.exists(jump_csv) and os.path.exists(jump_tex)
        and os.path.exists(f"{REP_FIGURES}/{jump_png}")):
    _log("resume guard: gap_jump_robustness_T18.{csv,tex,png} exist, skipping §8c")
else:
    _log(f"starting §8c gender-gap jump robustness (forest plot, T{THRESHOLD})")
    # Filters internally to ±3y, so passing the full (T18-windowed) `df` is fine.
    gap_jump_specs = plot_gap_jump_robustness(df, REP_FIGURES, threshold=THRESHOLD)

    # Minimum detectable effect (MDE) at 80% power, 5% two-sided: MDE = (z_.975 +
    # z_.80)*SE, with SE recovered from each 95% CI. Frames the null as informative:
    # "an equalizing effect of >= MDE would have been detected".
    _z975, _z80 = 1.959963984540054, 0.8416212335729143
    gap_jump_specs['se'] = (gap_jump_specs['hi'] - gap_jump_specs['lo']) / (2 * _z975)
    gap_jump_specs['mde80'] = (_z975 + _z80) * gap_jump_specs['se']
    gap_jump_specs.to_csv(jump_csv, index=False)
    _tex = gap_jump_specs[['spec', 'est', 'lo', 'hi', 'p', 'mde80']]
    with open(jump_tex, "w") as fout:
        fout.write(_tex.to_latex(
            index=False, float_format='%.2f', escape=True,
            caption=('Jump in the Women$-$Men turnout gap at age 18 across '
                     'specifications, with the 80\\% minimum detectable effect.'),
            label='tab:gap_jump_robustness'))
    _mde_rd = gap_jump_specs.loc[gap_jump_specs['kind'] == 'rd', 'mde80'].iloc[0]
    print(f"\nMDE (80% power, 5% two-sided): rdrobust benchmark = {_mde_rd:.2f}pp "
          f"(any equalizing effect >= {_mde_rd:.2f}pp would have been detected)")
    _log(f"Saved: {os.path.basename(jump_csv)} / {os.path.basename(jump_tex)}")

# Checkpoint no longer needed now that Figure 4 (its only consumer) is done.
if os.path.exists(decile_partial) and os.path.exists(f"{REP_FIGURES}/{fig4_name}"):
    os.remove(decile_partial)
    _log("removed decile re-run checkpoint (run completed successfully)")

print("\n04_gender_gap: DONE")
