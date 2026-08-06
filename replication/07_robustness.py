"""07 — Robustness Checks: bandwidth sensitivity, placebo cutpoints, and
rdplot visualisation for the main age-18/age-70 RDDs and their NBI/PCA
subgroups, plus `bw_test.png`.

Transcribed from notebooks/05_robustness.ipynb (cells 1-3, 9, 11, 13, 15,
17, 19, 21). The McCrary/rddensity cells (4-7) are SKIPPED: they only
`display()` a formal test object and are not tied to any of the 9 artifacts
this script owns; the numeric density test (t_jk/p_jk) is already exported
by `02_main_rdd.py` as `density_test_results.csv`, and the McCrary visual
(`plot_mccrary_visual`, `03_analysis.ipynb`) is not thesis-referenced (Task
3 confirmed this — see its report). Preserves the CLAUDE.md-flagged pandas-
3.x quirk pattern (rddensity may need `.values.copy()` + try/except) even
though it is not exercised here, since it is not needed for this script's
scope.

`bw_test.png` origin (NOT produced by any cell in the current notebook):
byte-level reverse-engineering (the curated `outputs/figures/bw_test.png`
coef/CI values were extracted and matched exactly, to 3 decimals, against a
re-run of cell 9's own bandwidth loop restricted to bw in {20, 40, 60}: T18
coef 0.1244/0.1738/0.1921, T70 coef -0.0076/-0.0226/-0.0269 at bw=20/40/60)
shows it is literally cell 9's own computation (same rdrobust call,
`masspoints="off"`, coef NOT scaled to pp) rendered at a truncated bandwidth
grid — a "bw test" / preview run predating the final `range(20, 301, 20)`
sweep, left in the repo and picked up by the appendix caption alongside
`threshold_density.png` (owned by `01_descriptives.py`). This script
reproduces it as a by-product of the main bandwidth sweep (same 15-point
grid; bw_test.png just re-renders the bw in {20,40,60} subset), with zero
extra rdrobust calls.

Data load: memory-safe via `load_window(['voted','nbi','pca_index'],
thresholds=(18,70))` (parquet predicate pushdown to the union of the +-3y
windows), matching `02_main_rdd.py`'s pattern; nbi_median/pca_median are
computed on a separate column-scoped full-frame read (cell 3 computes them
on the full `national.parquet`, before windowing).

Resilience: the four expensive sweeps (main/NBI/PCA bandwidth sensitivity,
main/NBI/PCA placebo cutpoints) are each checkpointed per (subgroup x
threshold) unit to a pickle in REP_TABLES, so a crash mid-sweep resumes at
the next un-checkpointed unit instead of redoing the whole figure; each
final PNG is independently resume-guarded (skip if it already exists).
"""
import gc
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
from rdrobust import rdrobust, rdplot
from config import FINAL_DIR

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})

warnings.filterwarnings('ignore')

BW_MAX = 365 * 3
BW_GRID = list(range(20, 301, 20))          # 15 bandwidths, matches cells 9/15/19


# ── Checkpoint helpers (pickled dict: unit key -> results DataFrame) ───────
def _ckpt_path(name):
    return f"{REP_TABLES}/_partial_robustness_{name}.pkl"


def _load_ckpt(name):
    path = _ckpt_path(name)
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    return {}


def _save_ckpt(name, data):
    with open(_ckpt_path(name), 'wb') as fh:
        pickle.dump(data, fh)


def _clear_ckpt(name):
    path = _ckpt_path(name)
    if os.path.exists(path):
        os.remove(path)


# ── Sweep helpers (verbatim logic from cells 9/11/15/17/19/21) ─────────────
def _bw_sweep(d, rv, bw_grid=BW_GRID, min_obs=500):
    """RD estimate + CI at each bandwidth in bw_grid (h fixed, masspoints='off')."""
    rows = []
    for bw in bw_grid:
        sub = d[d[rv].between(-bw, bw)]
        if len(sub) < min_obs:
            continue
        try:
            r = rdrobust(y=sub['voted'].values, x=sub[rv].values, c=0, h=bw, masspoints="off")
            rows.append({'bw': bw, 'coef': float(r.coef.iloc[0, 0]),
                        'ci_low': float(r.ci.iloc[0, 0]), 'ci_high': float(r.ci.iloc[0, 1])})
        except Exception:
            pass
    return pd.DataFrame(rows)


def _placebo_sweep(d, rv, true_bw, bw_max=BW_MAX, step=60, min_obs=500):
    """RD estimate + CI at a grid of fake cutoffs, excluding |c| <= 1.5*true_bw."""
    placebo_cutoffs = list(range(-int(bw_max * 0.8), int(bw_max * 0.8), step))
    placebo_cutoffs = [c for c in placebo_cutoffs if abs(c) > true_bw * 1.5]
    rows = []
    for c_p in placebo_cutoffs:
        sub = d[d[rv].between(c_p - bw_max // 2, c_p + bw_max // 2)].copy()
        sub['rv_centered'] = sub[rv] - c_p
        if len(sub) < min_obs:
            continue
        try:
            r = rdrobust(y=sub['voted'].values, x=sub['rv_centered'].values, c=0, masspoints="off")
            rows.append({'cutoff': c_p, 'coef': float(r.coef.iloc[0, 0]),
                        'ci_low': float(r.ci.iloc[0, 0]), 'ci_high': float(r.ci.iloc[0, 1])})
        except Exception:
            pass
    return pd.DataFrame(rows)


# ── Data load ────────────────────────────────────────────────────────────
# nbi_median / pca_median are computed on the FULL frame in the notebook
# (before windowing) — reproduce that with a separate column-scoped full pass
# (same pattern as 02_main_rdd.py).
_log("computing nbi_median / pca_median (full-frame column-scoped read)")
df_medians = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['nbi', 'pca_index'])
nbi_median = df_medians['nbi'].median()
pca_median = df_medians['pca_index'].median()
del df_medians
gc.collect()

df = load_window(['voted', 'nbi', 'pca_index'], thresholds=(18, 70), bw=BW_MAX)
_log(f"Records (windowed union of T18/T70 +-3y): {len(df):,}")

d18 = df[df['days_from_18'].between(-BW_MAX, BW_MAX)].copy()
d70 = df[df['days_from_70'].between(-BW_MAX, BW_MAX)].copy()
del df
gc.collect()

_log(f"Observations near threshold 18: {len(d18):,}")
_log(f"Observations near threshold 70: {len(d70):,}")
_log(f"NBI median: {nbi_median:.4f}, PCA median: {pca_median:.4f}")

# Main RDD estimates — needed only for the optimal-BW reference used to
# exclude placebo cutoffs too close to the true cutoff (cells 11/17/21).
_log("computing main rdd_18 / rdd_70 (masspoints='off') for optimal-BW reference")
rdd_18 = rdrobust(y=d18['voted'].values, x=d18['days_from_18'].values, c=0, masspoints="off")
rdd_70 = rdrobust(y=d70['voted'].values, x=d70['days_from_70'].values, c=0, masspoints="off")
_log(f"Optimal BW threshold 18: {float(rdd_18.bws.iloc[0, 0]):.1f} days")
_log(f"Optimal BW threshold 70: {float(rdd_70.bws.iloc[0, 0]):.1f} days")


# ── §2 — Bandwidth sensitivity, main estimates + bw_test.png ───────────────
fig_bw_main = f"{REP_FIGURES}/bandwidth_sensitivity.png"
fig_bw_test = f"{REP_FIGURES}/bw_test.png"
ckpt_bw_main = "bw_sweep_main"
if os.path.exists(fig_bw_main) and os.path.exists(fig_bw_test):
    _log("resume guard: bandwidth_sensitivity.png + bw_test.png exist, skipping")
else:
    _log("starting main bandwidth sweep (T18/T70, 20-300d step 20)")
    ckpt = _load_ckpt(ckpt_bw_main)
    main_specs = [
        ('T18', d18, 'days_from_18', 'red',    'Bandwidth Sensitivity — Threshold 18'),
        ('T70', d70, 'days_from_70', 'orange', 'Bandwidth Sensitivity — Threshold 70'),
    ]
    for key, d, rv, color, title in main_specs:
        if key in ckpt:
            _log(f"  {key}: loaded from checkpoint ({len(ckpt[key])} bw points)")
            continue
        _log(f"  {key}: running sweep ({len(BW_GRID)} bandwidths, n={len(d):,})")
        ckpt[key] = _bw_sweep(d, rv, BW_GRID, min_obs=500)
        _save_ckpt(ckpt_bw_main, ckpt)
        _log(f"  {key}: done, checkpointed")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (key, d, rv, color, title) in zip(axes, main_specs):
        df_bw = ckpt[key]
        ax.fill_between(df_bw['bw'], df_bw['ci_low'], df_bw['ci_high'], alpha=0.2, color=color)
        ax.plot(df_bw['bw'], df_bw['coef'], color=color, linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Bandwidth (days)")
        ax.set_ylabel("RD Estimate")
        ax.set_title(title)
    plt.suptitle("Sensitivity to Bandwidth Choice", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_bw_main, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_bw_main)}")

    # bw_test.png: same sweep, restricted to the bw in {20,40,60} subset
    # (see module docstring — this is what the curated artifact actually is).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (key, d, rv, color, title) in zip(axes, main_specs):
        sub = ckpt[key][ckpt[key]['bw'].isin([20, 40, 60])]
        ax.fill_between(sub['bw'], sub['ci_low'], sub['ci_high'], alpha=0.2, color=color)
        ax.plot(sub['bw'], sub['coef'], color=color, linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Bandwidth (days)")
        ax.set_ylabel("RD Estimate")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fig_bw_test, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_bw_test)}")
    _clear_ckpt(ckpt_bw_main)


# ── §3 — Placebo cutpoints, main estimates ──────────────────────────────
fig_pl_main = f"{REP_FIGURES}/placebo_cutpoints.png"
ckpt_pl_main = "placebo_main"
if os.path.exists(fig_pl_main):
    _log("resume guard: placebo_cutpoints.png exists, skipping")
else:
    _log("starting main placebo-cutpoint sweep (T18/T70)")
    ckpt = _load_ckpt(ckpt_pl_main)
    SAMPLE_SIZE = 300_000
    main_specs = [
        ('T18', d18, 'days_from_18', float(rdd_18.bws.iloc[0, 0]), 'red',
         'Placebo Cutpoints — Threshold 18'),
        ('T70', d70, 'days_from_70', float(rdd_70.bws.iloc[0, 0]), 'orange',
         'Placebo Cutpoints — Threshold 70'),
    ]
    for key, d, rv, true_bw, color, title in main_specs:
        if key in ckpt:
            _log(f"  {key}: loaded from checkpoint ({len(ckpt[key])} cutoffs)")
            continue
        d_sample = d.sample(n=min(SAMPLE_SIZE, len(d)), random_state=42)
        _log(f"  {key}: running placebo sweep (sample n={len(d_sample):,}, true_bw={true_bw:.1f}d)")
        ckpt[key] = _placebo_sweep(d_sample, rv, true_bw, step=60, min_obs=500)
        _save_ckpt(ckpt_pl_main, ckpt)
        _log(f"  {key}: done, checkpointed ({len(ckpt[key])} cutoffs)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (key, d, rv, true_bw, color, title) in zip(axes, main_specs):
        df_p = ckpt[key]
        ax.errorbar(df_p['cutoff'], df_p['coef'],
                    yerr=[df_p['coef'] - df_p['ci_low'], df_p['ci_high'] - df_p['coef']],
                    fmt='o', color='steelblue', capsize=3, markersize=4)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.axvline(0, color=color, linestyle='-', linewidth=1.5, label='True cutoff')
        ax.set_xlabel("Placebo cutoff (days from true threshold)")
        ax.set_ylabel("RD Estimate")
        ax.set_title(title)
        ax.legend()
    plt.suptitle("Placebo Cutpoint Tests", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_pl_main, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_pl_main)}")
    _clear_ckpt(ckpt_pl_main)


# ── §4 — rdplot visualisation ────────────────────────────────────────────
fig_rdplot18 = f"{REP_FIGURES}/rdplot_threshold_18.png"
if os.path.exists(fig_rdplot18):
    _log("resume guard: rdplot_threshold_18.png exists, skipping")
else:
    _log("starting rdplot — Threshold 18")
    res_18 = rdplot(y=d18["voted"].values, x=d18["days_from_18"].values, c=0,
                    masspoints="rd",
                    title="RD Plot — Threshold 18",
                    x_label="Days from 18-year threshold",
                    y_label="Turnout")
    res_18.rdplot.save(fig_rdplot18, bbox_inches="tight")
    _log(f"Saved: {os.path.basename(fig_rdplot18)}")

fig_rdplot70 = f"{REP_FIGURES}/rdplot_threshold_70.png"
if os.path.exists(fig_rdplot70):
    _log("resume guard: rdplot_threshold_70.png exists, skipping")
else:
    _log("starting rdplot — Threshold 70")
    res_70 = rdplot(y=d70["voted"].values, x=d70["days_from_70"].values, c=0,
                    masspoints="rd",
                    title="RD Plot — Threshold 70",
                    x_label="Days from 70-year threshold",
                    y_label="Turnout")
    res_70.rdplot.save(fig_rdplot70, bbox_inches="tight")
    _log(f"Saved: {os.path.basename(fig_rdplot70)}")


# ── §5 — Bandwidth sensitivity, NBI subgroups ───────────────────────────
fig_bw_nbi = f"{REP_FIGURES}/bandwidth_sensitivity_nbi.png"
ckpt_bw_nbi = "bw_sweep_nbi"
if os.path.exists(fig_bw_nbi):
    _log("resume guard: bandwidth_sensitivity_nbi.png exists, skipping")
else:
    _log("starting NBI-subgroup bandwidth sweep (T18/T70 x Low/High NBI)")
    ckpt = _load_ckpt(ckpt_bw_nbi)
    nbi_groups = [('Low NBI', True), ('High NBI', False)]
    colors_nbi = {'Low NBI': '#3498db', 'High NBI': '#e74c3c'}
    thr_specs = [
        ('T18', d18, 'days_from_18', 'Threshold 18'),
        ('T70', d70, 'days_from_70', 'Threshold 70'),
    ]
    for thr_key, d_full, rv, threshold_label in thr_specs:
        for label, is_low in nbi_groups:
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            if unit in ckpt:
                _log(f"  {unit}: loaded from checkpoint")
                continue
            mask = d_full['nbi'] <= nbi_median if is_low else d_full['nbi'] > nbi_median
            _log(f"  {unit}: running sweep (n={int(mask.sum()):,})")
            ckpt[unit] = _bw_sweep(d_full[mask], rv, BW_GRID, min_obs=200)
            _save_ckpt(ckpt_bw_nbi, ckpt)
            _log(f"  {unit}: done, checkpointed")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (thr_key, d_full, rv, threshold_label) in enumerate(thr_specs):
        for ax, (label, is_low) in zip(axes[row], nbi_groups):
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            df_bw = ckpt[unit]
            color = colors_nbi[label]
            ax.fill_between(df_bw['bw'], df_bw['ci_low'], df_bw['ci_high'], alpha=0.2, color=color)
            ax.plot(df_bw['bw'], df_bw['coef'], color=color, linewidth=2)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel("Bandwidth (days)")
            ax.set_ylabel("RD Estimate")
            ax.set_title(f"{label} — {threshold_label}")
    plt.suptitle("Bandwidth Sensitivity — NBI Subgroups", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_bw_nbi, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_bw_nbi)}")
    _clear_ckpt(ckpt_bw_nbi)


# ── §6 — Placebo cutpoints, NBI subgroups ───────────────────────────────
fig_pl_nbi = f"{REP_FIGURES}/placebo_cutpoints_nbi.png"
ckpt_pl_nbi = "placebo_nbi"
if os.path.exists(fig_pl_nbi):
    _log("resume guard: placebo_cutpoints_nbi.png exists, skipping")
else:
    _log("starting NBI-subgroup placebo sweep (T18/T70 x Low/High NBI)")
    ckpt = _load_ckpt(ckpt_pl_nbi)
    SAMPLE_SIZE = 200_000
    nbi_groups = [('Low NBI', True), ('High NBI', False)]
    thr_specs = [
        ('T18', d18, 'days_from_18', float(rdd_18.bws.iloc[0, 0]), 'Threshold 18'),
        ('T70', d70, 'days_from_70', float(rdd_70.bws.iloc[0, 0]), 'Threshold 70'),
    ]
    for thr_key, d_full, rv, true_bw, threshold_label in thr_specs:
        for label, is_low in nbi_groups:
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            if unit in ckpt:
                _log(f"  {unit}: loaded from checkpoint ({len(ckpt[unit])} cutoffs)")
                continue
            mask = d_full['nbi'] <= nbi_median if is_low else d_full['nbi'] > nbi_median
            d_sub = d_full[mask].sample(n=min(SAMPLE_SIZE, int(mask.sum())), random_state=42)
            _log(f"  {unit}: running placebo sweep (sample n={len(d_sub):,})")
            ckpt[unit] = _placebo_sweep(d_sub, rv, true_bw, step=90, min_obs=200)
            _save_ckpt(ckpt_pl_nbi, ckpt)
            _log(f"  {unit}: done, checkpointed")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (thr_key, d_full, rv, true_bw, threshold_label) in enumerate(thr_specs):
        for ax, (label, is_low) in zip(axes[row], nbi_groups):
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            df_p = ckpt[unit]
            ax.errorbar(df_p['cutoff'], df_p['coef'],
                        yerr=[df_p['coef'] - df_p['ci_low'], df_p['ci_high'] - df_p['coef']],
                        fmt='o', color='steelblue', capsize=3, markersize=4)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='red', linestyle='-', linewidth=1.5, label='True cutoff')
            ax.set_xlabel("Placebo cutoff (days from true threshold)")
            ax.set_ylabel("RD Estimate")
            ax.set_title(f"{label} — {threshold_label}")
            ax.legend()
    plt.suptitle("Placebo Cutpoint Tests — NBI Subgroups", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_pl_nbi, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_pl_nbi)}")
    _clear_ckpt(ckpt_pl_nbi)


# ── §7 — Bandwidth sensitivity, PCA subgroups ───────────────────────────
fig_bw_pca = f"{REP_FIGURES}/bandwidth_sensitivity_pca.png"
ckpt_bw_pca = "bw_sweep_pca"
if os.path.exists(fig_bw_pca):
    _log("resume guard: bandwidth_sensitivity_pca.png exists, skipping")
else:
    _log("starting PCA-subgroup bandwidth sweep (T18/T70 x Low/High PCA)")
    ckpt = _load_ckpt(ckpt_bw_pca)
    pca_groups = [('Low PCA', True), ('High PCA', False)]
    colors_pca = {'Low PCA': '#2ecc71', 'High PCA': '#e67e22'}
    thr_specs = [
        ('T18', d18, 'days_from_18', 'Threshold 18'),
        ('T70', d70, 'days_from_70', 'Threshold 70'),
    ]
    for thr_key, d_full, rv, threshold_label in thr_specs:
        for label, is_low in pca_groups:
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            if unit in ckpt:
                _log(f"  {unit}: loaded from checkpoint")
                continue
            mask = d_full['pca_index'] <= pca_median if is_low else d_full['pca_index'] > pca_median
            _log(f"  {unit}: running sweep (n={int(mask.sum()):,})")
            ckpt[unit] = _bw_sweep(d_full[mask], rv, BW_GRID, min_obs=200)
            _save_ckpt(ckpt_bw_pca, ckpt)
            _log(f"  {unit}: done, checkpointed")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (thr_key, d_full, rv, threshold_label) in enumerate(thr_specs):
        for ax, (label, is_low) in zip(axes[row], pca_groups):
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            df_bw = ckpt[unit]
            color = colors_pca[label]
            ax.fill_between(df_bw['bw'], df_bw['ci_low'], df_bw['ci_high'], alpha=0.2, color=color)
            ax.plot(df_bw['bw'], df_bw['coef'], color=color, linewidth=2)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel("Bandwidth (days)")
            ax.set_ylabel("RD Estimate")
            ax.set_title(f"{label} — {threshold_label}")
    plt.suptitle("Bandwidth Sensitivity — PCA Subgroups", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_bw_pca, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_bw_pca)}")
    _clear_ckpt(ckpt_bw_pca)


# ── §8 — Placebo cutpoints, PCA subgroups ───────────────────────────────
fig_pl_pca = f"{REP_FIGURES}/placebo_cutpoints_pca.png"
ckpt_pl_pca = "placebo_pca"
if os.path.exists(fig_pl_pca):
    _log("resume guard: placebo_cutpoints_pca.png exists, skipping")
else:
    _log("starting PCA-subgroup placebo sweep (T18/T70 x Low/High PCA)")
    ckpt = _load_ckpt(ckpt_pl_pca)
    SAMPLE_SIZE = 200_000
    pca_groups = [('Low PCA', True), ('High PCA', False)]
    thr_specs = [
        ('T18', d18, 'days_from_18', float(rdd_18.bws.iloc[0, 0]), 'Threshold 18'),
        ('T70', d70, 'days_from_70', float(rdd_70.bws.iloc[0, 0]), 'Threshold 70'),
    ]
    for thr_key, d_full, rv, true_bw, threshold_label in thr_specs:
        for label, is_low in pca_groups:
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            if unit in ckpt:
                _log(f"  {unit}: loaded from checkpoint ({len(ckpt[unit])} cutoffs)")
                continue
            mask = d_full['pca_index'] <= pca_median if is_low else d_full['pca_index'] > pca_median
            d_sub = d_full[mask].sample(n=min(SAMPLE_SIZE, int(mask.sum())), random_state=42)
            _log(f"  {unit}: running placebo sweep (sample n={len(d_sub):,})")
            ckpt[unit] = _placebo_sweep(d_sub, rv, true_bw, step=90, min_obs=200)
            _save_ckpt(ckpt_pl_pca, ckpt)
            _log(f"  {unit}: done, checkpointed")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (thr_key, d_full, rv, true_bw, threshold_label) in enumerate(thr_specs):
        for ax, (label, is_low) in zip(axes[row], pca_groups):
            unit = f"{thr_key}_{'low' if is_low else 'high'}"
            df_p = ckpt[unit]
            ax.errorbar(df_p['cutoff'], df_p['coef'],
                        yerr=[df_p['coef'] - df_p['ci_low'], df_p['ci_high'] - df_p['coef']],
                        fmt='o', color='steelblue', capsize=3, markersize=4)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='red', linestyle='-', linewidth=1.5, label='True cutoff')
            ax.set_xlabel("Placebo cutoff (days from true threshold)")
            ax.set_ylabel("RD Estimate")
            ax.set_title(f"{label} — {threshold_label}")
            ax.legend()
    plt.suptitle("Placebo Cutpoint Tests — PCA Subgroups", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_pl_pca, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_pl_pca)}")
    _clear_ckpt(ckpt_pl_pca)


# ── §9 — Integer-day heaping and the density test at placebo cutoffs ─────
# The Cattaneo-Jansson-Ma density test rejects continuity at age 18. With a
# non-manipulable running variable (date of birth) that cannot be sorting, and
# the manuscript attributes it to integer-day heaping plus the test's
# hypersensitivity at this N. This block evidences that claim instead of
# asserting it: panel A shows the heaping itself, panel B re-runs the SAME test
# on the SAME sample with only the cutoff moved, where no legal threshold and so
# no manipulation can exist. Rejection at those placebo cutoffs shows the
# statistic is picking up the variable's granularity, not the threshold.
fig_heap = f"{REP_FIGURES}/density_heaping_placebo_T18.png"
csv_heap = f"{REP_TABLES}/density_placebo_T18.csv"

if os.path.exists(fig_heap) and os.path.exists(csv_heap):
    _log("resume guard: density_heaping_placebo_T18.png + .csv exist, skipping")
else:
    _log("starting heaping / density-placebo block (Sec. 9)")
    from rddensity import rddensity

    x18 = d18['days_from_18'].to_numpy(dtype=float)

    # Same sample and same test as the headline density check; only c moves.
    PLACEBO_CUTOFFS = [-540, -360, -180, 0, 180, 360, 540]
    dens_rows = []
    for c_p in PLACEBO_CUTOFFS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = rddensity(x18, c=c_p)
        t_jk = float(r.test['t_jk'])
        p_jk = float(r.test['p_jk'])
        dens_rows.append({'cutoff': c_p, 't_jk': t_jk, 'p_jk': p_jk,
                          'is_true_cutoff': c_p == 0})
        _log(f"  density test at c={c_p:+5d}: T={t_jk:+.2f}, p={p_jk:.4f}")

    df_dens = pd.DataFrame(dens_rows)
    df_dens.to_csv(csv_heap, index=False)
    _log(f"Saved: {os.path.basename(csv_heap)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A — the heaping itself, 1-day bins, both sides of the cutoff.
    ax = axes[0]
    win = 365
    xs = x18[(x18 >= -win) & (x18 <= win)]
    ax.hist(xs, bins=np.arange(-win - 0.5, win + 1.5, 1.0),
            color='steelblue', linewidth=0)
    ax.axvline(0, color='red', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Days from 18th birthday on election day')
    ax.set_ylabel('Registered voters (1-day bins)')
    ax.set_title('A. Integer-day heaping in the running variable', fontsize=10)

    # Panel B — the same density test with only the cutoff moved.
    ax = axes[1]
    true_row = df_dens[df_dens['is_true_cutoff']]
    plac_row = df_dens[~df_dens['is_true_cutoff']]
    ax.scatter(plac_row['cutoff'], plac_row['t_jk'].abs(), color='steelblue',
               s=45, zorder=3, label='Placebo cutoff (no legal threshold)')
    ax.scatter(true_row['cutoff'], true_row['t_jk'].abs(), color='red',
               s=70, zorder=4, label='True cutoff (age 18)')
    ax.axhline(1.96, color='black', linestyle='--', linewidth=0.9)
    ax.annotate('5% critical value', (ax.get_xlim()[0], 1.96),
                textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.set_xlabel('Cutoff at which the density test is run (days)')
    ax.set_ylabel('|T| (jackknife)')
    ax.set_title('B. Density test moved away from the threshold', fontsize=10)
    ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(fig_heap, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {os.path.basename(fig_heap)}")


print("\n07_robustness: DONE")
