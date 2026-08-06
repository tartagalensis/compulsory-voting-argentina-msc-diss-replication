"""02 — Main RDD: Table 2 (rdd_results) and the main-threshold RD artifacts.
Transcribed from notebooks/04_rdd.ipynb (all code cells, post-1b06b5a state):
helpers (cell 2), data load (cell 3), covariate balance figure (cell 5) and
formal balance test (cell 7), main RDD at T18/T70 (cells 9/11), gender
heterogeneity at T18/T70 (cells 13/15) — all four using the base-then-
clustered pattern (an unclustered rdrobust selects the MSE-optimal h/b, then
a second call with cluster=... re-estimates the variance at those fixed
bandwidths) — NBI/PCA heterogeneity and gender x NBI/PCA interactions at
T18/T70 (cells 17-31, cluster= without fixing h, masspoints 'off' for the
single-SES splits and 'rd' for the interactions), CSV export (cell 33) and
LaTeX export with the baked-in [ht]/\\centering/\\resizebox wrapper (cell 35).

Only the data-load cell changes in substance: cell 3 of the notebook reads
the full national.parquet, computes nbi_median/pca_median on the FULL frame,
then windows into d18/d70. Here that full-frame median pass is a separate
column-scoped read (nbi, pca_index only); the windowed frame comes from
load_window(['voted', 'gender', 'nbi', 'pca_index']), which is memory-safe
(parquet predicate pushdown to the union of the +-3y windows) and yields the
same cluster_id partition as the notebook's ngroup() (see _common.py).

Also appends a density-test table (run_density_test.py's exact invocation of
src.analysis.density_test), reusing the already-windowed d18/d70 (idempotent
re-filter to the same +-3y window as run_density_test.py's own read).
"""
import gc
import os
from datetime import datetime

from _common import setup, load_window, REP_FIGURES, REP_TABLES  # noqa: F401
setup()


def _log(msg):
    """Timestamped, flushed progress line — localizes the failure if a run dies."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from rdrobust import rdrobust
from config import FINAL_DIR
from src.analysis import density_test, plot_rdd_visual

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})


# ── HELPER: bin plot ────────────────────────────────────────────────────────
def plot_rdd(ax, data, running_var, outcome, threshold_label, color, bin_size=30, bw=365 * 3):
    d = data[data[running_var].between(-bw, bw)].copy()
    d['bin'] = (d[running_var] // bin_size) * bin_size
    bins = d.groupby('bin')[outcome].mean().reset_index()
    left  = bins[bins['bin'] < 0]
    right = bins[bins['bin'] >= 0]
    for side in [left, right]:
        z = np.polyfit(side['bin'], side[outcome] * 100, 1)
        ax.plot(side['bin'], np.poly1d(z)(side['bin']), color='steelblue', linewidth=2)
    ax.scatter(bins['bin'], bins[outcome] * 100, s=15, color='steelblue', alpha=0.7, zorder=5)
    ax.axvline(0, color=color, linestyle='--', linewidth=1.5, label=threshold_label)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())


# ── HELPER: extract rdrobust results ─────────────────────────────────────
def extract_rdd(rdd_obj, threshold, subgroup):
    return {
        'threshold':        threshold,
        'subgroup':         subgroup,
        'coef':             round(float(rdd_obj.coef.iloc[0, 0]), 4),
        'se':               round(float(rdd_obj.se.iloc[0, 0]), 4),
        't_stat':           round(float(rdd_obj.t.iloc[0, 0]), 3),
        'p_value':          round(float(rdd_obj.pv.iloc[0, 0]), 6),
        'ci_low':           round(float(rdd_obj.ci.iloc[0, 0]), 4),
        'ci_high':          round(float(rdd_obj.ci.iloc[0, 1]), 4),
        'robust_ci_low':    round(float(rdd_obj.ci.iloc[2, 0]), 4),
        'robust_ci_high':   round(float(rdd_obj.ci.iloc[2, 1]), 4),
        'bw':               round(float(rdd_obj.bws.iloc[0, 0]), 1),
        'n_eff_left':       int(rdd_obj.N_h[0]),
        'n_eff_right':      int(rdd_obj.N_h[1]),
    }


# ── Checkpointing helpers ────────────────────────────────────────────────
# The 30-estimate loop below used to accumulate `results` purely in memory
# and write rdd_results.csv only at the very end; a kill mid-run forced a
# full recompute (~50 min). These helpers extend the covariate-balance
# section's RESUME GUARD pattern to the estimate blocks: after each of the
# six estimate "kinds" (Total, gender, NBI, PCA, gender x NBI interactions,
# gender x PCA interactions — each spanning both T18 and T70), the
# accumulated `results` are flushed to a partial checkpoint CSV; on restart
# that checkpoint is preloaded and any (threshold, subgroup) pair already
# present is skipped.
PARTIAL_RESULTS_PATH = f"{REP_TABLES}/rdd_results.partial.csv"


def _load_partial(path):
    """Preload accumulated results from a partial checkpoint CSV.

    Returns a list of row-dicts (the same shape `results.append(...)`
    produces), or [] if no checkpoint exists yet.
    """
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict('records')


def _done_pairs(results):
    """Set of (threshold, subgroup) pairs already present in `results`."""
    return {(r['threshold'], r['subgroup']) for r in results}


def _flush_partial(results, path):
    """Write the accumulated `results` to the partial checkpoint CSV.

    Called after each estimate block so a kill loses at most one block
    instead of the full 30-estimate run.
    """
    pd.DataFrame(results).to_csv(path, index=False)


results = _load_partial(PARTIAL_RESULTS_PATH)
done_pairs = _done_pairs(results)
if done_pairs:
    _log(f"resume guard: preloaded {len(results)} estimate(s) from "
         f"rdd_results.partial.csv — pairs already done: {sorted(done_pairs)}")

# ── Data load ────────────────────────────────────────────────────────────
# nbi_median / pca_median are computed on the FULL frame in the notebook
# (before windowing) — reproduce that with a separate column-scoped full pass.
df_medians = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['nbi', 'pca_index'])
nbi_median = df_medians['nbi'].median()
pca_median = df_medians['pca_index'].median()
del df_medians
gc.collect()

BW_MAX = 365 * 3
df = load_window(['voted', 'gender', 'nbi', 'pca_index'])
print(f"Records (windowed union of T18/T70 +-3y): {len(df):,}")

d18 = df[df['days_from_18'].between(-BW_MAX, BW_MAX)].copy()
d70 = df[df['days_from_70'].between(-BW_MAX, BW_MAX)].copy()
del df
gc.collect()

print(f"Observations near threshold 18: {len(d18):,}")
print(f"Observations near threshold 70: {len(d70):,}")
print(f"NBI median: {nbi_median:.4f}")
print(f"PCA median: {pca_median:.4f}")
_log("data loaded and windowed")

# ── RD visuals: rdd_threshold_18/70.png ─────────────────────────────────
# Origin: 03_analysis.ipynb cell 10 (plot_rdd_visual on the full register),
# owned by this script in the replication package task map. plot_rdd_visual
# itself restricts to the +-3y window of its running variable, so calling it
# on d18/d70 (exactly those windows) is equivalent to the notebook call.
# (RESUME GUARD: skip if both figures already exist from a killed run.)
if (os.path.exists(f"{REP_FIGURES}/rdd_threshold_18.png")
        and os.path.exists(f"{REP_FIGURES}/rdd_threshold_70.png")):
    _log("resume guard: rdd_threshold_18/70.png exist, skipping RD visuals")
else:
    plot_rdd_visual(d18, 'days_from_18', 'Threshold: age 18', 'red',
                    REP_FIGURES, filename='rdd_threshold_18.png')
    plot_rdd_visual(d70, 'days_from_70', 'Threshold: age 70', 'orange',
                    REP_FIGURES, filename='rdd_threshold_70.png')
    _log("rdd_threshold_18/70.png written")

# ── Covariate balance figure ────────────────────────────────────────────
# (RESUME GUARD: a previous run may have been killed after finishing this
# section — the figure is the first artifact written; skip if it already
# exists so a relaunch does not redo finished work.)
if os.path.exists(f"{REP_FIGURES}/covariate_balance.png"):
    _log("resume guard: covariate_balance.png exists, skipping balance figure")
else:
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for (ax, d, col, ylabel, title, color) in [
        (axes[0, 0], d18, 'gender',    '% Female',  'Gender Balance — Threshold 18', 'red'),
        (axes[0, 1], d70, 'gender',    '% Female',  'Gender Balance — Threshold 70', 'orange'),
        (axes[1, 0], d18, 'nbi',       'NBI (%)',   'NBI Balance — Threshold 18',    'red'),
        (axes[1, 1], d70, 'nbi',       'NBI (%)',   'NBI Balance — Threshold 70',    'orange'),
        (axes[2, 0], d18, 'pca_index', 'PCA index', 'PCA Balance — Threshold 18',    'red'),
        (axes[2, 1], d70, 'pca_index', 'PCA index', 'PCA Balance — Threshold 70',    'orange'),
    ]:
        rv = 'days_from_18' if color == 'red' else 'days_from_70'
        d2 = d.copy()
        d2['bin'] = (d2[rv] // 30) * 30
        if col == 'gender':
            d2['val'] = (d2['gender'] == 'F').astype(int)
            scale = 100
        elif col == 'nbi':
            d2['val'] = d2['nbi']
            scale = 100
        else:
            d2['val'] = d2['pca_index']
            scale = 1
        bins = d2.groupby('bin')['val'].mean().reset_index()
        ax.scatter(bins['bin'], bins['val'] * scale, s=15, color='steelblue', alpha=0.7)
        ax.axvline(0, color=color, linestyle='--', linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Days from threshold")
        ax.set_ylabel(ylabel)
        if col in ('gender', 'nbi'):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.suptitle("Covariate Balance at RDD Thresholds", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{REP_FIGURES}/covariate_balance.png", bbox_inches='tight')
    plt.close(fig)
    _log("covariate_balance.png written")

# ── Formal covariate balance test ───────────────────────────────────────
# (RESUME GUARD: skip if the CSV already exists — the section's only output.)
if os.path.exists(f"{REP_TABLES}/covariate_balance.csv"):
    _log("resume guard: covariate_balance.csv exists, skipping formal balance test")
else:
    print("Formal covariate balance — rdrobust with covariates as outcome")
    print("=" * 60)
    SAMPLE = 500_000
    balance_results = []
    for label, d, rv in [("T18", d18, "days_from_18"), ("T70", d70, "days_from_70")]:
        d_s = d.sample(n=min(SAMPLE, len(d)), random_state=42)
        for cov_label, y_vals in [
            ("% Female",  (d_s["gender"] == "F").astype(float).values),
            ("NBI",        d_s["nbi"].values),
            ("PCA index",  d_s["pca_index"].values),
        ]:
            r = rdrobust(y=y_vals, x=d_s[rv].values, c=0, masspoints="off")
            coef  = float(r.coef.iloc[0, 0])
            pval  = float(r.pv.iloc[0, 0])
            ci_lo = float(r.ci.iloc[2, 0])
            ci_hi = float(r.ci.iloc[2, 1])
            status = "OK" if pval > 0.05 else "FAIL"
            print(f"  [{status}] {label} | {cov_label:<12} coef={coef:+.4f}  p={pval:.3f}  robust CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
            balance_results.append({"threshold": label, "covariate": cov_label,
                                     "coef": round(coef, 4), "p_value": round(pval, 4),
                                     "robust_ci_low": round(ci_lo, 4), "robust_ci_high": round(ci_hi, 4)})
            _log(f"balance estimate done: {label} | {cov_label}")
    pd.DataFrame(balance_results).to_csv(f"{REP_TABLES}/covariate_balance.csv", index=False)
    print("Saved: covariate_balance.csv")

# ── RDD — THRESHOLD 18 ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("RDD — THRESHOLD 18")
print("(SEs clustered by circuit at the base-run MSE-optimal h/b)")
print("=" * 50)

# Base (unclustered) run selects the MSE-optimal h/b; the clustered run
# re-estimates the variance at those fixed bandwidths, so the point estimate,
# BW and effective N are unchanged by construction. Clustering by circuit is
# warranted for every subgroup, not only the SES ones: within-circuit
# correlation (shared polling places, local mobilization, circuit-level
# enforcement of sanctions) does not depend on whether SES enters the
# particular equation.
if (18, 'Total') in done_pairs:
    _log("resume guard: T18 Total already in partial results, skipping")
else:
    rdd_18_base = rdrobust(y=d18['voted'].values, x=d18['days_from_18'].values, c=0, masspoints="off")
    rdd_18 = rdrobust(y=d18['voted'].values, x=d18['days_from_18'].values, c=0, masspoints="off",
                     cluster=d18['cluster_id'].values,
                     h=float(rdd_18_base.bws.iloc[0, 0]), b=float(rdd_18_base.bws.iloc[1, 0]))
    results.append(extract_rdd(rdd_18, 18, 'Total'))
    done_pairs.add((18, 'Total'))
    _log("estimate done: T18 Total")
    print(rdd_18)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_rdd(ax, d18, 'days_from_18', 'voted', 'Threshold: age 18', 'red')
    ax.set_xlabel("Days from 18-year threshold")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("RDD — Threshold 18: Effect of Compulsory Voting on Turnout")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REP_FIGURES}/rdd_main_18.png", bbox_inches='tight')
    plt.close(fig)

# ── RDD — THRESHOLD 70 ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("RDD — THRESHOLD 70")
print("(SEs clustered by circuit at the base-run MSE-optimal h/b)")
print("=" * 50)

# Base (unclustered) run selects the MSE-optimal h/b; the clustered run
# re-estimates the variance at those fixed bandwidths, so the point estimate,
# BW and effective N are unchanged by construction. Clustering by circuit is
# warranted for every subgroup, not only the SES ones: within-circuit
# correlation (shared polling places, local mobilization, circuit-level
# enforcement of sanctions) does not depend on whether SES enters the
# particular equation.
if (70, 'Total') in done_pairs:
    _log("resume guard: T70 Total already in partial results, skipping")
else:
    rdd_70_base = rdrobust(y=d70['voted'].values, x=d70['days_from_70'].values, c=0, masspoints="off")
    rdd_70 = rdrobust(y=d70['voted'].values, x=d70['days_from_70'].values, c=0, masspoints="off",
                     cluster=d70['cluster_id'].values,
                     h=float(rdd_70_base.bws.iloc[0, 0]), b=float(rdd_70_base.bws.iloc[1, 0]))
    results.append(extract_rdd(rdd_70, 70, 'Total'))
    done_pairs.add((70, 'Total'))
    _log("estimate done: T70 Total")
    print(rdd_70)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_rdd(ax, d70, 'days_from_70', 'voted', 'Threshold: age 70', 'orange')
    ax.set_xlabel("Days from 70-year threshold")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("RDD — Threshold 70: Effect of Compulsory Voting on Turnout")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REP_FIGURES}/rdd_main_70.png", bbox_inches='tight')
    plt.close(fig)

# Block 1/6 (Total, T18+T70) complete — checkpoint.
_flush_partial(results, PARTIAL_RESULTS_PATH)
_log(f"checkpoint: block 1/6 (Total) flushed to rdd_results.partial.csv ({len(results)} total)")

# ── HETEROGENEOUS EFFECTS BY GENDER — THRESHOLD 18 ──────────────────────
print("\n" + "=" * 50)
print("HETEROGENEOUS EFFECTS BY GENDER — THRESHOLD 18")
print("(SEs clustered by circuit at the base-run MSE-optimal h/b)")
print("=" * 50)

# Clustered by circuit at the base-run MSE-optimal h/b (see Total cells above):
# point estimate, BW and effective N unchanged; only the variance is re-estimated.
for gender in ['F', 'M']:
    label = 'Female' if gender == 'F' else 'Male'
    if (18, label) in done_pairs:
        _log(f"resume guard: T18 {label} already in partial results, skipping")
        continue
    d = d18[d18['gender'] == gender]
    rdd_base = rdrobust(y=d['voted'].values, x=d['days_from_18'].values, c=0, masspoints="off")
    rdd = rdrobust(y=d['voted'].values, x=d['days_from_18'].values, c=0, masspoints="off",
                   cluster=d['cluster_id'].values,
                   h=float(rdd_base.bws.iloc[0, 0]), b=float(rdd_base.bws.iloc[1, 0]))
    results.append(extract_rdd(rdd, 18, label))
    done_pairs.add((18, label))
    _log(f"estimate done: T18 {label}")
    print(f"\n{label}:")
    print(rdd)

# ── HETEROGENEOUS EFFECTS BY GENDER — THRESHOLD 70 ──────────────────────
print("\n" + "=" * 50)
print("HETEROGENEOUS EFFECTS BY GENDER — THRESHOLD 70")
print("(SEs clustered by circuit at the base-run MSE-optimal h/b)")
print("=" * 50)

# Clustered by circuit at the base-run MSE-optimal h/b (see Total cells above):
# point estimate, BW and effective N unchanged; only the variance is re-estimated.
for gender in ['F', 'M']:
    label = 'Female' if gender == 'F' else 'Male'
    if (70, label) in done_pairs:
        _log(f"resume guard: T70 {label} already in partial results, skipping")
        continue
    d = d70[d70['gender'] == gender]
    rdd_base = rdrobust(y=d['voted'].values, x=d['days_from_70'].values, c=0, masspoints="off")
    rdd = rdrobust(y=d['voted'].values, x=d['days_from_70'].values, c=0, masspoints="off",
                   cluster=d['cluster_id'].values,
                   h=float(rdd_base.bws.iloc[0, 0]), b=float(rdd_base.bws.iloc[1, 0]))
    results.append(extract_rdd(rdd, 70, label))
    done_pairs.add((70, label))
    _log(f"estimate done: T70 {label}")
    print(f"\n{label}:")
    print(rdd)

# Block 2/6 (gender, T18+T70) complete — checkpoint.
_flush_partial(results, PARTIAL_RESULTS_PATH)
_log(f"checkpoint: block 2/6 (gender) flushed to rdd_results.partial.csv ({len(results)} total)")

# ── HETEROGENEOUS EFFECTS BY NBI — THRESHOLD 18 ─────────────────────────
print("\n" + "=" * 50)
print("HETEROGENEOUS EFFECTS BY NBI — THRESHOLD 18")
print("(SEs clustered by circuit)")
print("=" * 50)

for label, mask in [('Low NBI', d18['nbi'] <= nbi_median), ('High NBI', d18['nbi'] > nbi_median)]:
    if (18, label) in done_pairs:
        _log(f"resume guard: T18 {label} already in partial results, skipping")
        continue
    d = d18[mask]
    rdd = rdrobust(y=d['voted'].values, x=d['days_from_18'].values, c=0, masspoints="off",
                   cluster=d['cluster_id'].values)
    results.append(extract_rdd(rdd, 18, label))
    done_pairs.add((18, label))
    _log(f"estimate done: T18 {label}")
    print(f"\n{label}:")
    print(rdd)

# ── HETEROGENEOUS EFFECTS BY NBI — THRESHOLD 70 ─────────────────────────
print("\n" + "=" * 50)
print("HETEROGENEOUS EFFECTS BY NBI — THRESHOLD 70")
print("(SEs clustered by circuit)")
print("=" * 50)

for label, mask in [('Low NBI', d70['nbi'] <= nbi_median), ('High NBI', d70['nbi'] > nbi_median)]:
    if (70, label) in done_pairs:
        _log(f"resume guard: T70 {label} already in partial results, skipping")
        continue
    d = d70[mask]
    rdd = rdrobust(y=d['voted'].values, x=d['days_from_70'].values, c=0, masspoints="off",
                   cluster=d['cluster_id'].values)
    results.append(extract_rdd(rdd, 70, label))
    done_pairs.add((70, label))
    _log(f"estimate done: T70 {label}")
    print(f"\n{label}:")
    print(rdd)

# Block 3/6 (NBI, T18+T70) complete — checkpoint.
_flush_partial(results, PARTIAL_RESULTS_PATH)
_log(f"checkpoint: block 3/6 (NBI) flushed to rdd_results.partial.csv ({len(results)} total)")

# ── HETEROGENEOUS EFFECTS BY PCA — THRESHOLD 18 ─────────────────────────
print("\n" + "=" * 50)
print("HETEROGENEOUS EFFECTS BY PCA — THRESHOLD 18")
print("(SEs clustered by circuit)")
print("=" * 50)

for label, mask in [('Low PCA', d18['pca_index'] <= pca_median), ('High PCA', d18['pca_index'] > pca_median)]:
    if (18, label) in done_pairs:
        _log(f"resume guard: T18 {label} already in partial results, skipping")
        continue
    d = d18[mask]
    rdd = rdrobust(y=d['voted'].values, x=d['days_from_18'].values, c=0, masspoints="off",
                   cluster=d['cluster_id'].values)
    results.append(extract_rdd(rdd, 18, label))
    done_pairs.add((18, label))
    _log(f"estimate done: T18 {label}")
    print(f"\n{label}:")
    print(rdd)

# ── HETEROGENEOUS EFFECTS BY PCA — THRESHOLD 70 ─────────────────────────
print("\n" + "=" * 50)
print("HETEROGENEOUS EFFECTS BY PCA — THRESHOLD 70")
print("(SEs clustered by circuit)")
print("=" * 50)

for label, mask in [('Low PCA', d70['pca_index'] <= pca_median), ('High PCA', d70['pca_index'] > pca_median)]:
    if (70, label) in done_pairs:
        _log(f"resume guard: T70 {label} already in partial results, skipping")
        continue
    d = d70[mask]
    rdd = rdrobust(y=d['voted'].values, x=d['days_from_70'].values, c=0, masspoints="off",
                   cluster=d['cluster_id'].values)
    results.append(extract_rdd(rdd, 70, label))
    done_pairs.add((70, label))
    _log(f"estimate done: T70 {label}")
    print(f"\n{label}:")
    print(rdd)

# Block 4/6 (PCA, T18+T70) complete — checkpoint.
_flush_partial(results, PARTIAL_RESULTS_PATH)
_log(f"checkpoint: block 4/6 (PCA) flushed to rdd_results.partial.csv ({len(results)} total)")

# ── GENDER x NBI INTERACTIONS — THRESHOLD 18 ────────────────────────────
print("\n" + "=" * 60)
print("GENDER x NBI INTERACTIONS — THRESHOLD 18  (SEs clustered by circuit)")
print("=" * 60)
d = d18
for label, mask in [
    ("Female x Low NBI",  (d["gender"] == "F") & (d["nbi"] <= nbi_median)),
    ("Female x High NBI", (d["gender"] == "F") & (d["nbi"] >  nbi_median)),
    ("Male x Low NBI",    (d["gender"] == "M") & (d["nbi"] <= nbi_median)),
    ("Male x High NBI",   (d["gender"] == "M") & (d["nbi"] >  nbi_median)),
]:
    if (18, label) in done_pairs:
        _log(f"resume guard: T18 {label} already in partial results, skipping")
        continue
    dd = d[mask]
    rv = "days_from_18"
    if len(dd) < 500:
        print(f"  {label}: too few obs ({len(dd):,}), skipping")
        continue
    r = rdrobust(y=dd["voted"].values, x=dd[rv].values, c=0, masspoints="rd",
                 cluster=dd['cluster_id'].values)
    res = extract_rdd(r, threshold=18, subgroup=label)
    results.append(res)
    done_pairs.add((18, label))
    _log(f"estimate done: T18 {label}")
    print(f"  {label}: coef={res['coef']*100:+.2f}pp  se={res['se']*100:.2f}  p={res['p_value']:.4f}")

# ── GENDER x NBI INTERACTIONS — THRESHOLD 70 ────────────────────────────
print("\n" + "=" * 60)
print("GENDER x NBI INTERACTIONS — THRESHOLD 70  (SEs clustered by circuit)")
print("=" * 60)
d = d70
for label, mask in [
    ("Female x Low NBI",  (d["gender"] == "F") & (d["nbi"] <= nbi_median)),
    ("Female x High NBI", (d["gender"] == "F") & (d["nbi"] >  nbi_median)),
    ("Male x Low NBI",    (d["gender"] == "M") & (d["nbi"] <= nbi_median)),
    ("Male x High NBI",   (d["gender"] == "M") & (d["nbi"] >  nbi_median)),
]:
    if (70, label) in done_pairs:
        _log(f"resume guard: T70 {label} already in partial results, skipping")
        continue
    dd = d[mask]
    rv = "days_from_70"
    if len(dd) < 500:
        print(f"  {label}: too few obs ({len(dd):,}), skipping")
        continue
    r = rdrobust(y=dd["voted"].values, x=dd[rv].values, c=0, masspoints="rd",
                 cluster=dd['cluster_id'].values)
    res = extract_rdd(r, threshold=70, subgroup=label)
    results.append(res)
    done_pairs.add((70, label))
    _log(f"estimate done: T70 {label}")
    print(f"  {label}: coef={res['coef']*100:+.2f}pp  se={res['se']*100:.2f}  p={res['p_value']:.4f}")

# Block 5/6 (gender x NBI interactions, T18+T70) complete — checkpoint.
_flush_partial(results, PARTIAL_RESULTS_PATH)
_log(f"checkpoint: block 5/6 (gender x NBI) flushed to rdd_results.partial.csv ({len(results)} total)")

# ── GENDER x PCA INTERACTIONS — THRESHOLD 18 ────────────────────────────
print("\n" + "=" * 60)
print("GENDER x PCA INTERACTIONS — THRESHOLD 18  (SEs clustered by circuit)")
print("=" * 60)
d = d18
for label, mask in [
    ("Female x Low PCA",  (d["gender"] == "F") & (d["pca_index"] <= pca_median)),
    ("Female x High PCA", (d["gender"] == "F") & (d["pca_index"] >  pca_median)),
    ("Male x Low PCA",    (d["gender"] == "M") & (d["pca_index"] <= pca_median)),
    ("Male x High PCA",   (d["gender"] == "M") & (d["pca_index"] >  pca_median)),
]:
    if (18, label) in done_pairs:
        _log(f"resume guard: T18 {label} already in partial results, skipping")
        continue
    dd = d[mask]
    rv = "days_from_18"
    if len(dd) < 500:
        print(f"  {label}: too few obs ({len(dd):,}), skipping")
        continue
    r = rdrobust(y=dd["voted"].values, x=dd[rv].values, c=0, masspoints="rd",
                 cluster=dd['cluster_id'].values)
    res = extract_rdd(r, threshold=18, subgroup=label)
    results.append(res)
    done_pairs.add((18, label))
    _log(f"estimate done: T18 {label}")
    print(f"  {label}: coef={res['coef']*100:+.2f}pp  se={res['se']*100:.2f}  p={res['p_value']:.4f}")

# ── GENDER x PCA INTERACTIONS — THRESHOLD 70 ────────────────────────────
print("\n" + "=" * 60)
print("GENDER x PCA INTERACTIONS — THRESHOLD 70  (SEs clustered by circuit)")
print("=" * 60)
d = d70
for label, mask in [
    ("Female x Low PCA",  (d["gender"] == "F") & (d["pca_index"] <= pca_median)),
    ("Female x High PCA", (d["gender"] == "F") & (d["pca_index"] >  pca_median)),
    ("Male x Low PCA",    (d["gender"] == "M") & (d["pca_index"] <= pca_median)),
    ("Male x High PCA",   (d["gender"] == "M") & (d["pca_index"] >  pca_median)),
]:
    if (70, label) in done_pairs:
        _log(f"resume guard: T70 {label} already in partial results, skipping")
        continue
    dd = d[mask]
    rv = "days_from_70"
    if len(dd) < 500:
        print(f"  {label}: too few obs ({len(dd):,}), skipping")
        continue
    r = rdrobust(y=dd["voted"].values, x=dd[rv].values, c=0, masspoints="rd",
                 cluster=dd['cluster_id'].values)
    res = extract_rdd(r, threshold=70, subgroup=label)
    results.append(res)
    done_pairs.add((70, label))
    _log(f"estimate done: T70 {label}")
    print(f"  {label}: coef={res['coef']*100:+.2f}pp  se={res['se']*100:.2f}  p={res['p_value']:.4f}")

# Block 6/6 (gender x PCA interactions, T18+T70) complete — checkpoint.
_flush_partial(results, PARTIAL_RESULTS_PATH)
_log(f"checkpoint: block 6/6 (gender x PCA) flushed to rdd_results.partial.csv ({len(results)} total)")

# ── Store results table ──────────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results.to_csv(f"{REP_TABLES}/rdd_results.csv", index=False)
print("\n✓ Saved: rdd_results.csv")
print(df_results.to_string(index=False))

# All 30 estimates now live in rdd_results.csv — the partial checkpoint is
# no longer needed.
if os.path.exists(PARTIAL_RESULTS_PATH):
    os.remove(PARTIAL_RESULTS_PATH)
    _log("removed rdd_results.partial.csv (run completed successfully)")

# ── Results table (LaTeX-ready) ─────────────────────────────────────────
df_r = pd.read_csv(f"{REP_TABLES}/rdd_results.csv")

df_r["Coef. (pp)"]    = df_r["coef"].mul(100).apply(lambda x: f"{x:.2f}")
df_r["SE (pp)"]       = df_r["se"].mul(100).apply(lambda x: f"{x:.2f}")
ci_lo = df_r["robust_ci_low"].mul(100).apply(lambda x: f"{x:.2f}")
ci_hi = df_r["robust_ci_high"].mul(100).apply(lambda x: f"{x:.2f}")
df_r["Robust 95% CI"] = "[" + ci_lo + ", " + ci_hi + "]"
df_r["BW (days)"]     = df_r["bw"].apply(lambda x: f"{x:.1f}")
df_r["N (eff.)"]      = (df_r["n_eff_left"] + df_r["n_eff_right"]).apply(lambda x: f"{x:,}")
df_r["Threshold"]     = df_r["threshold"].map({18: "T18 (age 18)", 70: "T70 (age 70)"})

cols = ["Threshold", "subgroup", "Coef. (pp)", "SE (pp)", "Robust 95% CI", "BW (days)", "N (eff.)"]
tbl = df_r[cols].rename(columns={"subgroup": "Subgroup"})
print(tbl.to_string(index=False))

# LaTeX export
latex_str = tbl.to_latex(
    index=False, escape=True,
    caption="RDD Estimates: Effect of Compulsory Voting on Turnout (percentage points)",
    label="tab:rdd_results",
    column_format="llrrrrr"
)
# Keep the manual float/overflow fixes ([ht], centering, resizebox) across
# regenerations — they were once lost by re-running this cell.
latex_str = latex_str.replace("\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n", 1)
latex_str = latex_str.replace("\\begin{tabular}{llrrrrr}",
                              "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{llrrrrr}", 1)
latex_str = latex_str.replace("\\end{tabular}\n", "\\end{tabular}%\n}\n", 1)
# Table note: states the SE scheme and pre-empts the two readings the table
# invites on its own — that the CIs are conventional, and that the pooled row
# should sit between the subgroup rows. Sits outside the resizebox so it is not
# scaled with the tabular.
_note = (
    "\\par\\vspace{0.6em}\n"
    "{\\footnotesize\\begin{minipage}{\\linewidth}\\emph{Note:} Standard errors "
    "clustered by electoral circuit (CRV1); inference uses the robust, "
    "bias-corrected 95\\% confidence intervals of \\citet{calonico2014}. Each row "
    "is estimated at its own MSE-optimal bandwidth (column BW), so the pooled "
    "(Total) estimate need not fall between the sex- or stratum-specific "
    "ones.\\end{minipage}}\n"
)
latex_str = latex_str.replace("\\end{table}", _note + "\\end{table}", 1)
with open(f"{REP_TABLES}/rdd_results_latex.tex", "w") as fout:
    fout.write(latex_str)
print("Saved:", f"{REP_TABLES}/rdd_results_latex.tex")

# ── Density (manipulation) test ─────────────────────────────────────────
# Same invocation as run_density_test.py (repo root); reuses the already-
# windowed d18/d70 (idempotent re-filter to the same +-3y window as that
# script's own read).
print("\n" + "=" * 50)
print("DENSITY TEST (Cattaneo-Jansson-Ma)")
print("=" * 50)
_log("starting density test")
dens = pd.DataFrame([density_test(d18, 'days_from_18', bandwidth_max=BW_MAX),
                     density_test(d70, 'days_from_70', bandwidth_max=BW_MAX)])
dens['note'] = 'CJM rddensity, jackknife VCE'
dens.to_csv(f"{REP_TABLES}/density_test_results.csv", index=False)
print(dens[['threshold_var', 't_jk', 'p_jk', 'n']].round(4).to_string(index=False))
print("\nSaved: density_test_results.csv")

print("\n02_main_rdd: DONE")
