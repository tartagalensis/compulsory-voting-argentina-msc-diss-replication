"""
analysis.py
-----------
Reusable plotting and aggregation functions for the exploratory analysis
of the Argentine 2025 electoral dataset.

All plot functions save figures to FIGURES_DIR and display them inline.
Aggregation functions return DataFrames for further use in the notebook.

Usage:
    from src.analysis import (
        plot_nbi_distribution, plot_pca_distribution,
        province_composition, circuit_stats,
        plot_age_distribution, plot_gender_composition,
        plot_turnout_by_age, plot_turnout_by_age_gender,
        plot_rdd_visual, plot_turnout_by_nbi_quantile,
        plot_turnout_by_pca_quantile, plot_nbi_vs_pca,
        plot_turnout_vs_nbi_circuits, plot_turnout_vs_nbi_gender,
        plot_turnout_vs_pca_circuits, plot_turnout_vs_pca_gender,
        plot_mccrary_visual, plot_gender_rdd_gap_age18,
        rdd_gender_by_ses_quantile,
        run_ses_decile_rdd, run_ses_continuous_interaction,
        ses_gradient_test,
        plot_decile_rdd_per_sex, plot_decile_rdd_gap,
        plot_marginal_cutoff_continuous,
        plot_provincial_turnout, plot_threshold_density,
        gender_gap_means_by_decile, pooled_rd_by_decile,
        plot_gender_means_and_rd, plot_gender_gap_rd_and_decile,
        plot_gap_jump_robustness,
        unpack_triple_interaction, plot_triple_cells_grouped,
        plot_gender_gap_baseline_endpoint,
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

COLORS = {'F': '#e74c3c', 'M': '#2980b9'}

plt.rcParams.update({
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         11,
})

# ── AGGREGATIONS ──────────────────────────────────────────────────────────────

def province_composition(df, province_names, bw_rdd=365*3):
    """Return composition tables for the full padron and both RDD windows.

    Returns
    -------
    df_full : DataFrame — full padron composition by province
    df_18   : DataFrame — RDD window around threshold 18
    df_70   : DataFrame — RDD window around threshold 70
    """
    def _agg(d, total):
        """Aggregate voter composition by province for a given padron slice."""
        out = (
            d.groupby('province_id')
            .agg(
                n_voters   =('voted',      'count'),
                n_circuits =('circuit_id', 'nunique'),
                turnout    =('voted',      'mean'),
                pct_female =('gender',     lambda x: (x == 'F').mean()),
                avg_age    =('age',        'mean'),
                avg_nbi    =('nbi',        'mean'),
            )
            .reset_index()
        )
        out['province']     = out['province_id'].map(province_names)
        out['pct_of_total'] = out['n_voters'] / total * 100
        return out.sort_values('n_voters', ascending=False)

    df_full = _agg(df, len(df))
    df_18   = _agg(df[df['days_from_18'].between(-bw_rdd, bw_rdd)],
                   df['days_from_18'].between(-bw_rdd, bw_rdd).sum())
    df_70   = _agg(df[df['days_from_70'].between(-bw_rdd, bw_rdd)],
                   df['days_from_70'].between(-bw_rdd, bw_rdd).sum())

    return df_full, df_18, df_70


def circuit_stats(df):
    """Return a circuit-level DataFrame with turnout, NBI, PCA index and voter count."""
    return (
        df.groupby(['province_id', 'circuit_id'])
        .agg(
            turnout   =('voted',      'mean'),
            nbi       =('nbi',        'first'),
            pca_index =('pca_index',  'first'),
            n_voters  =('voted',      'count'),
        )
        .reset_index()
    )

# ── PLOT FUNCTIONS ────────────────────────────────────────────────────────────

def plot_nbi_distribution(df, figures_dir):
    """Histogram of NBI across electoral circuits with LOWESS overlay."""
    os.makedirs(figures_dir, exist_ok=True)
    df_c = df.drop_duplicates('circuit_id')[['circuit_id', 'nbi']]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_c['nbi'] * 100, bins=80, color='steelblue', alpha=0.8, edgecolor='none')

    counts, edges = np.histogram(df_c['nbi'] * 100, bins=80, density=True)
    sm = lowess(counts, edges[:-1], frac=0.15)
    ax2 = ax.twinx()
    ax2.plot(sm[:, 0], sm[:, 1], color='red', linewidth=2, label='LOWESS')
    ax2.set_ylabel("Density")

    ax.set_xlabel("NBI (%)")
    ax.set_ylabel("Number of circuits")
    ax.set_title("Distribution of NBI Across Electoral Circuits — Argentina 2025")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/nbi_distribution.png", bbox_inches='tight')
    plt.show()


def plot_age_distribution(df, figures_dir):
    """Histogram of voter ages with RDD threshold lines."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df['age'], bins=100, color='steelblue', alpha=0.8, edgecolor='none')
    ax.axvline(18, color='red',    linestyle='--', linewidth=1.2, label='Threshold: 18')
    ax.axvline(70, color='orange', linestyle='--', linewidth=1.2, label='Threshold: 70')
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of voters")
    ax.set_title("Age Distribution of the Electoral Roll — Argentina 2025")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/age_distribution.png", bbox_inches='tight')
    plt.show()
    print(df['age'].describe().to_string())


def plot_gender_composition(df, figures_dir):
    """Bar chart of gender counts and age distribution by gender."""
    os.makedirs(figures_dir, exist_ok=True)
    counts = df['gender'].value_counts()
    pct    = df['gender'].value_counts(normalize=True) * 100

    print("\nGender composition:")
    for g in counts.index:
        print(f"  {g}: {counts[g]:,} ({pct[g]:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(counts.index, counts.values,
                color=['#e74c3c', '#2980b9', 'grey'], alpha=0.8)
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Number of voters")
    axes[0].set_title("Gender Composition — Full Padron")

    for g, c in COLORS.items():
        axes[1].hist(df[df['gender'] == g]['age'], bins=80,
                     alpha=0.5, color=c, label=g, edgecolor='none')
    axes[1].axvline(18, color='black', linestyle='--', linewidth=1)
    axes[1].axvline(70, color='black', linestyle=':',  linewidth=1)
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Number of voters")
    axes[1].set_title("Age Distribution by Gender")
    axes[1].legend()

    plt.suptitle("Gender Composition — Argentina 2025", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/gender_composition.png", bbox_inches='tight')
    plt.show()


def plot_turnout_by_age(df, figures_dir, age_min=14, age_max=75):
    """Line plot of average turnout by age in years."""
    os.makedirs(figures_dir, exist_ok=True)
    d = df[df['age'].between(age_min, age_max)]

    # Bin by 30-day intervals for precision
    d = d.copy()
    d['age_bin'] = (d['age_days'] // 30) * 30
    turnout = d.groupby('age_bin')['voted'].mean().reset_index()
    turnout['age_years'] = turnout['age_bin'] / 365.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(turnout['age_years'], turnout['voted'] * 100, color='steelblue', linewidth=1.5)
    ax.axvline(18, color='red',    linestyle='--', linewidth=1.2, label='Threshold: 18 (compulsory starts)')
    ax.axvline(70, color='orange', linestyle='--', linewidth=1.2, label='Threshold: 70 (compulsory ends)')
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Voter Turnout by Age — Argentina 2025")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_by_age.png", bbox_inches='tight')
    plt.show()


def plot_turnout_by_age_gender(df, figures_dir, age_min=14, age_max=75):
    """Line plot of average turnout by age, split by gender."""
    os.makedirs(figures_dir, exist_ok=True)
    d = df[df['age'].between(age_min, age_max)].copy()
    d['age_bin'] = (d['age_days'] // 30) * 30
    turnout = d.groupby(['age_bin', 'gender'])['voted'].mean().reset_index()
    turnout['age_years'] = turnout['age_bin'] / 365.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for g, c in COLORS.items():
        t = turnout[turnout['gender'] == g]
        ax.plot(t['age_years'], t['voted'] * 100, label=g, color=c, linewidth=1.5)
    ax.axvline(18, color='black', linestyle='--', linewidth=1, label='Threshold: 18')
    ax.axvline(70, color='black', linestyle=':',  linewidth=1, label='Threshold: 70')
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Voter Turnout by Age and Gender — Argentina 2025")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_by_age_gender.png", bbox_inches='tight')
    plt.show()


def plot_rdd_visual(df, running_var, threshold_label, color, figures_dir,
                    bw_days=365*3, bin_size=30, filename=None):
    """Binned scatter plot of turnout around an RDD threshold."""
    os.makedirs(figures_dir, exist_ok=True)
    d = df[df[running_var].between(-bw_days, bw_days)].copy()
    d['bin'] = (d[running_var] // bin_size) * bin_size
    bins = d.groupby('bin')['voted'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(bins['bin'], bins['voted'] * 100, s=15, color='steelblue', alpha=0.7)
    ax.axvline(0, color=color, linestyle='--', linewidth=1.5, label=threshold_label)
    ax.set_xlabel(f"Days from threshold")
    ax.set_ylabel("Turnout (%)")
    ax.set_title(f"RDD Visual — {threshold_label}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    fname = filename or f"rdd_visual_{running_var}.png"
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight')
    plt.show()


def plot_turnout_by_nbi_quantile(df, figures_dir, q=50):
    """Line plot of average turnout by NBI quantile (default q=50)."""
    os.makedirs(figures_dir, exist_ok=True)
    d = df.copy()
    d['nbi_quantile'] = pd.qcut(d['nbi'], q=q, labels=False) + 1
    turnout = d.groupby('nbi_quantile')['voted'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(turnout['nbi_quantile'], turnout['voted'] * 100,
            color='steelblue', linewidth=1.5)
    ax.set_xlabel("NBI quantile (low → high)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Voter Turnout by NBI Quantile — Argentina 2025")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks([1, q // 4, q // 2, 3 * q // 4, q])
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_by_nbi_decile.png", bbox_inches='tight')
    plt.show()


def plot_turnout_vs_nbi_circuits(df_circuit, figures_dir):
    """Scatter of turnout vs NBI by circuit (log scale, LOWESS)."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')
    ax.scatter(df_circuit['nbi'] * 100, df_circuit['turnout'] * 100,
               s=df_circuit['n_voters'] / 100, alpha=0.4,
               color='steelblue', edgecolors='none')
    sm = lowess(df_circuit['turnout'] * 100, df_circuit['nbi'] * 100, frac=0.3)
    ax.plot(sm[:, 0], sm[:, 1], color='red', linewidth=2, label='LOWESS')
    ax.set_xlabel("NBI — % households with unmet basic needs (log scale)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Voter Turnout vs NBI by Electoral Circuit — Argentina 2025\n"
                 "(dot size = number of registered voters)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_vs_nbi_circuits.png", bbox_inches='tight')
    plt.show()
    print(f"Circuits: {len(df_circuit):,} — Correlation: {df_circuit['nbi'].corr(df_circuit['turnout']):.3f}")


def plot_turnout_vs_nbi_gender(df_circuit, figures_dir):
    """LOWESS overlay by gender and facet scatter of turnout vs NBI."""
    os.makedirs(figures_dir, exist_ok=True)

    # Plot A: LOWESS by gender
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')
    for g, c in COLORS.items():
        d = df_circuit[df_circuit['gender'] == g]
        sm = lowess(d['turnout'] * 100, d['nbi'] * 100, frac=0.3)
        ax.scatter(d['nbi'] * 100, d['turnout'] * 100,
                   s=d['n_voters'] / 200, alpha=0.2, color=c, edgecolors='none')
        ax.plot(sm[:, 0], sm[:, 1], color=c, linewidth=2.5,
                label=f"{'Female' if g == 'F' else 'Male'} (LOWESS)")
    ax.set_xlabel("NBI (log scale)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Turnout vs NBI by Circuit and Gender — Argentina 2025")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_vs_nbi_gender_lowess.png", bbox_inches='tight')
    plt.show()

    # Plot B: Facet
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, g in zip(axes, ['F', 'M']):
        d = df_circuit[df_circuit['gender'] == g]
        sm = lowess(d['turnout'] * 100, d['nbi'] * 100, frac=0.3)
        ax.set_xscale('log')
        ax.scatter(d['nbi'] * 100, d['turnout'] * 100,
                   s=d['n_voters'] / 200, alpha=0.3,
                   color=COLORS[g], edgecolors='none')
        ax.plot(sm[:, 0], sm[:, 1], color='black', linewidth=2, label='LOWESS')
        ax.set_xlabel("NBI (log scale)")
        ax.set_ylabel("Turnout (%)" if g == 'F' else "")
        ax.set_title("Female" if g == 'F' else "Male")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.legend()
    fig.suptitle("Voter Turnout vs NBI by Circuit — Argentina 2025", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_vs_nbi_facet_gender.png", bbox_inches='tight')
    plt.show()


def plot_pca_distribution(df, figures_dir):
    """Histogram of PCA deprivation index across electoral circuits with LOWESS overlay."""
    os.makedirs(figures_dir, exist_ok=True)
    df_c = df.drop_duplicates('circuit_id')[['circuit_id', 'pca_index']]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_c['pca_index'], bins=80, color='steelblue', alpha=0.8, edgecolor='none')

    counts, edges = np.histogram(df_c['pca_index'], bins=80, density=True)
    sm = lowess(counts, edges[:-1], frac=0.15)
    ax2 = ax.twinx()
    ax2.plot(sm[:, 0], sm[:, 1], color='red', linewidth=2, label='LOWESS')
    ax2.set_ylabel("Density")

    ax.set_xlabel("PCA deprivation index (0 = least deprived nationally)")
    ax.set_ylabel("Number of circuits")
    ax.set_title("Distribution of PCA Deprivation Index Across Electoral Circuits — Argentina 2025")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/pca_distribution.png", bbox_inches='tight')
    plt.show()
    print(df_c['pca_index'].describe().round(3).to_string())


def plot_turnout_by_pca_quantile(df, figures_dir, q=50):
    """Line plot of average turnout by PCA deprivation quantile."""
    os.makedirs(figures_dir, exist_ok=True)
    d = df.copy()
    d['pca_quantile'] = pd.qcut(d['pca_index'], q=q, labels=False) + 1
    turnout = d.groupby('pca_quantile')['voted'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(turnout['pca_quantile'], turnout['voted'] * 100,
            color='steelblue', linewidth=1.5)
    ax.set_xlabel("PCA deprivation quantile (low → high)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Voter Turnout by PCA Deprivation Quantile — Argentina 2025")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks([1, q // 4, q // 2, 3 * q // 4, q])
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_by_pca_quantile.png", bbox_inches='tight')
    plt.show()


def plot_nbi_vs_pca(df_circuit, figures_dir):
    """Scatter of NBI vs PCA deprivation index at circuit level with LOWESS and correlation."""
    os.makedirs(figures_dir, exist_ok=True)
    r = df_circuit['nbi'].corr(df_circuit['pca_index'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_circuit['nbi'] * 100, df_circuit['pca_index'],
               s=df_circuit['n_voters'] / 150, alpha=0.35,
               color='steelblue', edgecolors='none')
    sm = lowess(df_circuit['pca_index'], df_circuit['nbi'] * 100, frac=0.3)
    ax.plot(sm[:, 0], sm[:, 1], color='red', linewidth=2, label=f'LOWESS  (r = {r:.2f})')
    ax.set_xlabel("NBI — % households with unmet basic needs")
    ax.set_ylabel("PCA deprivation index")
    ax.set_title("NBI vs PCA Deprivation Index by Electoral Circuit — Argentina 2025\n"
                 "(dot size = number of registered voters)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/nbi_vs_pca.png", bbox_inches='tight')
    plt.show()
    print(f"Circuits: {len(df_circuit):,} — Pearson r: {r:.3f}")


def plot_turnout_vs_pca_circuits(df_circuit, figures_dir):
    """Scatter of turnout vs PCA deprivation index by circuit with LOWESS."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_circuit['pca_index'], df_circuit['turnout'] * 100,
               s=df_circuit['n_voters'] / 100, alpha=0.4,
               color='steelblue', edgecolors='none')
    sm = lowess(df_circuit['turnout'] * 100, df_circuit['pca_index'], frac=0.3)
    ax.plot(sm[:, 0], sm[:, 1], color='red', linewidth=2, label='LOWESS')
    ax.set_xlabel("PCA deprivation index (0 = least deprived nationally)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Voter Turnout vs PCA Deprivation Index by Electoral Circuit — Argentina 2025\n"
                 "(dot size = number of registered voters)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_vs_pca_circuits.png", bbox_inches='tight')
    plt.show()
    print(f"Circuits: {len(df_circuit):,} — Correlation: {df_circuit['pca_index'].corr(df_circuit['turnout']):.3f}")


def plot_turnout_vs_pca_gender(df_circuit, figures_dir):
    """LOWESS overlay by gender and facet scatter of turnout vs PCA deprivation index."""
    os.makedirs(figures_dir, exist_ok=True)

    # Plot A: LOWESS by gender
    fig, ax = plt.subplots(figsize=(10, 6))
    for g, c in COLORS.items():
        d = df_circuit[df_circuit['gender'] == g]
        sm = lowess(d['turnout'] * 100, d['pca_index'], frac=0.3)
        ax.scatter(d['pca_index'], d['turnout'] * 100,
                   s=d['n_voters'] / 200, alpha=0.2, color=c, edgecolors='none')
        ax.plot(sm[:, 0], sm[:, 1], color=c, linewidth=2.5,
                label=f"{'Female' if g == 'F' else 'Male'} (LOWESS)")
    ax.set_xlabel("PCA deprivation index (0 = least deprived nationally)")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Turnout vs PCA Deprivation Index by Circuit and Gender — Argentina 2025")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_vs_pca_gender_lowess.png", bbox_inches='tight')
    plt.show()

    # Plot B: Facet
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, g in zip(axes, ['F', 'M']):
        d = df_circuit[df_circuit['gender'] == g]
        sm = lowess(d['turnout'] * 100, d['pca_index'], frac=0.3)
        ax.scatter(d['pca_index'], d['turnout'] * 100,
                   s=d['n_voters'] / 200, alpha=0.3,
                   color=COLORS[g], edgecolors='none')
        ax.plot(sm[:, 0], sm[:, 1], color='black', linewidth=2, label='LOWESS')
        ax.set_xlabel("PCA deprivation index")
        ax.set_ylabel("Turnout (%)" if g == 'F' else "")
        ax.set_title("Female" if g == 'F' else "Male")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.legend()
    fig.suptitle("Voter Turnout vs PCA Deprivation Index by Circuit — Argentina 2025", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/turnout_vs_pca_facet_gender.png", bbox_inches='tight')
    plt.show()


def plot_mccrary_visual(df, figures_dir, bw_days=365*3, bins=100):
    """Visual McCrary density test — histogram around each threshold."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, label, color in [
        (axes[0], 'days_from_18', 'Threshold: age 18', 'red'),
        (axes[1], 'days_from_70', 'Threshold: age 70', 'orange'),
    ]:
        data = df[df[col].between(-bw_days, bw_days)][col]
        ax.hist(data, bins=bins, color='steelblue', alpha=0.7, edgecolor='none')
        ax.axvline(0, color=color, linestyle='--', linewidth=1.5, label=label)
        ax.set_xlabel("Days from threshold")
        ax.set_ylabel("Number of voters")
        ax.set_title(f"Density — {label}")
        ax.legend()
    plt.suptitle("McCrary Density Test (Visual) — Argentina 2025", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/mccrary_density_visual.png", bbox_inches='tight')
    plt.show()


def plot_gender_rdd_gap_age18(df, figures_dir, bw_days=365, poly_degree=1,
                              alpha_level=0.10, filename=None):
    """
    Two-panel RDD figure at the age-18 (compulsory voting) cutoff by sex.

    Left panel: mean turnout per integer value of `days_from_18` for each sex,
    with local polynomial fits of `poly_degree` on each side of the cutoff
    (Cattaneo-Idrobo-Titiunik style). Vertical dashed line at 0.

    Right panel: the female-minus-male turnout gap from a *single* OLS of
    `voted ~ days_from_18 * right * female` (polynomial degree `poly_degree`)
    with cluster-robust SEs by `circuit_id`. The gap is read off the
    female-related coefficients and plotted across the window with its
    pointwise 95% CI. Horizontal dashed line at gap=0 (parity); light pink
    shading marks the region favoring women (gap > 0).

    Per-sex jumps are also estimated with `rdrobust` (uniform kernel,
    MSE-optimal bandwidth, clustered by `circuit_id`). Prints τ for women, τ
    for men, their difference and a normal-approx p-value, and the local
    linear estimate of turnout just to the right of the cutoff for each sex
    with its `(1 - alpha_level)*100`% CI.

    Returns a dict with the rdrobust objects, the difference test, the level
    estimates, and the fitted interaction OLS.
    """
    import statsmodels.api as sm
    from scipy.stats import norm
    from rdrobust import rdrobust

    os.makedirs(figures_dir, exist_ok=True)

    sub = df[df['gender'].isin(['F', 'M'])].copy()
    sub = sub[sub['days_from_18'].between(-bw_days, bw_days)].copy()
    sub['female'] = (sub['gender'] == 'F').astype(int)
    sub['right']  = (sub['days_from_18'] >= 0).astype(int)

    binned = (sub.groupby(['gender', 'days_from_18'])['voted']
                 .mean()
                 .reset_index())

    def _fit_poly(d_, deg):
        x = d_['days_from_18'].to_numpy(dtype=float)
        y = d_['voted'].to_numpy(dtype=float)
        X = np.column_stack([x**i for i in range(deg + 1)])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def _pred_poly(coef, xg):
        X = np.column_stack([xg**i for i in range(len(coef))])
        return X @ coef

    xL = np.linspace(-bw_days, -0.5, 200)
    xR = np.linspace(0.5, bw_days, 200)
    fits = {}
    for g in ['F', 'M']:
        d_L = sub[(sub['gender'] == g) & (sub['days_from_18'] < 0)]
        d_R = sub[(sub['gender'] == g) & (sub['days_from_18'] >= 0)]
        fits[g] = (_fit_poly(d_L, poly_degree), _fit_poly(d_R, poly_degree))

    # Single interaction model for the gap.
    # Design: 1, x..x^p, r, r*x..r*x^p, f, f*x..f*x^p, f*r, f*r*x..f*r*x^p
    def _design(x, r, f, deg):
        cols = [np.ones_like(x, dtype=float)]
        for i in range(1, deg + 1):
            cols.append(x**i)
        cols.append(r.astype(float))
        for i in range(1, deg + 1):
            cols.append(r * x**i)
        cols.append(f.astype(float))
        for i in range(1, deg + 1):
            cols.append(f * x**i)
        cols.append((f * r).astype(float))
        for i in range(1, deg + 1):
            cols.append(f * r * x**i)
        return np.column_stack(cols)

    x_arr = sub['days_from_18'].to_numpy(dtype=float)
    r_arr = sub['right'].to_numpy()
    f_arr = sub['female'].to_numpy()
    y_arr = sub['voted'].to_numpy(dtype=float)
    X_mat = _design(x_arr, r_arr, f_arr, poly_degree)

    ols = sm.OLS(y_arr, X_mat).fit(
        cov_type='cluster',
        cov_kwds={'groups': sub['circuit_id'].to_numpy()},
    )

    xg = np.linspace(-bw_days, bw_days, 400)
    rg = (xg >= 0).astype(int)
    X1 = _design(xg, rg, np.ones_like(xg, dtype=int),  poly_degree)
    X0 = _design(xg, rg, np.zeros_like(xg, dtype=int), poly_degree)
    C  = X1 - X0  # contrast: F − M predictions
    beta = np.asarray(ols.params)
    Vb   = np.asarray(ols.cov_params())
    gap  = C @ beta
    gap_se = np.sqrt(np.einsum('ij,jk,ik->i', C, Vb, C))
    z95   = norm.ppf(0.975)
    gap_lo, gap_hi = gap - z95 * gap_se, gap + z95 * gap_se

    # rdrobust per sex
    rd = {}
    for g in ['F', 'M']:
        dg = sub[sub['gender'] == g]
        rd[g] = rdrobust(
            y=dg['voted'].to_numpy(dtype=float),
            x=dg['days_from_18'].to_numpy(dtype=float),
            kernel='uniform',
            bwselect='mserd',
            cluster=dg['circuit_id'].to_numpy(),
        )

    tau_F = float(rd['F'].coef.iloc[0, 0])
    tau_M = float(rd['M'].coef.iloc[0, 0])
    se_F  = float(rd['F'].se.iloc[0, 0])
    se_M  = float(rd['M'].se.iloc[0, 0])
    h_F   = float(rd['F'].bws.iloc[0, 0])
    h_M   = float(rd['M'].bws.iloc[0, 0])
    diff  = tau_F - tau_M
    se_d  = float(np.sqrt(se_F**2 + se_M**2))
    z_d   = diff / se_d
    p_d   = 2 * (1 - norm.cdf(abs(z_d)))

    print("RD effects at age-18 cutoff "
          "(rdrobust, uniform kernel, MSE-optimal BW, clustered by circuit):")
    print(f"  Women: tau = {tau_F:.4f} ({tau_F*100:.2f} pp)  "
          f"SE = {se_F:.4f}  h = {h_F:.1f} d")
    print(f"  Men:   tau = {tau_M:.4f} ({tau_M*100:.2f} pp)  "
          f"SE = {se_M:.4f}  h = {h_M:.1f} d")
    print(f"  Difference (F − M): {diff:.4f} ({diff*100:.2f} pp)  "
          f"SE = {se_d:.4f}  z = {z_d:.3f}  p = {p_d:.4f}")

    # Turnout level just to the right of cutoff (local linear, clustered SE)
    pct_ci = int(round((1 - alpha_level) * 100))
    z_lvl  = norm.ppf(1 - alpha_level / 2)
    print(f"\nTurnout level just to the right of cutoff "
          f"({pct_ci}% CI, local linear within rdrobust h_r, clustered SE):")
    levels = {}
    for g in ['F', 'M']:
        h_r = float(rd[g].bws.iloc[0, 1])
        d_r = sub[(sub['gender'] == g)
                  & (sub['days_from_18'] >= 0)
                  & (sub['days_from_18'] <= h_r)]
        Xr = sm.add_constant(d_r['days_from_18'].to_numpy(dtype=float))
        m  = sm.OLS(d_r['voted'].to_numpy(dtype=float), Xr).fit(
            cov_type='cluster',
            cov_kwds={'groups': d_r['circuit_id'].to_numpy()},
        )
        b0, sb0 = float(m.params[0]), float(m.bse[0])
        lo, hi  = b0 - z_lvl * sb0, b0 + z_lvl * sb0
        levels[g] = {'level': b0, 'ci_low': lo, 'ci_high': hi, 'h_r': h_r}
        print(f"  {g}: level = {b0*100:.2f}%  "
              f"CI = [{lo*100:.2f}%, {hi*100:.2f}%]  h_r = {h_r:.1f} d")

    colors = {'F': '#c44e52', 'M': '#4c72b0'}
    labels = {'F': 'Women', 'M': 'Men'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for g in ['F', 'M']:
        b = binned[binned['gender'] == g]
        ax.scatter(b['days_from_18'], b['voted'] * 100,
                   s=8, alpha=0.35, color=colors[g],
                   label=f"{labels[g]} (binned mean)")
        cL, cR = fits[g]
        ax.plot(xL, _pred_poly(cL, xL) * 100,
                color=colors[g], linewidth=2)
        ax.plot(xR, _pred_poly(cR, xR) * 100,
                color=colors[g], linewidth=2,
                label=f"{labels[g]} (poly deg {poly_degree})")
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
    ax.set_xlabel("Days from 18th birthday on election day")
    ax.set_ylabel("Turnout (%)")
    ax.set_title("Turnout by sex around the age-18 compulsory-voting cutoff")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    ax = axes[1]
    ymax = max(float(np.nanmax(gap_hi * 100)), 0.5)
    ymin = min(float(np.nanmin(gap_lo * 100)), -0.5)
    ax.axhspan(0, ymax * 1.10, color='#c44e52', alpha=0.06,
               label='Region favoring women')
    ax.fill_between(xg, gap_lo * 100, gap_hi * 100,
                    color='gray', alpha=0.25, label='95% CI')
    ax.plot(xg, gap * 100, color='black', linewidth=2,
            label='Predicted F − M gap')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linestyle=':', linewidth=1)
    ax.set_ylim(ymin * 1.10, ymax * 1.15)
    ax.set_xlabel("Days from 18th birthday on election day")
    ax.set_ylabel("Turnout gap, Women − Men (pp)")
    ax.set_title("Estimated sex × side-of-cutoff gap (single OLS, clustered SE)")
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fname = filename or 'rdd_gender_gap_age18.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()

    return {
        'rd_female': rd['F'],
        'rd_male':   rd['M'],
        'tau_female': tau_F, 'tau_male': tau_M,
        'diff': diff, 'diff_se': se_d, 'diff_p': p_d,
        'levels_right': levels,
        'ols': ols,
    }


def _gender_rd_subgroup(sub, poly_degree):
    """
    Run the gender RD machinery on a pre-filtered subset (F/M only, already
    inside the bandwidth window). Returns per-sex rdrobust jumps (uniform
    kernel, MSE-optimal BW, clustered by circuit) and the pre- and post-cutoff
    F-M turnout gaps from a single OLS with cluster-robust SEs.
    """
    import warnings
    import statsmodels.api as sm
    from scipy.stats import norm
    from rdrobust import rdrobust

    out = {'n': int(len(sub))}

    for g in ['F', 'M']:
        dg = sub[sub['gender'] == g]
        if len(dg) < 200 or dg['voted'].nunique() < 2:
            out[f'tau_{g}'] = np.nan
            out[f'se_{g}']  = np.nan
            out[f'h_{g}']   = np.nan
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = rdrobust(
                    y=dg['voted'].to_numpy(dtype=float),
                    x=dg['days_from_18'].to_numpy(dtype=float),
                    kernel='uniform',
                    bwselect='mserd',
                    cluster=dg['circuit_id'].to_numpy(),
                )
            out[f'tau_{g}'] = float(r.coef.iloc[0, 0])
            out[f'se_{g}']  = float(r.se.iloc[0, 0])
            out[f'h_{g}']   = float(r.bws.iloc[0, 0])
        except Exception:
            out[f'tau_{g}'] = np.nan
            out[f'se_{g}']  = np.nan
            out[f'h_{g}']   = np.nan

    tau_F, tau_M = out['tau_F'], out['tau_M']
    se_F, se_M   = out['se_F'], out['se_M']
    if np.isfinite(tau_F) and np.isfinite(tau_M):
        diff = tau_F - tau_M
        se_d = float(np.sqrt(se_F**2 + se_M**2))
        z    = diff / se_d if se_d > 0 else np.nan
        p    = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    else:
        diff, se_d, p = np.nan, np.nan, np.nan
    out['diff_FM'] = diff
    out['se_diff'] = se_d
    out['p_diff']  = p

    # Single interaction OLS for pre/post gaps: voted ~ x * right * female
    s = sub.copy()
    s['female'] = (s['gender'] == 'F').astype(int)
    s['right']  = (s['days_from_18'] >= 0).astype(int)
    x = s['days_from_18'].to_numpy(dtype=float)
    r = s['right'].to_numpy()
    f = s['female'].to_numpy()

    cols = [np.ones_like(x, dtype=float)]
    for i in range(1, poly_degree + 1): cols.append(x**i)
    cols.append(r.astype(float))
    for i in range(1, poly_degree + 1): cols.append(r * x**i)
    cols.append(f.astype(float))
    for i in range(1, poly_degree + 1): cols.append(f * x**i)
    cols.append((f * r).astype(float))
    for i in range(1, poly_degree + 1): cols.append(f * r * x**i)
    X_mat = np.column_stack(cols)

    try:
        ols = sm.OLS(s['voted'].to_numpy(dtype=float), X_mat).fit(
            cov_type='cluster',
            cov_kwds={'groups': s['circuit_id'].to_numpy()},
        )
        beta = np.asarray(ols.params)
        Vb   = np.asarray(ols.cov_params())
        zer  = np.zeros(poly_degree)
        # Layout: [1, x..x^p, r, r*x..r*x^p, f, f*x..f*x^p, fr, fr*x..fr*x^p]
        # gap_post = F=1 minus F=0 at x=0, r=1 => picks up coef on f and on fr
        c_post = np.concatenate([[0.0], zer, [0.0], zer, [1.0], zer, [1.0], zer])
        # gap_pre  = same at x=0, r=0 => picks up coef on f only
        c_pre  = np.concatenate([[0.0], zer, [0.0], zer, [1.0], zer, [0.0], zer])
        out['gap_pre']     = float(c_pre  @ beta)
        out['se_gap_pre']  = float(np.sqrt(c_pre  @ Vb @ c_pre))
        out['gap_post']    = float(c_post @ beta)
        out['se_gap_post'] = float(np.sqrt(c_post @ Vb @ c_post))
    except Exception:
        out['gap_pre'] = out['se_gap_pre'] = np.nan
        out['gap_post'] = out['se_gap_post'] = np.nan

    return out


def rdd_gender_by_ses_quantile(df, figures_dir, tables_dir,
                               ses_var='pca_index',
                               n_decile=10, n_plot_q=3,
                               bw_days=365, poly_degree=1,
                               filename=None, table_filename=None):
    """
    Repeat the age-18 gender RD inside SES quantiles defined at the
    circuit level (one quantile per circuit, then attached to its voters).

    For each of `n_decile` deciles, computes per-sex rdrobust jumps
    (uniform kernel, MSE-optimal BW, clustered by circuit) and the
    F-M sex difference with normal-approx p-value, plus the pre- and
    post-cutoff F-M turnout gaps from a single interacted OLS. The
    decile-level results are printed and saved to `tables_dir`.

    For each of `n_plot_q` plot-quantiles (e.g. 3=terciles, 5=quintiles),
    the same machinery is run and the *post-cutoff* F-M turnout gap
    (the user's "equalizing effect" of CV) is plotted next to the
    pre-cutoff gap with 95% CIs.

    Higher `ses_var` (default: `pca_index`) means more deprivation, so
    quantile 1 = least deprived, quantile `n_plot_q` = most deprived.
    """
    _Q_NAME = {2: 'half', 3: 'tercile', 4: 'quartile', 5: 'quintile',
               6: 'sextile', 10: 'decile'}
    q_label = _Q_NAME.get(n_plot_q, f'q{n_plot_q}')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Circuit-level SES quantiles (one assignment per circuit).
    circuit_ses = (
        df[['province_id', 'circuit_id', ses_var]]
        .drop_duplicates(['province_id', 'circuit_id'])
        .reset_index(drop=True)
    )
    circuit_ses['ses_decile'] = pd.qcut(
        circuit_ses[ses_var], q=n_decile,  labels=False, duplicates='drop') + 1
    circuit_ses['ses_plot_q'] = pd.qcut(
        circuit_ses[ses_var], q=n_plot_q, labels=False, duplicates='drop') + 1

    sub = df[df['gender'].isin(['F', 'M'])
             & df['days_from_18'].between(-bw_days, bw_days)].copy()
    sub = sub.merge(
        circuit_ses[['province_id', 'circuit_id', 'ses_decile', 'ses_plot_q']],
        on=['province_id', 'circuit_id'], how='left',
    )

    # ── Per-decile RD ─────────────────────────────────────────────────
    decile_rows = []
    print(f"Running gender RD inside each of {n_decile} {ses_var} deciles "
          f"(this can take a few minutes)...")
    for q in sorted(sub['ses_decile'].dropna().unique()):
        s = sub[sub['ses_decile'] == q]
        r = _gender_rd_subgroup(s, poly_degree)
        r['ses_decile'] = int(q)
        decile_rows.append(r)
        print(f"  Decile {int(q):2d}: n={r['n']:>7,}  "
              f"tau_F={r['tau_F']*100:6.2f}pp  tau_M={r['tau_M']*100:6.2f}pp  "
              f"F-M={r['diff_FM']*100:6.2f}pp (p={r['p_diff']:.3f})  "
              f"gap_post={r['gap_post']*100:5.2f}pp")

    df_decile = pd.DataFrame(decile_rows)[
        ['ses_decile', 'n', 'tau_F', 'se_F', 'h_F',
         'tau_M', 'se_M', 'h_M',
         'diff_FM', 'se_diff', 'p_diff',
         'gap_pre', 'se_gap_pre', 'gap_post', 'se_gap_post']
    ]

    csv_fname = table_filename or f"rdd_gender_by_{ses_var}_decile.csv"
    df_decile.to_csv(f"{tables_dir}/{csv_fname}", index=False)
    print(f"\nSaved decile table: {tables_dir}/{csv_fname}")

    # ── Per-plot-quantile RD ──────────────────────────────────────────
    print(f"\nRunning gender RD inside each of {n_plot_q} {ses_var} {q_label}s...")
    plot_rows = []
    for q in sorted(sub['ses_plot_q'].dropna().unique()):
        s = sub[sub['ses_plot_q'] == q]
        r = _gender_rd_subgroup(s, poly_degree)
        r['ses_plot_q'] = int(q)
        plot_rows.append(r)
        print(f"  {q_label.capitalize()} {int(q)}: n={r['n']:>7,}  "
              f"gap_pre={r['gap_pre']*100:5.2f}pp  "
              f"gap_post={r['gap_post']*100:5.2f}pp  "
              f"F-M jump diff={r['diff_FM']*100:5.2f}pp (p={r['p_diff']:.3f})")
    df_plotq = pd.DataFrame(plot_rows)

    # ── Plot: post-cutoff gap by quantile, with pre-cutoff for context
    z95 = 1.959963984540054  # norm.ppf(0.975)
    yerr_pre  = z95 * df_plotq['se_gap_pre'].to_numpy()  * 100
    yerr_post = z95 * df_plotq['se_gap_post'].to_numpy() * 100

    xpos = np.arange(len(df_plotq))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 2 + 1.6 * n_plot_q), 5))
    ax.bar(xpos - width/2, df_plotq['gap_pre']  * 100, width,
           yerr=yerr_pre,  color='#9aa0a6', alpha=0.85, capsize=4,
           label='Pre-cutoff F − M gap (just left of 18)')
    ax.bar(xpos + width/2, df_plotq['gap_post'] * 100, width,
           yerr=yerr_post, color='#c44e52', alpha=0.90, capsize=4,
           label='Post-cutoff F − M gap (just right of 18) — "equalizing"')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(xpos)
    qmin, qmax = df_plotq['ses_plot_q'].min(), df_plotq['ses_plot_q'].max()
    ax.set_xticklabels([
        f"{q_label.capitalize()} {int(q)}\n"
        + ('(least deprived)' if q == qmin
           else '(most deprived)' if q == qmax
           else '')
        for q in df_plotq['ses_plot_q']
    ])
    ax.set_xlabel(f"Circuit SES {q_label} (by {_ses_label(ses_var)}; higher = more deprived)")
    ax.set_ylabel("F − M turnout gap (pp), 95% CI")
    ax.set_title(
        f"Equalizing effect of compulsory voting at age 18, "
        f"by SES {q_label} ({ses_var})")
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fname = filename or f'rdd_gender_gap_by_{ses_var}_{q_label}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved figure: {figures_dir}/{fname}")

    return {'decile': df_decile, q_label: df_plotq}


# ─────────────────────────────────────────────────────────────────────────────
# 07_gradient.ipynb — SES gradient RDD helpers (match 04 estimation settings:
# triangular kernel, polynomial degree 1, MSE-optimal BW, NO clustering).
# ─────────────────────────────────────────────────────────────────────────────

def _ses_decile_rdd_one(d, threshold_var, decile, alpha_level=0.10,
                        min_obs_per_sex=200, masspoints='rd', cluster_var=None):
    """
    Inside a single SES decile subset `d` (already F/M only, pre-filtered to
    bandwidth window), run rdrobust separately by sex with default settings
    matching 04_rdd.ipynb (triangular kernel, p=1, mserd). If `cluster_var` is
    a column of `d`, SEs are cluster-robust on it (e.g. the electoral circuit);
    otherwise they are the rdrobust default (no clustering, matching 04_rdd).
    Also compute post-cutoff turnout per sex with (1-alpha_level)% CI via
    a local linear regression on the right side within rdrobust's h_r.

    Returns a dict (one row of the decile table).
    """
    import warnings
    import statsmodels.api as sm
    from scipy.stats import norm
    from rdrobust import rdrobust

    z_lvl = norm.ppf(1 - alpha_level / 2)
    out = {'decile': int(decile), 'n_voters': int(len(d))}

    for g in ['F', 'M']:
        dg = d[d['gender'] == g]
        keys_nan = [f'tau_{g}', f'se_{g}', f'p_{g}',
                    f'ci_lo_{g}', f'ci_hi_{g}', f'bw_{g}',
                    f'post_{g}', f'post_lo_{g}', f'post_hi_{g}']
        if len(dg) < min_obs_per_sex or dg['voted'].nunique() < 2:
            for k in keys_nan: out[k] = np.nan
            out[f'n_eff_l_{g}'] = 0
            out[f'n_eff_r_{g}'] = 0
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rd_kwargs = dict(
                    y=dg['voted'].values,
                    x=dg[threshold_var].values,
                    c=0, masspoints=masspoints,
                )
                if cluster_var is not None:
                    rd_kwargs['cluster'] = dg[cluster_var].values
                r = rdrobust(**rd_kwargs)
            tau_g = float(r.coef.iloc[0, 0])
            se_g = float(r.se.iloc[0, 0])
            p_g = float(r.pv.iloc[0, 0])
            ci_lo = float(r.ci.iloc[2, 0])
            ci_hi = float(r.ci.iloc[2, 1])
            h_l = float(r.bws.iloc[0, 0])
            h_r = float(r.bws.iloc[0, 1])
            n_eff_l = int(r.N_h[0])
            n_eff_r = int(r.N_h[1])
        except Exception:
            for k in keys_nan: out[k] = np.nan
            out[f'n_eff_l_{g}'] = 0
            out[f'n_eff_r_{g}'] = 0
            continue

        # Post-cutoff level via local linear on right side within h_r.
        post_lvl, post_lo, post_hi = np.nan, np.nan, np.nan
        if np.isfinite(h_r) and h_r > 0:
            d_r = dg[(dg[threshold_var] >= 0) & (dg[threshold_var] <= h_r)]
            if len(d_r) >= 50:
                Xr = sm.add_constant(d_r[threshold_var].values.astype(float))
                m = sm.OLS(d_r['voted'].values.astype(float), Xr).fit()
                b0 = float(m.params[0])
                sb0 = float(m.bse[0])
                post_lvl = b0
                post_lo = b0 - z_lvl * sb0
                post_hi = b0 + z_lvl * sb0

        out.update({
            f'tau_{g}': tau_g, f'se_{g}': se_g, f'p_{g}': p_g,
            f'ci_lo_{g}': ci_lo, f'ci_hi_{g}': ci_hi,
            f'bw_{g}': h_l,
            f'post_{g}': post_lvl,
            f'post_lo_{g}': post_lo, f'post_hi_{g}': post_hi,
            f'n_eff_l_{g}': n_eff_l, f'n_eff_r_{g}': n_eff_r,
        })

    # M − F gap with normal-approx p-value.
    if np.isfinite(out.get('tau_M', np.nan)) and np.isfinite(out.get('tau_F', np.nan)):
        from scipy.stats import norm as _norm
        gap = out['tau_M'] - out['tau_F']
        se_gap = float(np.sqrt(out['se_M']**2 + out['se_F']**2))
        z = gap / se_gap if se_gap > 0 else np.nan
        p_gap = 2 * (1 - _norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    else:
        gap, se_gap, p_gap = np.nan, np.nan, np.nan
    out.update({'gap_MF': gap, 'se_gap': se_gap, 'p_gap': p_gap})
    return out


def run_ses_decile_rdd(df, ses_var, threshold, n_deciles=10,
                       bandwidth=365*3, alpha_level=0.10, verbose=True,
                       cluster_var=None):
    """
    Assign individuals to `n_deciles` population-weighted SES deciles, then
    run the age RD separately by sex inside each decile (default rdrobust
    settings — to match 04_rdd.ipynb). If `cluster_var` is a column of `df`,
    per-decile SEs are cluster-robust on it (e.g. the electoral circuit);
    otherwise they are unclustered (the default). Compute the M-F gap with a
    normal-approx p-value and post-cutoff turnout levels per sex with
    (1-alpha_level)*100% CIs.

    Returns (decile_df, boundaries).
    """
    threshold_var = f'days_from_{threshold}'
    d = df[df['gender'].isin(['F', 'M'])
           & df[threshold_var].between(-bandwidth, bandwidth)].copy()

    d['ses_decile'] = pd.qcut(
        d[ses_var], q=n_deciles, labels=False, duplicates='drop') + 1

    quantiles = np.linspace(0, 1, n_deciles + 1)
    boundaries = d[ses_var].quantile(quantiles).values

    if verbose:
        print(f"\n[{ses_var.upper()} × T{threshold}] decile boundaries: "
              + ", ".join(f"{b:.4f}" for b in boundaries))

    rows = []
    for q in sorted(d['ses_decile'].dropna().unique()):
        sub = d[d['ses_decile'] == q]
        res = _ses_decile_rdd_one(sub, threshold_var, q, alpha_level,
                                  cluster_var=cluster_var)
        res['ses_var'] = ses_var
        res['threshold'] = threshold
        rows.append(res)
        if verbose:
            tF = res['tau_F'] * 100
            tM = res['tau_M'] * 100
            gm = res['gap_MF'] * 100
            pp = res['p_gap']
            print(f"  Decile {int(q):2d}: n={res['n_voters']:>7,}  "
                  f"tau_F={tF:+6.2f}pp  tau_M={tM:+6.2f}pp  "
                  f"M-F={gm:+5.2f}pp (p={pp:.3f})")

    return pd.DataFrame(rows), boundaries


def run_ses_continuous_interaction(df, ses_var, threshold,
                                   bandwidth=None, poly_degree=1,
                                   bandwidth_max=365*3, verbose=True,
                                   cluster_var=None):
    """
    Single RD over the full sample with the discontinuity term fully
    interacted with sex and continuous SES:
        voted ~ x * T * ses * female,   T = 1[x >= 0]
    Fit inside the unconditional MSE-optimal rdrobust bandwidth (or a
    user-specified one).

    Variance estimator: HC1 robust SEs by default. If `cluster_var` is given
    (a column of `df` identifying the clustering unit, e.g. the electoral
    circuit), SEs are cluster-robust (CRV1) on that variable instead. SES is
    constant within a circuit and turnout residuals are correlated within it
    (Moulton problem), so clustering by circuit is the appropriate estimator
    for the SES-dependent terms; HC1 understates their SEs.

    Reports the implied pp change in the cutoff effect (per sex) and in the
    M-F gap from the 10th to the 90th SES percentile, and returns a grid
    of marginal cutoff effects across SES (with SEs) for plotting.
    """
    import warnings
    import statsmodels.api as sm
    from rdrobust import rdrobust

    threshold_var = f'days_from_{threshold}'
    d = df[df['gender'].isin(['F', 'M'])
           & df[threshold_var].between(-bandwidth_max, bandwidth_max)].copy()

    if bandwidth is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r0 = rdrobust(y=d['voted'].values, x=d[threshold_var].values,
                          c=0, masspoints='off')
        bandwidth = float(r0.bws.iloc[0, 0])
        bw_note = 'MSE-optimal (unconditional)'
    else:
        bw_note = 'user-specified'
    if verbose:
        print(f"  Continuous interaction BW: {bandwidth:.1f}d ({bw_note})")

    d_w = d[d[threshold_var].between(-bandwidth, bandwidth)].copy()
    x = d_w[threshold_var].values.astype(float)
    T = (x >= 0).astype(float)
    s = d_w[ses_var].values.astype(float)
    f = (d_w['gender'] == 'F').astype(float).values
    y = d_w['voted'].values.astype(float)

    # Column order (16): x_pow ∈ {0,1}, T ∈ {0,1}, ses ∈ {0,1}, female ∈ {0,1}
    # idx = (fem<<3) | (ses<<2) | (T<<1) | (x_pow)
    def build_X(x, T, s, f):
        cols = []
        for fem in [None, f]:
            for ses in [None, s]:
                for trt in [None, T]:
                    for xt in [None, x]:
                        col = np.ones_like(x, dtype=float)
                        for arr in (xt, trt, ses, fem):
                            if arr is not None:
                                col = col * arr
                        cols.append(col)
        return np.column_stack(cols)

    X = build_X(x, T, s, f)
    if cluster_var is not None:
        groups = d_w[cluster_var].to_numpy()
        ols = sm.OLS(y, X).fit(cov_type='cluster',
                               cov_kwds={'groups': groups})
    else:
        ols = sm.OLS(y, X).fit(cov_type='HC1')
    beta = np.asarray(ols.params)
    V = np.asarray(ols.cov_params())

    p10 = float(np.quantile(s, 0.10))
    p90 = float(np.quantile(s, 0.90))
    delta_s = p90 - p10

    # Column indices for the relevant interaction terms.
    # i_T = 2  (T)
    # i_T_ses = 6  (T * ses)
    # i_T_fem = 10 (T * female)
    # i_T_ses_fem = 14 (T * ses * female)

    def contrast(idx_w):
        c = np.zeros(16)
        for i, w in idx_w.items():
            c[i] = w
        val = float(c @ beta)
        se  = float(np.sqrt(c @ V @ c))
        return val, se

    dtau_M, se_dtau_M = contrast({6: delta_s})                    # f=0
    dtau_F, se_dtau_F = contrast({6: delta_s, 14: delta_s})       # f=1
    dgap_MF, se_dgap = contrast({14: -delta_s})                   # M − F change

    # Grid for marginal cutoff effect across SES.
    s_grid = np.linspace(float(s.min()), float(s.max()), 200)
    tau_M_grid = np.zeros_like(s_grid)
    se_M_grid  = np.zeros_like(s_grid)
    tau_F_grid = np.zeros_like(s_grid)
    se_F_grid  = np.zeros_like(s_grid)
    for k, sv in enumerate(s_grid):
        v, se = contrast({2: 1.0, 6: sv})                         # f=0
        tau_M_grid[k], se_M_grid[k] = v, se
        v, se = contrast({2: 1.0, 6: sv, 10: 1.0, 14: sv})        # f=1
        tau_F_grid[k], se_F_grid[k] = v, se

    if verbose:
        print(f"  Δ cutoff effect (p10→p90): "
              f"M={dtau_M*100:+.2f}pp (SE {se_dtau_M*100:.2f}), "
              f"F={dtau_F*100:+.2f}pp (SE {se_dtau_F*100:.2f})")
        print(f"  Δ M−F gap (p10→p90): "
              f"{dgap_MF*100:+.2f}pp (SE {se_dgap*100:.2f})")

    return {
        'ses_var': ses_var, 'threshold': threshold,
        'bandwidth': bandwidth, 'p10': p10, 'p90': p90, 'delta_s': delta_s,
        'dtau_M': dtau_M, 'se_dtau_M': se_dtau_M,
        'dtau_F': dtau_F, 'se_dtau_F': se_dtau_F,
        'dgap_MF': dgap_MF, 'se_dgap_MF': se_dgap,
        's_grid': s_grid,
        'tau_M_grid': tau_M_grid, 'se_M_grid': se_M_grid,
        'tau_F_grid': tau_F_grid, 'se_F_grid': se_F_grid,
        'ols': ols,
    }


def ses_gradient_test(decile_df, verbose=True):
    """
    Inverse-variance-weighted WLS of decile-level coefficients on decile rank.
    Run separately for tau_M, tau_F, and gap_MF. Returns a dict; reports slope
    (pp per decile), SE, and total pp change across the SES range.
    """
    import statsmodels.api as sm
    from scipy.stats import t as t_dist

    d = decile_df.dropna(subset=['tau_F', 'tau_M', 'se_F', 'se_M',
                                 'gap_MF', 'se_gap']).copy()
    n_dec = int(d['decile'].max())
    span = n_dec - 1
    out = {'n_deciles': n_dec, 'span_units': span}

    for name, tau_col, se_col in [
        ('tau_M',  'tau_M',  'se_M'),
        ('tau_F',  'tau_F',  'se_F'),
        ('gap_MF', 'gap_MF', 'se_gap'),
    ]:
        y = d[tau_col].values
        w = 1.0 / (d[se_col].values ** 2)
        X = sm.add_constant(d['decile'].values.astype(float))
        wls = sm.WLS(y, X, weights=w).fit()
        slope = float(wls.params[1])
        slope_se = float(wls.bse[1])
        t_stat = slope / slope_se if slope_se > 0 else np.nan
        df_res = wls.df_resid
        p = 2 * (1 - t_dist.cdf(abs(t_stat), df_res)) if np.isfinite(t_stat) else np.nan
        out[f'{name}_slope']      = slope
        out[f'{name}_slope_se']   = slope_se
        out[f'{name}_p']          = p
        out[f'{name}_total_pp']   = slope * span
        out[f'{name}_total_se']   = slope_se * span

    if verbose:
        print(f"  Gradient test (WLS, IV-weighted, span = {span} deciles):")
        for name, label in [('tau_M', 'M effect'),
                            ('tau_F', 'F effect'),
                            ('gap_MF', 'M−F gap')]:
            sl = out[f'{name}_slope'] * 100
            se = out[f'{name}_slope_se'] * 100
            tot = out[f'{name}_total_pp'] * 100
            p = out[f'{name}_p']
            print(f"    {label:9s}: slope={sl:+5.3f}pp/dec  SE={se:.3f}  "
                  f"p={p:.3f}  total={tot:+5.2f}pp")
    return out


def gradient_clustering_robustness(df, ses_measures=('nbi', 'pca_index'),
                                   threshold=18, n_deciles=10, bandwidth=365 * 3,
                                   cluster_cols=('province_id', 'circuit_id')):
    """
    Robustness of the SES gradient to clustering standard errors by electoral
    circuit (Moulton). SES is constant within a circuit and turnout residuals
    are correlated within it, so default (unclustered) SEs on the SES-dependent
    terms can be understated. Point estimates are invariant to the variance
    estimator; only the SEs and p-values change. For each SES measure at the
    given threshold this compares, side by side:

      (A) continuous-interaction Δτ(p10→p90) per sex and the M−F gap change,
          with HC1 vs cluster-by-circuit SEs (CRV1);
      (B) the inverse-variance decile slope test (pp per decile) with per-decile
          RD SEs unclustered vs clustered by circuit.

    Returns a tidy DataFrame: one row per (SES × quantity × estimator) with the
    estimate (pp), SE (pp), p-value, and the implied SE inflation factor.
    """
    from scipy.stats import norm

    d = df.copy()
    d['_cluster_id'] = d.groupby(list(cluster_cols)).ngroup()
    rows = []

    for ses in ses_measures:
        # (A) continuous interaction: HC1 vs clustered ----------------------
        store = {}
        for est_label, cvar in [('HC1', None), ('cluster', '_cluster_id')]:
            r = run_ses_continuous_interaction(
                d, ses_var=ses, threshold=threshold, bandwidth=None,
                bandwidth_max=bandwidth, cluster_var=cvar, verbose=False)
            store[est_label] = r
            for q, vkey, skey in [
                ('dtau_M(p10->p90)', 'dtau_M', 'se_dtau_M'),
                ('dtau_F(p10->p90)', 'dtau_F', 'se_dtau_F'),
                ('dgap_MF(p10->p90)', 'dgap_MF', 'se_dgap_MF'),
            ]:
                val, se = r[vkey], r[skey]
                z = val / se if se > 0 else np.nan
                rows.append({
                    'ses': ses, 'block': 'continuous', 'quantity': q,
                    'estimator': est_label, 'estimate_pp': val * 100,
                    'se_pp': se * 100,
                    'p': 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan,
                    'bw_days': r['bandwidth']})

        # (B) decile slope test: unclustered vs clustered -------------------
        for est_label, cvar in [('unclustered', None), ('cluster', '_cluster_id')]:
            dec, _ = run_ses_decile_rdd(
                d, ses_var=ses, threshold=threshold, n_deciles=n_deciles,
                bandwidth=bandwidth, verbose=False, cluster_var=cvar)
            g = ses_gradient_test(dec, verbose=False)
            for name in ['tau_M', 'tau_F', 'gap_MF']:
                rows.append({
                    'ses': ses, 'block': 'decile_slope',
                    'quantity': f'{name}_slope', 'estimator': est_label,
                    'estimate_pp': g[f'{name}_slope'] * 100,
                    'se_pp': g[f'{name}_slope_se'] * 100,
                    'p': g[f'{name}_p'], 'bw_days': np.nan})

    out = pd.DataFrame(rows)

    # SE inflation factor (clustered / baseline) per quantity.
    base = {'continuous': 'HC1', 'decile_slope': 'unclustered'}
    infl = {}
    for (blk, q), grp in out.groupby(['block', 'quantity']):
        b = grp[grp['estimator'] == base[blk]]['se_pp']
        c = grp[grp['estimator'] == 'cluster']['se_pp']
        if len(b) and len(c) and float(b.iloc[0]) > 0:
            infl[(blk, q)] = float(c.iloc[0]) / float(b.iloc[0])
    out['se_inflation'] = out.apply(
        lambda r: infl.get((r['block'], r['quantity']), np.nan)
        if r['estimator'] == 'cluster' else np.nan, axis=1)
    return out


def density_test(df, threshold_var, bandwidth_max=365 * 3):
    """
    Cattaneo--Jansson--Ma local-polynomial manipulation (density) test at the
    cutoff, run on the ±`bandwidth_max` window of `threshold_var`. Returns the
    robust jackknife T statistic and p-value, plus the left/right density
    estimates. With a non-manipulable running variable a rejection reflects
    integer-day heaping and the test's hypersensitivity at very large N, not
    sorting; it must be read alongside covariate balance.
    """
    import warnings
    from rddensity import rddensity

    d = df[df[threshold_var].between(-bandwidth_max, bandwidth_max)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = rddensity(d[threshold_var].to_numpy(dtype=float), c=0)
    return {
        'threshold_var': threshold_var,
        't_jk': float(r.test['t_jk']),
        'p_jk': float(r.test['p_jk']),
        'dens_left':  float(r.hat['left']),
        'dens_right': float(r.hat['right']),
        'n': int(len(d)),
    }


# ── Plotting helpers (each takes per-threshold result dicts) ──────────────

# Human-readable names for the SES measures. Figure axes and titles must never
# show the raw column name (`pca_index`, `PCA_INDEX`): these figures go into the
# manuscript, where a variable name reads as unfinished work.
_SES_LABELS = {
    'pca_index': ('PCA deprivation', 'PCA deprivation index'),
    'nbi':       ('NBI', 'Unmet Basic Needs (NBI)'),
}


def _ses_label(ses_var, long=False):
    """Display name for an SES measure; falls back to the raw column name."""
    short, full = _SES_LABELS.get(ses_var, (ses_var.upper(), ses_var))
    return full if long else short


def plot_decile_rdd_per_sex(decile_by_thr, ses_var, figures_dir, filename=None):
    """Two-panel (T18, T70) plot: tau_F and tau_M across deciles with 95% CIs."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, t in zip(axes, [18, 70]):
        d = decile_by_thr[t]
        x = d['decile'].values
        for sex, col_lo, col_hi, color, label in [
            ('F', 'ci_lo_F', 'ci_hi_F', '#c44e52', 'Women (95% CI)'),
            ('M', 'ci_lo_M', 'ci_hi_M', '#4c72b0', 'Men (95% CI)'),
        ]:
            tau = d[f'tau_{sex}'].values * 100
            lo  = d[col_lo].values * 100
            hi  = d[col_hi].values * 100
            yerr = np.vstack([tau - lo, hi - tau])
            ax.errorbar(x, tau, yerr=yerr, marker='o', color=color,
                        capsize=3, linewidth=1.5, label=label)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel(f'{_ses_label(ses_var)} decile (1 = least deprived)')
        ax.set_ylabel('RD effect on turnout (pp)')
        ax.set_title(f'T{t}: per-sex RD across {_ses_label(ses_var, long=True)} deciles')
        ax.set_xticks(range(1, int(d['decile'].max()) + 1))
        ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fname = filename or f'rdd_gradient_per_sex_{ses_var}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved: {figures_dir}/{fname}")


def plot_decile_rdd_gap(decile_by_thr, ses_var, figures_dir, filename=None):
    """Two-panel plot: M-F gap in RD effect across deciles with 95% CIs."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, t in zip(axes, [18, 70]):
        d = decile_by_thr[t]
        x = d['decile'].values
        gap = d['gap_MF'].values * 100
        se  = d['se_gap'].values * 100
        ax.errorbar(x, gap, yerr=1.96 * se, marker='o', color='#444',
                    capsize=3, linewidth=1.5, label='M − F gap (95% CI)')
        ax.axhline(0, color='black', linestyle='--', linewidth=1,
                   label='Gender parity')
        ax.set_xlabel(f'{_ses_label(ses_var)} decile')
        ax.set_ylabel('M − F RD effect (pp)')
        ax.set_title(f'T{t}: M−F RD gap across {_ses_label(ses_var, long=True)} deciles')
        ax.set_xticks(range(1, int(d['decile'].max()) + 1))
        ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fname = filename or f'rdd_gradient_gap_{ses_var}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved: {figures_dir}/{fname}")


def plot_marginal_cutoff_continuous(cont_by_thr, ses_var, figures_dir,
                                    filename=None):
    """
    Two-panel plot of the marginal cutoff effect as a function of continuous
    SES for each sex, with 95% bands. p10 and p90 marked with vertical
    dotted lines.
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, t in zip(axes, [18, 70]):
        r = cont_by_thr[t]
        s = r['s_grid']
        for sex_key, color, label in [
            ('F', '#c44e52', 'Women'),
            ('M', '#4c72b0', 'Men'),
        ]:
            tau = r[f'tau_{sex_key}_grid'] * 100
            se  = r[f'se_{sex_key}_grid'] * 100
            ax.plot(s, tau, color=color, linewidth=2, label=label)
            ax.fill_between(s, tau - 1.96 * se, tau + 1.96 * se,
                            color=color, alpha=0.15)
        ax.axvline(r['p10'], color='gray', linestyle=':', linewidth=0.8)
        ax.axvline(r['p90'], color='gray', linestyle=':', linewidth=0.8)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel(f'{_ses_label(ses_var, long=True)} (continuous)')
        ax.set_ylabel('Marginal cutoff effect (pp)')
        ax.set_title(f'T{t}: marginal cutoff effect over {_ses_label(ses_var)}')
        ax.legend(fontsize=9, framealpha=0.9, loc='best')
    plt.tight_layout()
    fname = filename or f'rdd_gradient_marginal_{ses_var}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved: {figures_dir}/{fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 03_analysis.ipynb — descriptive geography and RD-window context figures.
# ─────────────────────────────────────────────────────────────────────────────

# Electoral province_id → INDEC codprov_censo (Censo 2022 province codes).
ELECTORAL_TO_INDEC = {
    1: '02',  2: '06',  3: '10',  4: '14',  5: '18',  6: '22',  7: '26',
    8: '30',  9: '34',  10: '38', 11: '42', 12: '46', 13: '50', 14: '54',
    15: '58', 16: '62', 17: '66', 18: '70', 19: '74', 20: '78', 21: '82',
    22: '86', 23: '90', 24: '94',
}


def plot_provincial_turnout(df, prov_geo_path, figures_dir, province_names=None,
                            cmap='YlGnBu', filename='provincial_turnout_map.png'):
    """Choropleth of raw average turnout by province (full padron).

    Descriptive counterpart to the RDD-effect map in 06_provincial.ipynb: this
    shows the observed turnout *level*, not a causal estimate. Province
    boundaries are read from `prov_geo_path` (a GeoJSON with an INDEC
    `codprov_censo` column) and matched to electoral `province_id` via
    ELECTORAL_TO_INDEC. Returns the merged GeoDataFrame.
    """
    import geopandas as gpd
    os.makedirs(figures_dir, exist_ok=True)

    prov = (
        df.groupby('province_id')['voted']
        .agg(turnout='mean', n_voters='count')
        .reset_index()
    )
    prov['turnout_pct'] = prov['turnout'] * 100
    prov['province'] = (prov['province_id'].map(province_names)
                        if province_names is not None
                        else prov['province_id'].astype(str))

    gdf = gpd.read_file(prov_geo_path)
    indec_to_electoral = {v: k for k, v in ELECTORAL_TO_INDEC.items()}
    gdf['province_id'] = gdf['codprov_censo'].map(indec_to_electoral)
    gmap = gdf.merge(prov, on='province_id', how='left')

    fig, ax = plt.subplots(figsize=(8, 11))
    gmap.plot(
        column='turnout_pct', ax=ax, cmap=cmap, legend=True,
        missing_kwds={'color': 'lightgrey', 'label': 'No data'},
        legend_kwds={'label': 'Turnout (%)', 'shrink': 0.55, 'pad': 0.01},
        edgecolor='white', linewidth=0.4,
    )
    ax.set_xlim(-74, -52)
    ax.set_ylim(-56, -21)
    ax.set_axis_off()
    ax.set_title('Raw Voter Turnout by Province — Argentina 2025', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{filename}", bbox_inches='tight', dpi=150)
    plt.show()

    rng = prov.sort_values('turnout_pct')
    print(f"Provinces: {len(prov)} — national turnout {df['voted'].mean()*100:.1f}%")
    print(f"  Lowest:  {rng.iloc[0]['province']} ({rng.iloc[0]['turnout_pct']:.1f}%)")
    print(f"  Highest: {rng.iloc[-1]['province']} ({rng.iloc[-1]['turnout_pct']:.1f}%)")
    return gmap


def plot_threshold_density(df, figures_dir, window_days=365*3, bw_band=None,
                           bins=120, filename='threshold_density.png'):
    """Histograms of days_from_18 and days_from_70 contextualising the RD window.

    Shows the density of registered voters by distance from each age cutoff
    within ±`window_days`. A representative analysis bandwidth `bw_band` (days;
    defaults to `window_days // 3`) is shaded to convey how many voters fall
    inside a typical RD window. Distinct from the McCrary visual, which tests
    for a *discontinuity* in this density at the cutoff.
    """
    os.makedirs(figures_dir, exist_ok=True)
    if bw_band is None:
        bw_band = window_days // 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, short, color in [
        (axes[0], 'days_from_18', 'age 18 (compulsory starts)', 'red'),
        (axes[1], 'days_from_70', 'age 70 (compulsory ends)',   'orange'),
    ]:
        data = df.loc[df[col].between(-window_days, window_days), col]
        ax.hist(data, bins=bins, color='steelblue', alpha=0.85, edgecolor='none')
        ax.axvspan(-bw_band, bw_band, color=color, alpha=0.12,
                   label=f'±{bw_band}d window')
        ax.axvline(0, color=color, linestyle='--', linewidth=1.5,
                   label=f'Threshold: {short}')
        n_band = int(df[col].between(-bw_band, bw_band).sum())
        ax.set_xlabel('Days from threshold')
        ax.set_ylabel('Number of voters')
        ax.set_title(f'Voter density around {short}')
        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        ax.annotate(f'n = {n_band:,}\nwithin ±{bw_band}d',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.85))
    fig.suptitle('Density of Voters Around the RD Thresholds — Argentina 2025',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{filename}", bbox_inches='tight', dpi=150)
    plt.show()


def gender_gap_means_by_decile(df, ses_var, threshold=18, n_deciles=10,
                               bandwidth=365*3, alpha_level=0.05,
                               side='right', means_window=90):
    """
    Raw Women − Men turnout difference within each SES decile, measured locally
    at the age-`threshold` cutoff.

    Deciles are defined exactly as in `run_ses_decile_rdd` — `pd.qcut` on
    `ses_var` over the full F/M sample inside ±`bandwidth` days of the cutoff —
    so they line up one-to-one with the per-sex RD table regardless of which
    voters enter the means.

    The group means are then computed on a *narrow* window near the cutoff so
    the gap is read close to the threshold (comparable to the local RD), not
    averaged over several age cohorts. `means_window` is the half-width in days
    (None ⇒ use the full `bandwidth`); `side` picks which part of it is used:
      - `side='right'`: voters in [0, means_window]  — the compulsory side,
        i.e. the gender gap just after turning `threshold`;
      - `side='left'`:  voters in [-means_window, 0)  — the voluntary side;
      - `side='both'`:  voters in [-means_window, means_window].

    For each decile this is a simple comparison of group means — not an RD
    estimate — returning mean turnout per sex, the Women − Men difference, a
    (1 - alpha_level) CI (unpooled binomial SE) and a two-proportion z-test
    p-value (pooled SE). Higher `ses_var` ⇒ more deprived, so decile 1 = least
    deprived. Returns a DataFrame.
    """
    from scipy.stats import norm

    if side not in ('right', 'left', 'both'):
        raise ValueError("side must be 'right', 'left' or 'both'")

    threshold_var = f'days_from_{threshold}'
    d = df[df['gender'].isin(['F', 'M'])
           & df[threshold_var].between(-bandwidth, bandwidth)].copy()
    # Deciles defined on the full window so they match the RD panel exactly.
    d['ses_decile'] = pd.qcut(
        d[ses_var], q=n_deciles, labels=False, duplicates='drop') + 1
    # Restrict to a narrow window on the requested side for the group means.
    mw = bandwidth if means_window is None else means_window
    if side == 'right':
        d = d[(d[threshold_var] >= 0) & (d[threshold_var] <= mw)]
    elif side == 'left':
        d = d[(d[threshold_var] < 0) & (d[threshold_var] >= -mw)]
    else:  # both
        d = d[d[threshold_var].between(-mw, mw)]

    z = norm.ppf(1 - alpha_level / 2)
    rows = []
    for q in sorted(d['ses_decile'].dropna().unique()):
        sub = d[d['ses_decile'] == q]
        F = sub.loc[sub['gender'] == 'F', 'voted'].to_numpy(dtype=float)
        M = sub.loc[sub['gender'] == 'M', 'voted'].to_numpy(dtype=float)
        nF, nM = len(F), len(M)
        if nF == 0 or nM == 0:
            continue
        pF, pM = float(F.mean()), float(M.mean())
        diff = pF - pM                                  # Women − Men
        se_unpooled = float(np.sqrt(pF * (1 - pF) / nF + pM * (1 - pM) / nM))
        p_pool = float((F.sum() + M.sum()) / (nF + nM))
        se_pool = float(np.sqrt(p_pool * (1 - p_pool) * (1 / nF + 1 / nM)))
        zstat = diff / se_pool if se_pool > 0 else np.nan
        pval = 2 * (1 - norm.cdf(abs(zstat))) if np.isfinite(zstat) else np.nan
        rows.append({
            'decile': int(q), 'n_F': nF, 'n_M': nM,
            'turnout_F': pF, 'turnout_M': pM, 'diff_FM': diff,
            'se': se_unpooled, 'ci_lo': diff - z * se_unpooled,
            'ci_hi': diff + z * se_unpooled, 'z': zstat, 'p_value': pval,
        })
    return pd.DataFrame(rows)


def pooled_rd_by_decile(df, ses_var, threshold=18, n_deciles=10,
                        bandwidth=365*3, min_obs=200):
    """
    Pooled (Women + Men) RD jump at the age-`threshold` cutoff within each SES
    decile.

    Uses the same windowed F/M sample and `pd.qcut` boundaries as
    `run_ses_decile_rdd`, and the same rdrobust settings (`c=0`,
    `masspoints='rd'`, default triangular kernel / p=1 / MSE-optimal BW, no
    clustering). Returns a DataFrame with decile, tau, se, ci_lo, ci_hi (robust
    CI), so it lines up one-to-one with the per-sex decile table.
    """
    import warnings
    from rdrobust import rdrobust

    threshold_var = f'days_from_{threshold}'
    d = df[df['gender'].isin(['F', 'M'])
           & df[threshold_var].between(-bandwidth, bandwidth)].copy()
    d['ses_decile'] = pd.qcut(
        d[ses_var], q=n_deciles, labels=False, duplicates='drop') + 1

    rows = []
    for q in sorted(d['ses_decile'].dropna().unique()):
        sub = d[d['ses_decile'] == q]
        rec = {'decile': int(q), 'tau': np.nan, 'se': np.nan,
               'ci_lo': np.nan, 'ci_hi': np.nan}
        if len(sub) >= min_obs and sub['voted'].nunique() >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = rdrobust(y=sub['voted'].values,
                                 x=sub[threshold_var].values,
                                 c=0, masspoints='rd')
                rec['tau']   = float(r.coef.iloc[0, 0])
                rec['se']    = float(r.se.iloc[0, 0])
                rec['ci_lo'] = float(r.ci.iloc[2, 0])
                rec['ci_hi'] = float(r.ci.iloc[2, 1])
            except Exception:
                pass
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_gender_means_and_rd(df, decile_rd_df, ses_var, figures_dir,
                             threshold=18, n_deciles=10, bandwidth=365*3,
                             alpha_level=0.05, side='right', means_window=90,
                             filename=None):
    """
    Two-panel figure at the age-`threshold` cutoff across `ses_var` deciles.

    LEFT — descriptive gender gap (group means): the raw Women − Men turnout
    difference per decile (`gender_gap_means_by_decile`), with (1-alpha_level)
    CIs and two-proportion z-test p-values. Points above 0 ⇒ women vote more
    (shaded red); below 0 ⇒ men vote more (shaded blue). Significant points
    (p < alpha_level) are filled; non-significant are hollow. The means are
    read on a narrow `means_window` (days) near the cutoff; `side` selects
    which part of it ('right' = compulsory side, the default — the gap just
    after the cutoff; 'left' = voluntary side; 'both').

    RIGHT — per-sex RD: the RD jump at the cutoff for women and men inside each
    decile, taken from `decile_rd_df` (a `run_ses_decile_rdd` output for this
    threshold), plus the pooled F+M estimate (`pooled_rd_by_decile`), shown as
    points-only dot-and-whisker with 95% CIs.

    Deciles match across panels because both use the same windowed F/M sample
    and `pd.qcut` on `ses_var`. Returns the group-means DataFrame.
    """
    os.makedirs(figures_dir, exist_ok=True)
    cF, cM, cBoth = '#c44e52', '#4c72b0', '#555555'
    mw = bandwidth if means_window is None else means_window
    _side_label = {'right': f'0 to +{mw}d post-cutoff',
                   'left': f'−{mw} to 0d pre-cutoff',
                   'both': f'±{mw}d around cutoff'}[side]

    means = gender_gap_means_by_decile(
        df, ses_var, threshold=threshold, n_deciles=n_deciles,
        bandwidth=bandwidth, alpha_level=alpha_level, side=side,
        means_window=means_window)

    pct = int(round((1 - alpha_level) * 100))
    print(f"Women − Men turnout gap by {ses_var} decile "
          f"(group means, T{threshold}, {_side_label}):")
    for _, r in means.iterrows():
        star = '*' if (np.isfinite(r['p_value']) and r['p_value'] < alpha_level) else ' '
        print(f"  Decile {int(r['decile']):2d}: F={r['turnout_F']*100:5.1f}%  "
              f"M={r['turnout_M']*100:5.1f}%  F−M={r['diff_FM']*100:+5.2f}pp "
              f"[{r['ci_lo']*100:+5.2f}, {r['ci_hi']*100:+5.2f}]  "
              f"p={r['p_value']:.3g} {star}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── LEFT: group-means Women − Men gap ─────────────────────────────────
    ax = axes[0]
    x = means['decile'].to_numpy()
    diff = means['diff_FM'].to_numpy() * 100
    lo = means['ci_lo'].to_numpy() * 100
    hi = means['ci_hi'].to_numpy() * 100
    sig = (means['p_value'].to_numpy() < alpha_level)

    ymax = max(float(np.nanmax(hi)), 0.5) * 1.15
    ymin = min(float(np.nanmin(lo)), -0.5) * 1.15
    ax.axhspan(0, ymax, color=cF, alpha=0.06)
    ax.axhspan(ymin, 0, color=cM, alpha=0.06)
    ax.text(0.98, 0.97, 'Women vote more', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color=cF)
    ax.text(0.98, 0.03, 'Men vote more', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color=cM)

    for xi, di, li, hi_i, si in zip(x, diff, lo, hi, sig):
        col = cF if di >= 0 else cM
        ax.errorbar(xi, di, yerr=[[di - li], [hi_i - di]], fmt='o',
                    color=col, capsize=3, linewidth=1.5,
                    markerfacecolor=(col if si else 'white'),
                    markeredgecolor=col, markersize=7, zorder=3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    _title_tag = {'right': f'0–{mw}d post-cutoff',
                  'left': f'{mw}d pre-cutoff',
                  'both': f'±{mw}d'}[side]
    ax.set_xlabel(f'{_ses_label(ses_var)} decile (1 = least deprived)')
    ax.set_ylabel('Women − Men turnout gap (pp)')
    ax.set_title(f'Raw gender gap by {_ses_label(ses_var, long=True)} decile '
                 f'(group means, {_title_tag}, {pct}% CI)')
    # Legend proxy for fill = significance.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='gray', linestyle='',
               markerfacecolor='gray', label=f'p < {alpha_level:g} (significant)'),
        Line2D([0], [0], marker='o', color='gray', linestyle='',
               markerfacecolor='white', label=f'p ≥ {alpha_level:g} (n.s.)'),
    ], fontsize=8, framealpha=0.9, loc='lower left')

    # ── RIGHT: per-sex and pooled RD across deciles (points only) ──────────
    ax = axes[1]
    d = decile_rd_df.sort_values('decile')
    xr = d['decile'].to_numpy()

    # Pooled (Women + Men) RD per decile — same sample, deciles and settings.
    pooled = pooled_rd_by_decile(
        df, ses_var, threshold=threshold, n_deciles=n_deciles,
        bandwidth=bandwidth).sort_values('decile')

    # Slight horizontal dodge so the three estimates per decile don't overlap.
    series = [
        ('Women',         cF,    -0.22, d['tau_F'].to_numpy(),
         d['ci_lo_F'].to_numpy(),   d['ci_hi_F'].to_numpy()),
        ('Both (pooled)', cBoth,  0.00, pooled['tau'].to_numpy(),
         pooled['ci_lo'].to_numpy(), pooled['ci_hi'].to_numpy()),
        ('Men',           cM,     0.22, d['tau_M'].to_numpy(),
         d['ci_lo_M'].to_numpy(),   d['ci_hi_M'].to_numpy()),
    ]
    for label, color, dx, tau, clo, chi in series:
        tau, clo, chi = tau * 100, clo * 100, chi * 100
        yerr = np.vstack([tau - clo, chi - tau])
        ax.errorbar(xr + dx, tau, yerr=yerr, fmt='o', color=color,
                    capsize=3, markersize=6, linestyle='none',
                    elinewidth=1.3, label=f'{label} (95% CI)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks(xr)
    ax.set_xlabel(f'{_ses_label(ses_var)} decile (1 = least deprived)')
    ax.set_ylabel('RD effect on turnout (pp)')
    ax.set_title(f'RD at age {threshold} across {_ses_label(ses_var, long=True)} deciles')
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle(
        f'Gender turnout gap vs RD effect by SES decile — age-{threshold} cutoff',
        fontweight='bold')
    plt.tight_layout()
    fname = filename or f'rdd_gradient_meansgap_vs_rd_{ses_var}_T{threshold}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved: {figures_dir}/{fname}")
    return means


def _decile_rd_points_panel(ax, df, decile_rd_df, ses_var, threshold,
                            n_deciles, bandwidth):
    """Per-sex and pooled RD jump per SES decile as a points-only panel."""
    cF, cM, cBoth = '#c44e52', '#4c72b0', '#555555'
    d = decile_rd_df.sort_values('decile')
    xr = d['decile'].to_numpy()
    pooled = pooled_rd_by_decile(
        df, ses_var, threshold=threshold, n_deciles=n_deciles,
        bandwidth=bandwidth).sort_values('decile')
    series = [
        ('Women',         cF,    -0.22, d['tau_F'].to_numpy(),
         d['ci_lo_F'].to_numpy(),   d['ci_hi_F'].to_numpy()),
        ('Both (pooled)', cBoth,  0.00, pooled['tau'].to_numpy(),
         pooled['ci_lo'].to_numpy(), pooled['ci_hi'].to_numpy()),
        ('Men',           cM,     0.22, d['tau_M'].to_numpy(),
         d['ci_lo_M'].to_numpy(),   d['ci_hi_M'].to_numpy()),
    ]
    for label, color, dx, tau, clo, chi in series:
        tau, clo, chi = tau * 100, clo * 100, chi * 100
        yerr = np.vstack([tau - clo, chi - tau])
        ax.errorbar(xr + dx, tau, yerr=yerr, fmt='o', color=color,
                    capsize=3, markersize=6, linestyle='none',
                    elinewidth=1.3, label=f'{label} (95% CI)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks(xr)
    ax.set_xlabel(f'{_ses_label(ses_var)} decile (1 = least deprived)')
    ax.set_ylabel('RD effect on turnout (pp)')
    ax.set_title(f'RD at age {threshold} across {_ses_label(ses_var, long=True)} deciles')
    ax.legend(fontsize=9, framealpha=0.9)


def plot_gender_gap_rd_and_decile(df, decile_rd_df, ses_var, figures_dir,
                                  threshold=18, n_deciles=10, bandwidth=365*3,
                                  bw_days=365, bin_days=7, filename=None,
                                  rd_result_csv=None,
                                  rd_label='RD estimate (§6.3)',
                                  lowess_frac=0.3):
    """
    Two-panel figure at the age-`threshold` cutoff.

    LEFT — RD visual of the gender gap: the Women − Men turnout difference binned
    every `bin_days` days of the running variable within ±`bw_days`, plotted
    against days from the cutoff (points with 95% two-proportion CIs), a vertical
    line at the cutoff, and a LOWESS fit of the binned gap on each side of the
    cutoff (fitted separately per side, never crossing it; `lowess_frac` matches
    the other LOWESS figures). The LOWESS curves are a visual guide only — the
    annotated estimate of the jump in the F − M gap is NOT derived from them: it
    is the canonical rdrobust common-bandwidth estimate read from
    `rd_result_csv` (the `gender_gap_global_rd_T18.csv` table produced in §8d,
    row "CANONICAL"), i.e. the same figure the thesis text reports. If
    `rd_result_csv` is None the annotation is skipped with a warning.
    Shading: red above 0 (women vote more), blue below (men vote more).

    RIGHT — per-sex and pooled RD jump per SES decile (`decile_rd_df` +
    `pooled_rd_by_decile`), points-only with 95% CIs.

    Returns a dict with the binned-gap table and the canonical RD gap estimate
    (as read from `rd_result_csv`; NaN if not provided).
    """
    os.makedirs(figures_dir, exist_ok=True)
    cF, cM = '#c44e52', '#4c72b0'
    tv = f'days_from_{threshold}'

    sub = df[df['gender'].isin(['F', 'M'])
             & df[tv].between(-bw_days, bw_days)].copy()
    sub['bin'] = (np.floor(sub[tv] / bin_days) * bin_days).astype(int)
    piv = sub.groupby(['bin', 'gender'])['voted'].agg(['mean', 'count']).unstack('gender')
    mF, mM = piv[('mean', 'F')], piv[('mean', 'M')]
    nF, nM = piv[('count', 'F')], piv[('count', 'M')]
    valid = mF.notna() & mM.notna() & (nF > 0) & (nM > 0)
    bins = piv.index[valid].to_numpy()
    mF, mM = mF[valid].to_numpy(), mM[valid].to_numpy()
    nF, nM = nF[valid].to_numpy(), nM[valid].to_numpy()
    gap = (mF - mM) * 100
    se = np.sqrt(mF * (1 - mF) / nF + mM * (1 - mM) / nM) * 100
    centers = bins + bin_days / 2.0

    # LOWESS of the binned gap, fitted separately on each side of the cutoff
    # (decorative — same smoother/frac as the turnout_vs_*_gender_lowess figures).
    left, right = centers < 0, centers >= 0
    smL = lowess(gap[left], centers[left], frac=lowess_frac)
    smR = lowess(gap[right], centers[right], frac=lowess_frac)

    # Trim the OUTER edges of the drawn curves. At |x| ~ bw_days the smoother has
    # one-sided support and chases the last one or two bins, producing a hook that
    # reads as a trend in the gap when it is an artefact of the fit. The cutoff
    # ends are left intact: that is where the curves stand in for the RD limits,
    # and trimming there would remove the informative part. Points are never
    # dropped, only the line.
    _edge = bw_days * 0.92
    smL = smL[smL[:, 0] >= -_edge]
    smR = smR[smR[:, 0] <= _edge]

    # Canonical rdrobust gap estimate for the annotation (read, not computed:
    # single source of truth shared with the thesis text).
    rd_gap = rd_p = rd_ci_lo = rd_ci_hi = np.nan
    if rd_result_csv is not None:
        rd = pd.read_csv(rd_result_csv)
        row = rd[rd['version'].str.contains('CANONICAL')].iloc[0]
        rd_gap, rd_p = float(row['gap_pp']), float(row['p'])
        rd_ci_lo, rd_ci_hi = float(row['ci_lo_pp']), float(row['ci_hi_pp'])
    else:
        print("WARNING: rd_result_csv not given — annotation skipped; "
              "pass outputs/tables/gender_gap_global_rd_T18.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── LEFT: binned F − M gap around the cutoff ───────────────────────────
    ax = axes[0]
    ymax = max(float(np.nanmax(gap + 1.96 * se)),
               float(np.nanmax(smL[:, 1])), float(np.nanmax(smR[:, 1])), 0.5)
    ymin = min(float(np.nanmin(gap - 1.96 * se)),
               float(np.nanmin(smL[:, 1])), float(np.nanmin(smR[:, 1])), -0.5)
    pad = (ymax - ymin) * 0.08
    ax.axhspan(0, ymax + pad, color=cF, alpha=0.05)
    ax.axhspan(ymin - pad, 0, color=cM, alpha=0.05)
    ax.errorbar(centers, gap, yerr=1.96 * se, fmt='o', ms=4, color='#444',
                alpha=0.55, elinewidth=0.7, capsize=0,
                label=f'F−M gap (~{bin_days}d bins, 95% CI)')
    ax.plot(smL[:, 0], smL[:, 1], color='black', lw=2)
    ax.plot(smR[:, 0], smR[:, 1], color='black', lw=2,
            label='LOWESS fit (each side, visual guide)')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold: age {threshold}')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.text(0.98, 0.97, 'Women vote more', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color=cF)
    ax.text(0.98, 0.03, 'Men vote more', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color=cM)
    ax.set_xlabel(f'Days from {threshold}th birthday on election day')
    ax.set_ylabel('Turnout gap, Women − Men (pp)')
    ax.set_title(f'Gender gap around the age-{threshold} cutoff (~{bin_days}-day bins)')
    ax.legend(fontsize=8, framealpha=0.9, loc='lower left')
    if np.isfinite(rd_gap):
        ax.annotate(
            f'{rd_label}: {rd_gap:+.2f}pp, p≈{rd_p:.2f}\n'
            f'95% CI [{rd_ci_lo:+.2f}, {rd_ci_hi:+.2f}]pp (rdrobust, common h)',
            xy=(0.02, 0.97), xycoords='axes fraction', ha='left', va='top',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.85))

    # ── RIGHT: per-sex / pooled RD by SES decile (unchanged) ───────────────
    _decile_rd_points_panel(axes[1], df, decile_rd_df, ses_var, threshold,
                            n_deciles, bandwidth)

    fig.suptitle(
        f'Gender-gap RD vs per-sex RD by SES decile — age-{threshold} cutoff',
        fontweight='bold')
    plt.tight_layout()
    fname = filename or f'rdd_gradient_gaprd_vs_decile_{ses_var}_T{threshold}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    if np.isfinite(rd_gap):
        print(f"Annotated RD gap (canonical, from {rd_result_csv}): "
              f"{rd_gap:+.3f}pp, p={rd_p:.3f}, "
              f"95% CI [{rd_ci_lo:+.2f}, {rd_ci_hi:+.2f}]pp")
    print(f"Saved: {figures_dir}/{fname}")
    return {
        'binned': pd.DataFrame({'bin': bins, 'center': centers,
                                'gap_pp': gap, 'se_pp': se, 'n_F': nF, 'n_M': nM}),
        'rd_gap_pp': rd_gap, 'rd_gap_p': rd_p,
        'rd_ci_lo_pp': rd_ci_lo, 'rd_ci_hi_pp': rd_ci_hi,
    }


def plot_gap_jump_robustness(df, figures_dir, threshold=18,
                             ols_bandwidths=(90, 180, 365, 730),
                             quad_bandwidths=(365, 730), bandwidth_max=365 * 3,
                             filename=None):
    """
    Forest plot of the jump in the Women − Men turnout gap at the age-`threshold`
    cutoff across specifications — to honestly document how robust the small,
    suggestive "equalizing" effect is.

    Specs: the interaction-OLS jump (the `female·right` coefficient of
    `voted ~ x · right · female`, HC1) at each linear `ols_bandwidths` and each
    `quad_bandwidths` (quadratic in x each side); plus the canonical rdrobust
    difference τ_F − τ_M at a COMMON bandwidth (pooled MSE-optimal h fixed for
    both sexes, triangular kernel, robust SE) — matching §8d / the text.
    Each spec is drawn as a point + 95% CI; vertical line at 0; the region < 0
    (gap narrows ⇒ equalizing) is lightly shaded; p-values annotated. Returns
    the specification table.
    """
    import warnings
    import statsmodels.api as sm
    from scipy.stats import norm
    from rdrobust import rdrobust

    tv = f'days_from_{threshold}'
    base = df[df['gender'].isin(['F', 'M'])
              & df[tv].between(-bandwidth_max, bandwidth_max)].copy()
    base['female'] = (base['gender'] == 'F').astype(float)

    def _ols_jump(bw, deg):
        s = base[base[tv].between(-bw, bw)]
        x = s[tv].to_numpy(float)
        f = s['female'].to_numpy(float)
        r = (s[tv] >= 0).astype(float).to_numpy()
        cols = [np.ones_like(x)]
        for p in range(1, deg + 1): cols.append(x ** p)
        cols.append(r)
        for p in range(1, deg + 1): cols.append(r * x ** p)
        cols.append(f)
        for p in range(1, deg + 1): cols.append(f * x ** p)
        cols.append(f * r)                                  # jump in the gap
        for p in range(1, deg + 1): cols.append(f * r * x ** p)
        m = sm.OLS(s['voted'].to_numpy(float), np.column_stack(cols)).fit(cov_type='HC1')
        j = 1 + deg + 1 + deg + 1 + deg
        return m.params[j] * 100, m.bse[j] * 100, m.pvalues[j]

    rows = []
    for bw in ols_bandwidths:
        b, se, p = _ols_jump(bw, 1)
        rows.append({'spec': f'OLS linear, ±{bw}d', 'est': b,
                     'lo': b - 1.96 * se, 'hi': b + 1.96 * se, 'p': p, 'kind': 'lin'})
    for bw in quad_bandwidths:
        b, se, p = _ols_jump(bw, 2)
        rows.append({'spec': f'OLS quadratic, ±{bw}d', 'est': b,
                     'lo': b - 1.96 * se, 'hi': b + 1.96 * se, 'p': p, 'kind': 'quad'})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rp = rdrobust(y=base['voted'].values, x=base[tv].values, c=0, masspoints='rd')
    h_common = float(rp.bws.iloc[0, 0])
    taus = {}
    for g in ['F', 'M']:
        dg = base[base['gender'] == g]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rr = rdrobust(y=dg['voted'].values, x=dg[tv].values, c=0,
                          masspoints='rd', h=h_common)
        taus[g] = (float(rr.coef.iloc[0, 0]) * 100, float(rr.se.iloc[2, 0]) * 100)
    diff = taus['F'][0] - taus['M'][0]
    se_d = float(np.sqrt(taus['F'][1] ** 2 + taus['M'][1] ** 2))
    p_d = 2 * (1 - norm.cdf(abs(diff / se_d)))
    rows.append({'spec': f'rdrobust τF−τM (common h={h_common:.0f}d, robust)',
                 'est': diff, 'lo': diff - 1.96 * se_d, 'hi': diff + 1.96 * se_d,
                 'p': p_d, 'kind': 'rd'})

    tab = pd.DataFrame(rows)

    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(tab) + 1.6))
    y = np.arange(len(tab))[::-1]
    cmap = {'lin': '#4c72b0', 'quad': '#55a868', 'rd': '#c44e52'}
    xmin = min(float(tab['lo'].min()), -0.3)
    xmax = max(float(tab['hi'].max()), 0.6)
    ax.axvspan(xmin * 1.12, 0, color='#c44e52', alpha=0.05)
    ax.text(0.02, 0.02, '← gap narrows (equalizing)', transform=ax.transAxes,
            fontsize=8.5, color='#c44e52', va='bottom', ha='left')
    for yi, (_, r) in zip(y, tab.iterrows()):
        ax.errorbar(r['est'], yi,
                    xerr=[[r['est'] - r['lo']], [r['hi'] - r['est']]],
                    fmt='o', color=cmap[r['kind']], capsize=3, markersize=6)
        ax.text(r['hi'] + (xmax - xmin) * 0.02, yi, f"p={r['p']:.2f}",
                va='center', fontsize=8, color='#333')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tab['spec'])
    ax.set_xlabel('Jump in Women − Men turnout gap at cutoff (pp, 95% CI)')
    ax.set_title(f'Is compulsory voting equalizing at age {threshold}? '
                 f'Gender-gap jump across specifications')
    ax.set_xlim(xmin * 1.12, xmax * 1.18)
    plt.tight_layout()
    fname = filename or f'gap_jump_robustness_T{threshold}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(tab.round(3).to_string(index=False))
    print(f"Saved: {figures_dir}/{fname}")
    return tab


# ─────────────────────────────────────────────────────────────────────────────
# 08_triple_interaction.ipynb — unpack the CV × sex × SES triple interaction.
# ─────────────────────────────────────────────────────────────────────────────

def unpack_triple_interaction(df, ses_var, threshold=18, bandwidth_max=365*3,
                              alpha_level=0.05, cluster_var=None):
    """
    Unpack the CV × sex × SES triple interaction at the age-`threshold` cutoff
    from the continuous spec (`run_ses_continuous_interaction`: a single OLS
    `voted ~ x · T · ses · female`, fit inside the unconditional MSE-optimal
    bandwidth). The spec and its StandardScaler are untouched — this only reads
    coefficients and forms linear contrasts.

    `cluster_var` is passed straight to `run_ses_continuous_interaction`: when a
    column name is given the OLS uses cluster-robust (CRV1) SEs on it (e.g. the
    electoral circuit), otherwise HC1. All reported SEs (triple coefficient and
    the four sex × SES cells) inherit that estimator.

    Returns
    -------
    triple : dict — the triple-interaction coefficient (T·female·ses) with SE,
             z, p, the fitting bandwidth, and the p10/p90 SES anchors.
    cells  : DataFrame — the four sex × SES cells (Men/Women × Low(p10)/High(p90)):
             the marginal cutoff effect `tau` (turnout prob.) with SE, p and a
             (1-alpha_level) CI, and the `baseline` turnout level just left of
             the cutoff (T=0, x=0) with SE and CI. All in probability units;
             multiply by 100 for pp.
    """
    from scipy.stats import norm

    r = run_ses_continuous_interaction(
        df, ses_var=ses_var, threshold=threshold, bandwidth=None,
        poly_degree=1, bandwidth_max=bandwidth_max, verbose=False,
        cluster_var=cluster_var)
    ols = r['ols']
    beta = np.asarray(ols.params)
    V = np.asarray(ols.cov_params())
    p10, p90, bw = r['p10'], r['p90'], r['bandwidth']
    z_a = norm.ppf(1 - alpha_level / 2)

    # Column layout (16): idx = (female<<3)|(ses<<2)|(T<<1)|(x_pow). Relevant:
    #   2=T, 4=ses, 6=T·ses, 8=female, 10=T·female, 12=female·ses, 14=T·ses·female
    def _contrast(idx_w):
        c = np.zeros(len(beta))
        for i, w in idx_w.items():
            c[i] = w
        v = float(c @ beta)
        s = float(np.sqrt(c @ V @ c))
        zz = v / s if s > 0 else np.nan
        pp = 2 * (1 - norm.cdf(abs(zz))) if np.isfinite(zz) else np.nan
        return v, s, pp

    triple = {
        'ses_var': ses_var, 'threshold': threshold,
        'coef': float(ols.params[14]), 'se': float(ols.bse[14]),
        'z': float(ols.tvalues[14]), 'p': float(ols.pvalues[14]),
        'bw_days': bw, 'p10': p10, 'p90': p90,
        # The fitted eq.(1) model itself (cov_type inherited from `cluster_var`),
        # exposed so callers can form joint linear-combination tests (`ols.t_test`)
        # off its covariance without re-fitting. Dropped by callers that build a
        # results frame via explicit column selection.
        'ols': ols,
    }

    rows = []
    for sex, fem in [('Men', 0), ('Women', 1)]:
        for lvl, sv in [('Low (p10)', p10), ('High (p90)', p90)]:
            # Marginal cutoff effect tau(ses, female).
            t_idx = {2: 1.0, 6: sv}
            if fem:
                t_idx.update({10: 1.0, 14: sv})
            tau, tau_se, tau_p = _contrast(t_idx)
            # Baseline turnout just left of the cutoff (T=0, x=0).
            b_idx = {0: 1.0, 4: sv}
            if fem:
                b_idx.update({8: 1.0, 12: sv})
            base, base_se, _ = _contrast(b_idx)
            rows.append({
                'ses_var': ses_var, 'sex': sex, 'ses_level': lvl, 'ses_val': sv,
                'tau': tau, 'tau_se': tau_se, 'tau_p': tau_p,
                'tau_lo': tau - z_a * tau_se, 'tau_hi': tau + z_a * tau_se,
                'baseline': base, 'baseline_se': base_se,
                'baseline_lo': base - z_a * base_se,
                'baseline_hi': base + z_a * base_se,
            })
    return triple, pd.DataFrame(rows)


def plot_triple_cells_grouped(cells_by_ses, figures_dir, value='tau',
                              threshold=18, filename=None):
    """
    Grouped bar chart of the sex × SES cells from `unpack_triple_interaction`.

    `cells_by_ses` is a dict {ses_var: cells DataFrame}. One panel per SES
    measure; bars grouped by deprivation level (Low/High) and coloured by sex,
    with 95% CI whiskers. `value` selects what is plotted:
      - 'tau'      → the marginal CV effect on turnout (pp);
      - 'baseline' → the voluntary-side baseline turnout level (pp).
    """
    os.makedirs(figures_dir, exist_ok=True)
    cF, cM = '#c44e52', '#4c72b0'
    lo_col, hi_col = f'{value}_lo', f'{value}_hi'
    title = {'tau': 'Marginal CV effect (τ) on turnout',
             'baseline': 'Baseline turnout (voluntary side, just below cutoff)'}[value]
    ylab = {'tau': 'RD effect on turnout (pp)',
            'baseline': 'Turnout (pp)'}[value]

    ses_list = list(cells_by_ses.keys())
    fig, axes = plt.subplots(1, len(ses_list),
                             figsize=(6.5 * len(ses_list), 5.5), squeeze=False)
    axes = axes[0]
    levels = ['Low (p10)', 'High (p90)']
    width = 0.36
    xpos = np.arange(len(levels))
    for ax, ses in zip(axes, ses_list):
        c = cells_by_ses[ses]
        for k, (sex, color) in enumerate([('Men', cM), ('Women', cF)]):
            vals, los, his = [], [], []
            for lv in levels:
                row = c[(c['sex'] == sex) & (c['ses_level'] == lv)].iloc[0]
                vals.append(row[value] * 100)
                los.append(row[lo_col] * 100)
                his.append(row[hi_col] * 100)
            vals, los, his = np.array(vals), np.array(los), np.array(his)
            off = (k - 0.5) * width
            yerr = np.vstack([vals - los, his - vals])
            ax.bar(xpos + off, vals, width, yerr=yerr, capsize=4,
                   color=color, alpha=0.9, label=sex)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(['Low\n(least deprived)', 'High\n(most deprived)'])
        ax.set_xlabel(f'{_ses_label(ses)} level')
        ax.set_ylabel(ylab)
        ax.set_title(_ses_label(ses, long=True))
        ax.legend(framealpha=0.9)
    fig.suptitle(f'{title} by sex × deprivation — age-{threshold} cutoff',
                 fontweight='bold')
    plt.tight_layout()
    fname = filename or f'triple_{value}_cells_T{threshold}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved: {figures_dir}/{fname}")

def plot_gender_gap_baseline_endpoint(cells, ses_var, figures_dir, threshold=18,
                                      filename=None):
    """
    Slope chart of the gender turnout gap from voluntary to compulsory voting at
    the age-`threshold` cutoff, by SES level — the "start → end" view behind the
    convergence argument.

    `cells` is the per-cell DataFrame from `unpack_triple_interaction` (columns
    `sex`, `ses_level`, `baseline`, `tau` in probability units). For each SES
    level a sub-panel plots, per sex, the voluntary-side baseline (x=0⁻) and the
    compulsory-side endpoint (baseline + τ, x=0⁺) connected by a line. The F−M
    gap at each end is annotated, and the panel title reports whether the gap
    narrows (convergence) or widens. Returns a tidy DataFrame of the gaps.
    """
    os.makedirs(figures_dir, exist_ok=True)
    cF, cM = '#c44e52', '#4c72b0'
    levels = [lv for lv in ['Low (p10)', 'High (p90)']
              if lv in set(cells['ses_level'])]

    rows = []
    fig, axes = plt.subplots(1, len(levels), figsize=(5.4 * len(levels), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, lvl in zip(axes, levels):
        ends = {}
        for sex, color in [('Women', cF), ('Men', cM)]:
            r = cells[(cells['sex'] == sex) & (cells['ses_level'] == lvl)].iloc[0]
            base = r['baseline'] * 100
            end = (r['baseline'] + r['tau']) * 100
            ends[sex] = (base, end)
            ax.plot([0, 1], [base, end], '-o', color=color, lw=2.2,
                    markersize=8, label=sex)
            # Two decimals, not one: the gap annotated below is differenced from
            # the unrounded levels, so one-decimal labels can subtract to a gap
            # 0.1pp away from it (52.6 - 51.1 = 1.5 against an annotated +1.6).
            ax.annotate(f'{base:.2f}', (0, base), textcoords='offset points',
                        xytext=(-8, 0), ha='right', va='center', fontsize=9, color=color)
            ax.annotate(f'{end:.2f}', (1, end), textcoords='offset points',
                        xytext=(8, 0), ha='left', va='center', fontsize=9, color=color)
        bgap = ends['Women'][0] - ends['Men'][0]
        egap = ends['Women'][1] - ends['Men'][1]
        change = egap - bgap
        # 'converges' asserted a finding the design does not license: the change at
        # p90 is -2.9pp with a CI touching zero, and Sec. 6.3 calls the
        # SES-conditionality descriptive only. The panel reports direction only.
        verdict = 'narrows' if change < 0 else 'widens'
        rows.append({'ses_var': ses_var, 'ses_level': lvl,
                     'base_gap_pp': bgap, 'end_gap_pp': egap, 'change_pp': change})
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Voluntary\n(just <18)', 'Compulsory\n(just ≥18)'])
        ax.set_title(f'{lvl}\nF−M gap {bgap:+.1f} → {egap:+.1f}pp  ({verdict})',
                     fontsize=10)
        ax.set_ylabel('Turnout (%)')
        ax.legend(framealpha=0.9, loc='lower right', fontsize=9)
    fig.suptitle(
        f'Gender turnout gap, voluntary → compulsory by {_ses_label(ses_var)} level '
        f'— age-{threshold} cutoff', fontweight='bold')
    plt.tight_layout()
    fname = filename or f'gender_gap_baseline_endpoint_{ses_var}_T{threshold}.png'
    plt.savefig(f"{figures_dir}/{fname}", bbox_inches='tight', dpi=150)
    plt.show()
    tab = pd.DataFrame(rows)
    print(tab.round(2).to_string(index=False))
    print(f"Saved: {figures_dir}/{fname}")
    return tab
