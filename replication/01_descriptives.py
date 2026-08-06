"""01 — Descriptives: full-register figures + descriptive statistics table.
Transcribed from notebooks/03_analysis.ipynb (producing cells only: sections
2, 2b, 4, 5, 6, 7, 9, 9b, 9c, 10, 10b, 11, 11b, 13, 14, 15).
Figures/tables land in replication/outputs/.

Every statistical choice (bins, quantiles, LOWESS fracs, windows) is kept
verbatim from the notebook — only output directories change (FIGURES_DIR /
TABLES_DIR -> REP_FIGURES / REP_TABLES) and the full-register loads are
split into column-scoped passes so the 35.9M-row frame is never held with
all 16 columns at once.
"""
import gc
import os

from _common import setup, load_window, REP_FIGURES, REP_TABLES  # noqa: F401
setup()

import pandas as pd
from config import FINAL_DIR, GEO_PROVINCIAS
from src.pipeline import PROVINCE_CONFIGS
from src.analysis import (
    circuit_stats,
    plot_nbi_distribution, plot_pca_distribution,
    plot_age_distribution, plot_gender_composition,
    plot_turnout_by_age, plot_turnout_by_age_gender,
    plot_turnout_by_nbi_quantile, plot_turnout_by_pca_quantile,
    plot_nbi_vs_pca,
    plot_turnout_vs_nbi_circuits, plot_turnout_vs_nbi_gender,
    plot_turnout_vs_pca_circuits, plot_turnout_vs_pca_gender,
    plot_provincial_turnout, plot_threshold_density,
)

province_names = {pid: cfg['name'] for pid, cfg in PROVINCE_CONFIGS.items()}

if not os.path.exists(GEO_PROVINCIAS):
    raise FileNotFoundError(
        f"Province boundaries GeoJSON not found at GEO_PROVINCIAS={GEO_PROVINCIAS}. "
        "Set GEO_PROVINCIAS in config.py to the local path of "
        "provincias_simplified.geojson (see config_template.py)."
    )

# ── Section 2. NBI Distribution ────────────────────────────────────────────
df = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['circuit_id', 'nbi'])
plot_nbi_distribution(df, REP_FIGURES)
del df
gc.collect()

# ── Section 2b. PCA Deprivation Index Distribution ─────────────────────────
df = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['circuit_id', 'pca_index'])
plot_pca_distribution(df, REP_FIGURES)
del df
gc.collect()

# ── Sections 4-7. Age distribution, gender composition, turnout by age ────
df = pd.read_parquet(f"{FINAL_DIR}/national.parquet",
                     columns=['age_days', 'gender', 'voted'])
df['age'] = df['age_days'] / 365.25

plot_age_distribution(df, REP_FIGURES)
plot_gender_composition(df, REP_FIGURES)
plot_turnout_by_age(df, REP_FIGURES)
plot_turnout_by_age_gender(df, REP_FIGURES)
del df
gc.collect()

# ── Section 9. Turnout by NBI Quantile ─────────────────────────────────────
df = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['nbi', 'voted'])
plot_turnout_by_nbi_quantile(df, REP_FIGURES)
del df
gc.collect()

# ── Section 9b. Turnout by PCA Deprivation Quantile ────────────────────────
df = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['pca_index', 'voted'])
plot_turnout_by_pca_quantile(df, REP_FIGURES)
del df
gc.collect()

# ── Sections 9c, 10, 10b, 11, 11b. Circuit-level NBI/PCA vs turnout ────────
df = pd.read_parquet(
    f"{FINAL_DIR}/national.parquet",
    columns=['province_id', 'circuit_id', 'gender', 'voted', 'nbi', 'pca_index'],
)

df_circuit = circuit_stats(df)
plot_nbi_vs_pca(df_circuit, REP_FIGURES)

print(f"Total circuits: {len(df_circuit):,}")
plot_turnout_vs_nbi_circuits(df_circuit, REP_FIGURES)
plot_turnout_vs_pca_circuits(df_circuit, REP_FIGURES)

df_circuit_gender = (
    df.groupby(["province_id", "circuit_id", "gender"])
    .agg(
        turnout   =("voted",     "mean"),
        nbi       =("nbi",       "first"),
        pca_index =("pca_index", "first"),
        n_voters  =("voted",     "count"),
    )
    .reset_index()
)

plot_turnout_vs_nbi_gender(df_circuit_gender, REP_FIGURES)
plot_turnout_vs_pca_gender(df_circuit_gender, REP_FIGURES)
del df, df_circuit, df_circuit_gender
gc.collect()

# ── Section 13. Descriptive Statistics ─────────────────────────────────────
df = pd.read_parquet(
    f"{FINAL_DIR}/national.parquet",
    columns=['voted', 'gender', 'age_days', 'nbi', 'pca_index',
             'days_from_18', 'days_from_70'],
)
df['age'] = df['age_days'] / 365.25


def _desc(d):
    return pd.Series({
        "N":               len(d),
        "Turnout (%)": d["voted"].mean() * 100,
        "Female (%)": (d["gender"] == "F").mean() * 100,
        "Mean age (years)": d["age"].mean(),
        "NBI (%)": d["nbi"].mean() * 100,
        "PCA index": d["pca_index"].mean(),
    })


rows = {
    "Full padron":             _desc(df),
    "T18 window ($\\pm$365 days)": _desc(df[df["days_from_18"].between(-365, 365)]),
    "T70 window ($\\pm$365 days)": _desc(df[df["days_from_70"].between(-365, 365)]),
}
df_desc = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Sample"})

df_desc["N"]                = df_desc["N"].apply(lambda x: f"{int(x):,}")
df_desc["Turnout (%)"]      = df_desc["Turnout (%)"].apply(lambda x: f"{x:.1f}")
df_desc["Female (%)"]       = df_desc["Female (%)"].apply(lambda x: f"{x:.1f}")
df_desc["Mean age (years)"] = df_desc["Mean age (years)"].apply(lambda x: f"{x:.2f}")
df_desc["NBI (%)"]          = df_desc["NBI (%)"].apply(lambda x: f"{x:.1f}")
df_desc["PCA index"]        = df_desc["PCA index"].apply(lambda x: f"{x:.3f}")

print(df_desc.to_string(index=False))

# Save CSV and LaTeX
df_desc.to_csv(f"{REP_TABLES}/descriptive_stats.csv", index=False)
# escape=False so $\pm$ survives in the window labels; the only other LaTeX
# special in this table is the % in the column headers, escaped here.
df_desc.columns = [c.replace("%", r"\%") for c in df_desc.columns]
latex_str = df_desc.to_latex(
    index=False, escape=False,
    caption="Descriptive Statistics of the Electoral Roll --- Argentina 2025",
    label="tab:descriptive_stats",
    column_format="lrrrrrr"
)
# Wrap in [ht]/centering/resizebox so the 7-column table shrinks to the text
# width instead of running off the right margin (the long "Sample" labels push
# it past \linewidth at 12pt). Same pattern as rdd_results_latex.
latex_str = latex_str.replace("\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n", 1)
latex_str = latex_str.replace(
    "\\begin{tabular}{lrrrrrr}",
    "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{lrrrrrr}", 1)
latex_str = latex_str.replace("\\end{tabular}\n", "\\end{tabular}%\n}\n", 1)
with open(f"{REP_TABLES}/descriptive_stats_latex.tex", "w") as fout:
    fout.write(latex_str)
print("Saved:", f"{REP_TABLES}/descriptive_stats.csv")
print("Saved:", f"{REP_TABLES}/descriptive_stats_latex.tex")
del df
gc.collect()

# ── Section 14. Provincial Turnout Choropleth (Raw) ────────────────────────
df = pd.read_parquet(f"{FINAL_DIR}/national.parquet", columns=['province_id', 'voted'])
_ = plot_provincial_turnout(df, GEO_PROVINCIAS, REP_FIGURES,
                            province_names=province_names)
del df
gc.collect()

# ── Section 15. Voter Density Around the RD Thresholds ─────────────────────
df = load_window([])
plot_threshold_density(df, REP_FIGURES)
del df
gc.collect()

print("01_descriptives: DONE")
