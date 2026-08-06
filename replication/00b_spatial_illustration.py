"""00b — Appendix A illustration: joining two incompatible geographies.

STAGE A. Like 00_build_data.py this reads the restricted raw inputs (the 2022
census tract file and the electoral-circuit layer), not data/final/national.parquet,
so it is NOT part of the Stage B chain that run_all.py executes by default.

What it produces
----------------
`spatial_join_illustration.png` — two panels over La Matanza (Buenos Aires
province), the largest and most socioeconomically heterogeneous district of the
conurbation, chosen as a legible test case:

  left   electoral circuits with census-tract boundaries overlaid. The two
         partitions are produced by different institutions and share no
         boundary system, which is why no identifier join is possible and a
         spatial one is needed.
  right  the tract centroids used to make the assignment, over circuits shaded
         by the resulting circuit-level NBI.

`spatial_join_illustration.csv` — the accompanying statistics: how many tract
centroids were assigned by containment against the nearest-circuit fallback, and
how many tracts straddle a circuit boundary by more than the buffer.

Method notes
------------
NBI is aggregated exactly as `src/pipeline.py:compute_nbi` does it, as
sum(households with at least one unmet need) / sum(households), i.e. weighted by
households. The proof-of-concept this figure descends from weighted by
population instead; that would depict an aggregation the thesis does not use.

Ambiguity is measured by intersecting tract polygons with circuit polygons after
shrinking the tracts by 20 m. The two layers were digitised independently, so
their shared edges carry slivers of overlap that are artefacts of precision
rather than genuine geographic ambiguity; the inward buffer absorbs them.
"""
import json
import os

from _common import setup, REP_FIGURES, REP_TABLES  # noqa: F401
setup()

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from shapely.geometry import shape

from config import CENSO_PATH, GEO_DIR

# La Matanza. The circuit layer carries only electoral codes (codprov/coddepto)
# with no department names, and the electoral department numbering is not the
# census one, so the district is defined from the census side (INDEC 6427) and
# the circuits are those whose centroid falls inside it. That needs no lookup
# table and cannot go stale if either coding changes.
DEPARTMENT = "La Matanza"
DEPT_CODE = 6427
CRS_METRIC = "EPSG:22185"   # POSGAR 2007 / Argentina 5 — centroids and buffers
BUFFER_M = -20              # inward buffer before the ambiguity overlay

fig_path = f"{REP_FIGURES}/spatial_join_illustration.png"
csv_path = f"{REP_TABLES}/spatial_join_illustration.csv"

# ── Census tracts ─────────────────────────────────────────────────────────
censo = pd.read_csv(CENSO_PATH, usecols=[
    "Código de radio",
    "Código de departamentos/comuna",
    "Total de hogares",
    "Hogares con al menos un indicador NBI",
    "Población total",
    "Geometría en GeoJSON",
])
censo = censo[censo["Código de departamentos/comuna"] == DEPT_CODE].copy()
print(f"Census tracts in {DEPARTMENT}: {len(censo)}")

tracts = gpd.GeoDataFrame(
    censo,
    geometry=censo["Geometría en GeoJSON"].apply(lambda g: shape(json.loads(g))),
    crs="EPSG:4326",
)
del censo

# Centroids in a metric CRS: taking them in a geographic CRS distorts the point.
centroids = tracts.copy()
centroids["geometry"] = centroids.to_crs(CRS_METRIC).centroid.to_crs("EPSG:4326")

# ── Circuits: those whose centroid falls inside the census district ───────
district = tracts.to_crs(CRS_METRIC).union_all()
circuits = gpd.read_file(os.path.join(GEO_DIR, "PBA.geojson")).to_crs(CRS_METRIC)
circuits["circuito_key"] = (
    circuits["circuito"].astype(str).str.strip().str.lstrip("0").str.upper()
)
circuits = circuits[circuits.geometry.centroid.within(district)].copy()
circuits = circuits.to_crs("EPSG:4326")
print(f"Circuits in {DEPARTMENT}: {len(circuits)}")

# ── Assignment: containment first, nearest circuit as fallback ────────────
joined = gpd.sjoin(centroids, circuits[["circuito_key", "geometry"]],
                   how="left", predicate="within")
joined["match_type"] = "within"

unmatched = joined[joined["circuito_key"].isna()].drop(
    columns=["index_right", "circuito_key", "match_type"])
if len(unmatched):
    nearest = gpd.sjoin_nearest(
        unmatched.to_crs(CRS_METRIC),
        circuits[["circuito_key", "geometry"]].to_crs(CRS_METRIC),
        how="left").to_crs("EPSG:4326")
    nearest["match_type"] = "nearest"
    joined = pd.concat([joined[joined["circuito_key"].notna()], nearest],
                       ignore_index=True)

n_within = int((joined["match_type"] == "within").sum())
n_nearest = int((joined["match_type"] == "nearest").sum())
print(f"Assigned by containment: {n_within} · by nearest circuit: {n_nearest}")

# ── Circuit-level NBI, household-weighted (as src/pipeline.py does it) ────
by_circuit = joined.groupby("circuito_key", as_index=False).agg(
    nbi_households=("Hogares con al menos un indicador NBI", "sum"),
    households=("Total de hogares", "sum"),
)
by_circuit["nbi_pct"] = 100 * by_circuit["nbi_households"] / by_circuit["households"]

# ── Ambiguity: tracts straddling a circuit boundary past the buffer ───────
tracts_proj = tracts.to_crs(CRS_METRIC)
shrunk = tracts_proj.copy()
shrunk["geometry"] = tracts_proj.geometry.buffer(BUFFER_M)
shrunk = shrunk[~shrunk.geometry.is_empty & shrunk.geometry.notna()]

overlaid = gpd.overlay(
    shrunk[["Código de radio", "geometry"]],
    circuits[["circuito_key", "geometry"]].to_crs(CRS_METRIC),
    how="intersection",
)
n_circuits = (overlaid.groupby("Código de radio")["circuito_key"]
              .nunique().rename("n_circuits").reset_index())
tracts = tracts.merge(n_circuits, on="Código de radio", how="left")
tracts["ambiguous"] = tracts["n_circuits"] > 1

n_amb, n_tot = int(tracts["ambiguous"].sum()), len(tracts)
pop_amb = tracts.loc[tracts["ambiguous"], "Población total"].sum()
pop_tot = tracts["Población total"].sum()
print(f"Straddling tracts: {n_amb}/{n_tot} ({100*n_amb/n_tot:.1f}%) · "
      f"population {pop_amb:,.0f}/{pop_tot:,.0f} ({100*pop_amb/pop_tot:.1f}%)")

pd.DataFrame([{
    "district": DEPARTMENT, "circuits": len(circuits), "tracts": n_tot,
    "assigned_within": n_within, "assigned_nearest": n_nearest,
    "tracts_straddling": n_amb, "pct_tracts_straddling": 100 * n_amb / n_tot,
    "pct_population_straddling": 100 * pop_amb / pop_tot,
    "buffer_m": BUFFER_M,
}]).to_csv(csv_path, index=False)
print("Saved:", csv_path)

# ── Figure ────────────────────────────────────────────────────────────────
# Wide and short on purpose: the binding constraint is the vertical space
# Appendix A leaves on its own page, since the appendix is at its 10-page cap.
circ_plot = circuits.to_crs(CRS_METRIC)
tract_plot = tracts.to_crs(CRS_METRIC)
cent_plot = joined.to_crs(CRS_METRIC)
nbi_plot = circuits.merge(by_circuit[["circuito_key", "nbi_pct"]],
                          on="circuito_key", how="left").to_crs(CRS_METRIC)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

circ_plot.plot(ax=axes[0], facecolor="#fdd0a2", edgecolor="black", linewidth=0.9)
tract_plot.plot(ax=axes[0], facecolor="none", edgecolor="steelblue",
                linewidth=0.35, alpha=0.85)
axes[0].legend(handles=[
    mpatches.Patch(facecolor="#fdd0a2", edgecolor="black", label="Electoral circuit"),
    mpatches.Patch(facecolor="none", edgecolor="steelblue", label="Census tract"),
], loc="lower right", fontsize=7, framealpha=0.9)
axes[0].set_title("The two partitions do not nest", fontsize=9, fontweight="bold")

nbi_plot.plot(ax=axes[1], column="nbi_pct", cmap="RdPu", edgecolor="black",
              linewidth=0.9, legend=True,
              legend_kwds={"label": "Circuit NBI (%), household-weighted",
                           "shrink": 0.85},
              missing_kwds={"color": "lightgrey"})
cent_plot.plot(ax=axes[1], color="steelblue", markersize=1.2, alpha=0.75)
axes[1].set_title("Centroids assigned, then aggregated", fontsize=9, fontweight="bold")

for ax in axes:
    ax.set_axis_off()
plt.tight_layout()
plt.savefig(fig_path, bbox_inches="tight", dpi=200)
print("Saved:", fig_path)
print("\n00b_spatial_illustration: DONE")
