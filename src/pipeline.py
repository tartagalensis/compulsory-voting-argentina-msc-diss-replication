
"""
pipeline.py
-----------
Core pipeline for the electoral-socioeconomic linkage project.

This module implements the full data processing pipeline to assign a
circuit-level NBI (Unsatisfied Basic Needs) indicator to each individual
in Argentina's 2025 electoral roll.

Pipeline steps:
    1. load_circuits         — Load standardized electoral circuit boundaries
    2. load_censo            — Load and filter census data by province
    3. build_censo_gdf       — Build GeoDataFrame and compute tract centroids
    4. spatial_join          — Assign each census tract to an electoral circuit
    5. compute_nbi           — Compute NBI per circuit as sum(NBI households) / sum(total households)
    6. fit_pca_national      — Fit PCA deprivation index on full national census (call once)
    7. compute_circuit_pca   — Apply fitted PCA and aggregate scores to circuit level
    8. load_padron           — Load full electoral roll with RDD variables
    9. join_nbi_padron       — Join circuit-level NBI and PCA index onto individual records
    10. run_province          — Run full pipeline for a single province and save to parquet

Output variables (individual level):
    circuit_id    — Electoral circuit identifier
    province_id   — Electoral province identifier
    age_days      — Age in days as of election day (October 25, 2025)
    days_from_18  — Days from the 18-year compulsory voting threshold
    days_from_70  — Days from the 70-year voluntary voting threshold
    compulsory    — 1 if subject to compulsory voting, 0 otherwise
    voted         — 1 if the individual voted, 0 otherwise
    gender        — Gender (F/M/X)
    nbi           — Circuit-level NBI (% households with at least one unmet basic need)
    pca_index     — Circuit-level PCA deprivation index (PC1, higher = more deprived)

Usage:
    from src.pipeline import run_province, fit_pca_national, PROVINCE_CONFIGS
    pca_transformer = fit_pca_national(CENSO_PATH)
    run_province(province_id=2, output_dir='data/processed', pca_transformer=pca_transformer)
"""


import os
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.append('../..')
from config import GEO_DIR, PADRON_DIR, CENSO_PATH

# ── PROVINCE CONFIGURATION ────────────────────────────────────────────────────
# Maps electoral province ID to file names and INDEC census code

PROVINCE_CONFIGS = {
    1:  {"name": "CABA",       "geo_file": "CABA.geojson",       "padron_file": "G 2025 01 PADINF con fecnac (anonimizado).txt", "indec_code": 2},
    2:  {"name": "PBA",        "geo_file": "PBA.geojson",         "padron_file": "G 2025 02 PADINF con fecnac (anonimizado).txt", "indec_code": 6},
    3:  {"name": "Catamarca",  "geo_file": "Catamarca.geojson",   "padron_file": "G 2025 03 PADINF con fecnac (anonimizado).txt", "indec_code": 10},
    4:  {"name": "Cordoba",    "geo_file": "Cordoba.geojson",     "padron_file": "G 2025 04 PADINF con fecnac (anonimizado).txt", "indec_code": 14},
    5:  {"name": "Corrientes", "geo_file": "Corrientes.geojson",  "padron_file": "G 2025 05 PADINF con fecnac (anonimizado).txt", "indec_code": 18},
    6:  {"name": "Chaco",      "geo_file": "Chaco.geojson",       "padron_file": "G 2025 06 PADINF con fecnac (anonimizado).txt", "indec_code": 22},
    7:  {"name": "Chubut",     "geo_file": "Chubut.geojson",      "padron_file": "G 2025 07 PADINF con fecnac (anonimizado).txt", "indec_code": 26},
    8:  {"name": "EntreRios",  "geo_file": "EntreRios.geojson",   "padron_file": "G 2025 08 PADINF con fecnac (anonimizado).txt", "indec_code": 30},
    9:  {"name": "Formosa",    "geo_file": "Formosa.geojson",     "padron_file": "G 2025 09 PADINF con fecnac (anonimizado).txt", "indec_code": 34},
    10: {"name": "Jujuy",      "geo_file": "Jujuy.geojson",       "padron_file": "G 2025 10 PADINF con fecnac (anonimizado).txt", "indec_code": 38},
    11: {"name": "LaPampa",    "geo_file": "LaPampa.geojson",     "padron_file": "G 2025 11 PADINF con fecnac (anonimizado).txt", "indec_code": 42},
    12: {"name": "LaRioja",    "geo_file": "LaRioja.geojson",     "padron_file": "G 2025 12 PADINF con fecnac (anonimizado).txt", "indec_code": 46},
    13: {"name": "Mendoza",    "geo_file": "Mendoza.geojson",     "padron_file": "G 2025 13 PADINF con fecnac (anonimizado).txt", "indec_code": 50},
    14: {"name": "Misiones",   "geo_file": "Misiones.geojson",    "padron_file": "G 2025 14 PADINF con fecnac (anonimizado).txt", "indec_code": 54},
    15: {"name": "Neuquen",    "geo_file": "Neuquen.geojson",     "padron_file": "G 2025 15 PADINF con fecnac (anonimizado).txt", "indec_code": 58},
    16: {"name": "RioNegro",   "geo_file": "RioNegro.geojson",    "padron_file": "G 2025 16 PADINF con fecnac (anonimizado).txt", "indec_code": 62},
    17: {"name": "Salta",      "geo_file": "Salta.geojson",       "padron_file": "G 2025 17 PADINF con fecnac (anonimizado).txt", "indec_code": 66},
    18: {"name": "SanJuan",    "geo_file": "SanJuan.geojson",     "padron_file": "G 2025 18 PADINF con fecnac (anonimizado).txt", "indec_code": 70},
    19: {"name": "SanLuis",    "geo_file": "SanLuis.geojson",     "padron_file": "G 2025 19 PADINF con fecnac (anonimizado).txt", "indec_code": 74},
    20: {"name": "SantaCruz",  "geo_file": "SantaCruz.geojson",   "padron_file": "G 2025 20 PADINF con fecnac (anonimizado).txt", "indec_code": 78},
    21: {"name": "SantaFe",    "geo_file": "SantaFe.geojson",     "padron_file": "G 2025 21 PADINF con fecnac (anonimizado).txt", "indec_code": 82},
    22: {"name": "Santiago",   "geo_file": "Santiago.geojson",    "padron_file": "G 2025 22 PADINF con fecnac (anonimizado).txt", "indec_code": 86},
    23: {"name": "Tucuman",    "geo_file": "Tucuman.geojson",     "padron_file": "G 2025 23 PADINF con fecnac (anonimizado).txt", "indec_code": 90},
    24: {"name": "TdF",        "geo_file": "TdF.geojson",         "padron_file": "G 2025 24 PADINF con fecnac (anonimizado).txt", "indec_code": 94},
}

ELECTION_DATE = pd.Timestamp('2025-10-25')

# Census column names for PCA deprivation features (from INDEC Censo 2022)
_DEPRIVATION_COLS = {
    'jefe_sec_completa': 'Hogares con jefes con secundaria completa o más',
    'con_computadora':   'Hogares con computadora',
    'hacinamiento':      'Hogares con hacinamiento (más de 2 personas por cuarto)',
    'sin_cloaca':        'Hogares sin cloaca',
    'sin_piso_tipo1':    'Hogares sin piso tipo 1 (cerámica, baldosa, mosaico, mármol, madera, alfombrado)',
}

_DEPRIVATION_FEATURES = [
    'pct_sin_secundaria', 'pct_sin_computadora',
    'pct_hacinamiento',   'pct_sin_cloaca', 'pct_sin_piso_tipo1',
]

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_circuits(province_id):
    """Load standardized electoral circuit boundaries for a province."""
    config = PROVINCE_CONFIGS[province_id]
    gdf = gpd.read_file(GEO_DIR + config["geo_file"])
    gdf['circuito_key'] = gdf['circuito'].astype(str).str.strip().str.lstrip('0').str.upper()
    return gdf


def load_censo(province_id):
    """Load and filter census data for a province."""
    config = PROVINCE_CONFIGS[province_id]
    indec_code = config["indec_code"]

    df = pd.read_csv(CENSO_PATH, usecols=[
        'Código de radio',
        'Código de departamentos/comuna',
        'Total de hogares',
        'Hogares con al menos un indicador NBI',
        'Población total',
        'Geometría en GeoJSON',
        *_DEPRIVATION_COLS.values(),
    ])

    prov_codes = [c for c in df['Código de departamentos/comuna'].unique()
                  if str(c).startswith(str(indec_code))]
    df = df[df['Código de departamentos/comuna'].isin(prov_codes)].copy()

    return df


def build_censo_gdf(df_censo):
    """Build GeoDataFrame from census data using official polygon geometry.
    Computes centroids in metric CRS to avoid distortion."""
    gdf = gpd.GeoDataFrame(
        df_censo,
        geometry=df_censo['Geometría en GeoJSON'].apply(lambda x: shape(json.loads(x))),
        crs='EPSG:4326'
    )
    gdf['geometry'] = gdf.to_crs('EPSG:22185').centroid.to_crs('EPSG:4326')
    return gdf


def spatial_join(censo_gdf, circuits_gdf):
    """Assign each census tract centroid to an electoral circuit.
    Primary: centroid falls within circuit polygon.
    Fallback: nearest circuit for unmatched centroids."""
    joined = gpd.sjoin(
        censo_gdf,
        circuits_gdf[['circuito_key', 'geometry']],
        how='left', predicate='within'
    )

    unmatched = joined[joined['circuito_key'].isna()].drop(
        columns=['index_right', 'circuito_key']
    )

    if len(unmatched) > 0:
        unmatched_proj = unmatched.to_crs('EPSG:22185')
        circuits_proj  = circuits_gdf[['circuito_key', 'geometry']].to_crs('EPSG:22185')
        nearest = gpd.sjoin_nearest(unmatched_proj, circuits_proj, how='left').to_crs('EPSG:4326')
        joined  = pd.concat([joined[joined['circuito_key'].notna()], nearest], ignore_index=True)

    return joined


def compute_nbi(joined):
    """Compute NBI per electoral circuit as:
    sum(households with NBI) / sum(total households)"""
    df_nbi = (
        joined
        .groupby('circuito_key', as_index=False)
        .agg(
            total_nbi_households=('Hogares con al menos un indicador NBI', 'sum'),
            total_households    =('Total de hogares',                      'sum')
        )
    )
    df_nbi['nbi']        = df_nbi['total_nbi_households'] / df_nbi['total_households']
    df_nbi['parent_key'] = df_nbi['circuito_key'].str.extract(r'^(\d+)')

    return df_nbi[['circuito_key', 'parent_key', 'nbi']]


def _build_deprivation_features(df):
    """Compute the 5 deprivation proportions from raw census counts.
    All variables oriented so higher value = more deprived."""
    t = df['Total de hogares'].replace(0, np.nan)
    df = df.copy()
    df['pct_sin_secundaria']  = 1 - df[_DEPRIVATION_COLS['jefe_sec_completa']] / t
    df['pct_sin_computadora'] = 1 - df[_DEPRIVATION_COLS['con_computadora']]   / t
    df['pct_hacinamiento']    =     df[_DEPRIVATION_COLS['hacinamiento']]       / t
    df['pct_sin_cloaca']      =     df[_DEPRIVATION_COLS['sin_cloaca']]         / t
    df['pct_sin_piso_tipo1']  =     df[_DEPRIVATION_COLS['sin_piso_tipo1']]     / t
    return df


def fit_pca_national(censo_path):
    """Fit PCA deprivation index on the full national census (all provinces).

    StandardScaler is applied before PCA so each of the 5 dimensions contributes
    equally to PC1, regardless of its variance across Argentina. Without it,
    pct_sin_cloaca (highest variance) would dominate and the index would not be
    genuinely multidimensional. See Filmer & Pritchett (2001).

    Scores are shifted by the national radio-level minimum so the index is
    anchored at 0 with a natural positive range.

    Must be called once before the province loop for cross-province comparability.
    Returns a dict with: scaler, pca, score_min, loadings, variance_explained.
    """
    df = pd.read_csv(censo_path, usecols=[
        'Total de hogares',
        *_DEPRIVATION_COLS.values(),
    ])
    df = _build_deprivation_features(df)
    X  = df[_DEPRIVATION_FEATURES].dropna()

    scaler = StandardScaler()
    pca    = PCA(n_components=1)
    X_scaled = scaler.fit_transform(X)
    pca.fit(X_scaled)

    # Ensure higher score = more deprived (all loadings should be positive)
    if pca.components_[0].mean() < 0:
        pca.components_ *= -1

    # Shift scores so national minimum = 0
    national_scores = pca.transform(X_scaled)[:, 0]
    score_min       = national_scores.min()

    loadings = dict(zip(_DEPRIVATION_FEATURES, pca.components_[0]))

    print("── PCA Deprivation Index (national fit) ──")
    print(f"  Variance explained by PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  Score range (national):    {score_min:.3f} → {national_scores.max():.3f}  (pre-shift)")
    print(f"  Score range (shifted):     0.000 → {national_scores.max() - score_min:.3f}")
    print("  Loadings:")
    for var, loading in loadings.items():
        print(f"    {var}: {loading:+.3f}")

    return {
        'scaler':             scaler,
        'pca':                pca,
        'score_min':          score_min,
        'loadings':           loadings,
        'variance_explained': pca.explained_variance_ratio_[0],
    }


def compute_circuit_pca(joined, pca_transformer):
    """Apply fitted PCA to census tracts and aggregate to electoral circuit level.

    Produces circuit-level weighted means (by total_hogares) for:
      - pca_index  : PC1 deprivation score
      - pct_sin_secundaria, pct_sin_computadora, pct_hacinamiento,
        pct_sin_cloaca, pct_sin_piso_tipo1

    Fallback: subcircuits inherit parent circuit values.
    """
    scaler    = pca_transformer['scaler']
    pca       = pca_transformer['pca']
    score_min = pca_transformer['score_min']

    d = _build_deprivation_features(joined.copy())
    X = d[_DEPRIVATION_FEATURES]
    valid = X.notna().all(axis=1)

    scores = np.full(len(d), np.nan)
    if valid.any():
        scores[valid.values] = pca.transform(scaler.transform(X[valid]))[:, 0] - score_min
    d['pca_score'] = scores
    d['_w']        = d['Total de hogares'].fillna(0)

    output_cols = ['pca_index'] + _DEPRIVATION_FEATURES

    def _wavg(g):
        """Compute household-weighted mean of pca_score and all 5 deprivation features."""
        w = g['_w']
        row = {}
        row['pca_index'] = np.average(g['pca_score'],  weights=w) if w.sum() > 0 and g['pca_score'].notna().any()  else np.nan
        for feat in _DEPRIVATION_FEATURES:
            row[feat] = np.average(g[feat], weights=w) if w.sum() > 0 and g[feat].notna().any() else np.nan
        return pd.Series(row)

    df_pca = (
        d.groupby('circuito_key')
        .apply(_wavg)
        .reset_index()
    )
    df_pca['parent_key'] = df_pca['circuito_key'].str.extract(r'^(\d+)')

    return df_pca[['circuito_key', 'parent_key'] + output_cols]


def load_padron(province_id):
    """Load full electoral roll for a province and compute RDD running variables.

    Parameters
    ----------
    province_id : int
        Electoral province identifier (1–24).

    Returns
    -------
    DataFrame with columns: circuit_id, circuito_key, parent_key, province_id,
    age, age_days, days_from_18, days_from_70, compulsory, voted, gender.

    days_from_18 / days_from_70 are signed distances from each threshold on
    election day (Oct 25, 2025), computed via relativedelta to handle leap years.
    Positive = past threshold (e.g. already turned 18).
    """
    config = PROVINCE_CONFIGS[province_id]

    df = pd.read_csv(
        PADRON_DIR + config["padron_file"],
        sep="|", dtype=str, encoding="utf-8",
        on_bad_lines="skip", engine="python",
        usecols=['TX_CIRC_NUMERO', 'FC_FECHA_NACIMIENTO', 'ESTADO', 'TX_GENERO']
    )

    df['FC_FECHA_NACIMIENTO'] = pd.to_datetime(
        df['FC_FECHA_NACIMIENTO'], dayfirst=True, errors='coerce'
    )
    
    df['FC_FECHA_NACIMIENTO'] = pd.to_datetime(
        df['FC_FECHA_NACIMIENTO'], dayfirst=True, errors='coerce'
    )

    df['age_days']     = (ELECTION_DATE - df['FC_FECHA_NACIMIENTO']).dt.days

    # Exact dates when each person turns 18 and 70 (accounts for leap years)
    df['date_18'] = df['FC_FECHA_NACIMIENTO'].apply(lambda x: x + relativedelta(years=18) if pd.notna(x) else pd.NaT)
    df['date_70'] = df['FC_FECHA_NACIMIENTO'].apply(lambda x: x + relativedelta(years=70) if pd.notna(x) else pd.NaT)

    df['days_from_18'] = (ELECTION_DATE - df['date_18']).dt.days
    df['days_from_70'] = (ELECTION_DATE - df['date_70']).dt.days

    df['age']        = df['age_days'] / 365.25
    df['compulsory'] = ((df['days_from_18'] >= 0) & (df['days_from_70'] < 0)).astype(int)
    df['voted']        = (df['ESTADO'].str.strip() == 'VOTO').astype(int)

    df = df.rename(columns={'TX_CIRC_NUMERO': 'circuit_id', 'TX_GENERO': 'gender'})
    df['circuito_key'] = df['circuit_id'].str.strip().str.lstrip('0').str.upper()
    df['parent_key']   = df['circuito_key'].str.extract(r'^(\d+)')
    df['province_id']  = province_id

    return df[['circuit_id', 'circuito_key', 'parent_key', 'province_id',
               'age', 'age_days', 'days_from_18', 'days_from_70', 'compulsory',
               'voted', 'gender']]



def join_nbi_padron(df_padron, df_nbi, df_pca=None):
    """Join circuit-level NBI (and optionally PCA variables) onto individual-level electoral roll.

    Three-tier fallback for unmatched circuits:
      1. Direct circuito_key match
      2. Parent circuit (numeric prefix) match — subcircuits inherit parent value
      3. Province mean — last resort for circuits with no census coverage

    Parameters
    ----------
    df_padron : DataFrame
        Individual-level electoral roll (output of load_padron).
    df_nbi : DataFrame
        Circuit-level NBI table (output of compute_nbi).
    df_pca : DataFrame or None
        Circuit-level PCA table (output of compute_circuit_pca). If None, PCA
        columns are omitted from the output.
    """
    df = df_padron.merge(
        df_nbi[['circuito_key', 'nbi']],
        on='circuito_key', how='left'
    )

    parent_nbi_map = (
        df_nbi.dropna(subset=['parent_key'])
        .drop_duplicates('parent_key')
        .set_index('parent_key')['nbi']
    )
    mask = df['nbi'].isna()
    df.loc[mask, 'nbi'] = df.loc[mask, 'parent_key'].map(parent_nbi_map)
    # Last resort: impute with province mean
    mask = df['nbi'].isna()
    df.loc[mask, 'nbi'] = df.loc[~mask, 'nbi'].mean()

    if df_pca is not None:
        pca_cols = ['pca_index'] + _DEPRIVATION_FEATURES
        df = df.merge(df_pca[['circuito_key'] + pca_cols], on='circuito_key', how='left')
        for col in pca_cols:
            parent_map = (
                df_pca.dropna(subset=['parent_key'])
                .drop_duplicates('parent_key')
                .set_index('parent_key')[col]
            )
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, 'parent_key'].map(parent_map)
            # Last resort: impute with province mean (affects circuits with no census coverage)
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[~mask, col].mean()

    return df


def run_province(province_id, output_dir, pca_transformer=None):
    """Run the full pipeline for a single province.
    Saves output to parquet and frees memory.

    Pass pca_transformer (from fit_pca_national) to include the PCA deprivation
    index and its 5 component variables in the output.
    """
    config = PROVINCE_CONFIGS[province_id]
    name   = config["name"]

    print(f"\n── {name} ──")

    circuits_gdf = load_circuits(province_id)
    print(f"Circuits loaded: {len(circuits_gdf)}")

    df_censo  = load_censo(province_id)
    censo_gdf = build_censo_gdf(df_censo)
    print(f"Census tracts loaded: {len(censo_gdf)}")
    del df_censo

    joined = spatial_join(censo_gdf, circuits_gdf)
    del censo_gdf, circuits_gdf

    df_nbi = compute_nbi(joined)

    df_pca = None
    if pca_transformer is not None:
        df_pca = compute_circuit_pca(joined, pca_transformer)
        print(f"PCA index computed: {df_pca['pca_index'].notna().sum()} circuits")
    del joined

    df_padron = load_padron(province_id)
    print(f"Padron loaded: {len(df_padron):,}")

    df_out  = join_nbi_padron(df_padron, df_nbi, df_pca)
    del df_padron, df_nbi, df_pca

    assigned = df_out['nbi'].notna().sum()
    print(f"NBI assigned: {assigned:,} / {len(df_out):,}")
    if pca_transformer is not None:
        print(f"PCA assigned: {df_out['pca_index'].notna().sum():,} / {len(df_out):,}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/province_{province_id:02d}.parquet"
    df_out.drop(columns=['circuito_key', 'parent_key']).to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    del df_out