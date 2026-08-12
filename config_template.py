# config_template.py — copy to config.py (which is gitignored) and adjust if needed.
#
# Project layout:
#   data/raw/        raw inputs (large external sources via symlinks; LAPOP .dta in repo)
#   data/processed/  per-province parquets produced by the pipeline
#   data/final/      consolidated national.parquet
#   outputs/figures/ exported figures
#   outputs/tables/  exported tables
#
# Create the raw symlinks once per machine, e.g.:
#   ln -s /path/to/circuitos_electorales_AR/2025   data/raw/circuitos_2025
#   ln -s /path/to/2025_registro_infractores       data/raw/padron_2025
#   ln -s /path/to/Censos_Argentina/2022           data/raw/censo_2022

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data (external sources reached via symlinks under data/raw/)
GEO_DIR    = os.path.join(ROOT, "data/raw/circuitos_2025/")   # GeoJSON circuit files by province
PADRON_DIR = os.path.join(ROOT, "data/raw/padron_2025/")      # Electoral roll files by province
CENSO_PATH = os.path.join(ROOT, "data/raw/censo_2022/Indicadores de hogares. Radios, 2022.csv")

# Processed data
PROCESSED_DIR = os.path.join(ROOT, "data/processed/")  # per-province parquets
FINAL_DIR     = os.path.join(ROOT, "data/final/")      # consolidated national.parquet

# Outputs (figures and tables only)
FIGURES_DIR = os.path.join(ROOT, "outputs/figures/")
TABLES_DIR  = os.path.join(ROOT, "outputs/tables/")

# Replication package outputs — the ONLY directory the thesis reads from.
REPLICATION_OUTPUTS = os.path.join(ROOT, "replication", "outputs")

# Province boundaries GeoJSON (INDEC codprov_censo codes), external to the repo.
# Used by plot_provincial_turnout (03_analysis.ipynb, 06_provincial.ipynb).
# Replace with the local path to provincias_simplified.geojson on your machine.
GEO_PROVINCIAS = "/path/to/geoAr/data/provincias_simplified.geojson"

# Election date
ELECTION_DATE = "2025-10-26"
