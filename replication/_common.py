"""Shared setup for the replication scripts.

Import and call setup() FIRST in every script; it must run before pyplot
is imported anywhere else.
"""
import os
import sys
import warnings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')

from config import FINAL_DIR, REPLICATION_OUTPUTS  # noqa: E402

REP_FIGURES = os.path.join(REPLICATION_OUTPUTS, "figures")
REP_TABLES = os.path.join(REPLICATION_OUTPUTS, "tables")


def setup():
    """Silence rdrobust mass-points noise and ensure output dirs exist."""
    warnings.filterwarnings('ignore')
    os.makedirs(REP_FIGURES, exist_ok=True)
    os.makedirs(REP_TABLES, exist_ok=True)


def load_window(columns, thresholds=(18, 70), bw=365 * 3):
    """Load national.parquet restricted to the union of the ±bw windows of
    the given thresholds, with only `columns` (+ the running variables and
    the cluster keys), and a circuit cluster_id.

    Matches 04_rdd cell 3: cluster_id = ngroup over (province_id, circuit_id);
    the ids differ from the notebook's full-frame ids but the partition is
    identical inside any estimation window, which is all clustering needs.
    """
    import pandas as pd
    need = set(columns) | {'province_id', 'circuit_id'} | \
        {f'days_from_{t}' for t in thresholds}
    dnf = [[(f'days_from_{t}', '>=', -bw), (f'days_from_{t}', '<=', bw)]
           for t in thresholds]
    df = pd.read_parquet(f"{FINAL_DIR}/national.parquet",
                         columns=sorted(need), filters=dnf)
    df['cluster_id'] = df.groupby(['province_id', 'circuit_id']).ngroup()
    return df
