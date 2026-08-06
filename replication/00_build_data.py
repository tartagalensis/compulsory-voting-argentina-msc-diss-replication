"""00 — Build data (Stage A): raw electoral roll + census + circuit boundaries
-> data/processed/*.parquet -> data/final/national.parquet.

Transcribed from notebooks/01_pipeline.ipynb (PCA fit + province loop) and
notebooks/02_consolidation.ipynb (consolidation + validation — no silent
drops). All heavy logic lives in src/pipeline.py; this script is only the
orchestration layer, following the CLAUDE.md rule of fitting the PCA
deprivation index once (fit_pca_national) BEFORE the province loop and
passing the same pca_transformer into every run_province call.

WARNING — this script has been syntax/dry-run validated but NEVER executed
end-to-end. It requires the restricted raw inputs (electoral roll, census,
circuit boundaries) reached via the symlinks under data/raw/ (see config.py /
config_template.py) that are not distributed with this repository. The
published data/final/national.parquet is the validated input consumed by
Stage B (replication/0N_*.py); rebuilding it from raw sources is out of
scope for the replication package and takes hours over the full 35.9M-row
national electoral roll.

Usage:
    python replication/00_build_data.py [--provinces 1,2,3] [--dry-run]

    --provinces   comma-separated electoral province IDs to process
                  (default: all 24, e.g. '1,2,3')
    --dry-run     resolve and print every raw input path plus whether it
                  exists on this machine, then exit 0 WITHOUT building
                  anything
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from config import GEO_DIR, PADRON_DIR, CENSO_PATH, PROCESSED_DIR, FINAL_DIR  # noqa: E402
from src.pipeline import PROVINCE_CONFIGS  # noqa: E402


def parse_args():
    """Parse CLI args. --help is handled by argparse and exits 0."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage A: build data/final/national.parquet from raw electoral "
            "roll, census, and circuit-boundary inputs (padron/censo/circuits "
            "symlinks under data/raw/)."
        )
    )
    parser.add_argument(
        "--provinces", default=None,
        help=(
            "Comma-separated electoral province IDs to process "
            "(default: all 24, e.g. '1,2,3')."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve and print every raw input path plus whether it exists "
            "on this machine, then exit without building anything."
        ),
    )
    return parser.parse_args()


def resolve_province_ids(provinces_arg):
    """Turn the --provinces CLI value into a validated list of province IDs."""
    if provinces_arg is None:
        return list(PROVINCE_CONFIGS.keys())
    ids = [int(p.strip()) for p in provinces_arg.split(",") if p.strip()]
    unknown = [p for p in ids if p not in PROVINCE_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown province id(s): {unknown}. "
            f"Valid ids: {sorted(PROVINCE_CONFIGS.keys())}"
        )
    return ids


def _check_path(path, label):
    """Print whether `path` exists, prefixed by `label`. Returns the bool."""
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {label}: {path}")
    return exists


def dry_run(province_ids):
    """Resolve and print every raw input path + existence, then exit 0
    without touching data/processed or data/final. Missing raw symlinks are
    expected on machines without access to the restricted raw data — this
    is reported clearly, not treated as an error."""
    print("=" * 70)
    print("DRY RUN — resolving Stage A input paths (nothing will be built)")
    print("=" * 70)

    all_ok = True

    print("\nNational census (fit_pca_national, called once):")
    all_ok &= _check_path(CENSO_PATH, "CENSO_PATH")

    print(f"\nPer-province inputs ({len(province_ids)} province(s) selected):")
    for pid in province_ids:
        cfg = PROVINCE_CONFIGS[pid]
        geo_path = os.path.join(GEO_DIR, cfg["geo_file"])
        padron_path = os.path.join(PADRON_DIR, cfg["padron_file"])
        print(f"  Province {pid:02d} ({cfg['name']}):")
        all_ok &= _check_path(geo_path, "    circuits geojson (GEO_DIR)")
        all_ok &= _check_path(padron_path, "    padron file (PADRON_DIR)")

    print("\nOutput directories (config-relative, created on build):")
    print(f"  PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"  FINAL_DIR:     {FINAL_DIR}")

    print("\n" + "=" * 70)
    if all_ok:
        print("All raw inputs resolved and present on this machine.")
    else:
        print(
            "Some raw inputs are MISSING on this machine. This is EXPECTED "
            "if the restricted raw-data symlinks (circuitos_2025, "
            "padron_2025, censo_2022 — see config.py / config_template.py) "
            "have not been set up here: Stage A cannot build without them, "
            "but this is not an error condition for --dry-run."
        )
    print("Dry run complete — exiting without building.")
    return 0


def build(province_ids):
    """Run Stage A end-to-end for `province_ids`: fit the national PCA
    deprivation index once, run the per-province pipeline
    (notebooks/01_pipeline.ipynb), then consolidate + validate
    (notebooks/02_consolidation.ipynb) into data/final/national.parquet."""
    import pandas as pd
    from src.pipeline import run_province, fit_pca_national

    if len(province_ids) < len(PROVINCE_CONFIGS):
        print(
            f"NOTE: running a {len(province_ids)}/{len(PROVINCE_CONFIGS)} "
            "province subset — data/final/national.parquet will only "
            "contain these provinces. Run with the default (all provinces) "
            "to rebuild the full national dataset."
        )

    # ── notebooks/01_pipeline.ipynb cell 3: fit PCA once, before the loop ──
    print("=" * 70)
    print("STAGE A — STEP 1: fit_pca_national (once, before the province loop)")
    print("=" * 70)
    pca_transformer = fit_pca_national(CENSO_PATH)

    # ── notebooks/01_pipeline.ipynb cell 4: per-province loop ──
    print("\n" + "=" * 70)
    print("STAGE A — STEP 2: per-province pipeline (run_province)")
    print("=" * 70)
    failed = []
    for province_id in province_ids:
        try:
            run_province(province_id, output_dir=PROCESSED_DIR, pca_transformer=pca_transformer)
        except Exception as e:
            print(f"Province {province_id} failed: {e}")
            failed.append(province_id)

    print("\n" + "=" * 50)
    print("PROVINCE LOOP COMPLETE")
    print("=" * 50)
    print(f"Provinces processed: {len(province_ids) - len(failed)} / {len(province_ids)}")
    if failed:
        print(f"Failed provinces: {failed}")
        raise RuntimeError(f"{len(failed)} province(s) failed during Stage A build: {failed}")

    # ── notebooks/02_consolidation.ipynb cell 3: consolidate ──
    print("\n" + "=" * 70)
    print("STAGE A — STEP 3: consolidation (notebooks/02_consolidation.ipynb)")
    print("=" * 70)
    dfs = []
    for province_id in province_ids:
        path = f"{PROCESSED_DIR}/province_{province_id:02d}.parquet"
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        dfs.append(pd.read_parquet(path))

    if not dfs:
        raise RuntimeError("No per-province parquet files found to consolidate.")

    df_national = pd.concat(dfs, ignore_index=True)
    del dfs

    print(f"Total records: {len(df_national):,}")
    print(f"Provinces:     {df_national['province_id'].nunique()}")
    print("\nRecords per province:")
    print(df_national.groupby('province_id').size().reset_index(name='n').to_string(index=False))

    # ── notebooks/02_consolidation.ipynb cell 5: validation — no silent drops ──
    validate_cols = ["nbi", "pca_index"] + [
        "pct_sin_secundaria", "pct_sin_computadora", "pct_hacinamiento",
        "pct_sin_cloaca", "pct_sin_piso_tipo1",
    ]
    print("\nMissing value validation:")
    print("-" * 40)
    all_ok = True
    for col in validate_cols:
        n_missing = df_national[col].isna().sum()
        status = "OK" if n_missing == 0 else f"FAIL - {n_missing:,} missing"
        print(f"  {col:<25} {status}")
        if n_missing > 0:
            all_ok = False

    if not all_ok:
        raise ValueError(
            "Missing values detected in the consolidated frame — re-run the "
            "province build before proceeding (see notebooks/01_pipeline.ipynb)."
        )
    print("All columns complete. Proceeding to save.")

    # ── notebooks/02_consolidation.ipynb cell 7: save ──
    os.makedirs(FINAL_DIR, exist_ok=True)
    df_national.to_parquet(f"{FINAL_DIR}/national.parquet", index=False)

    print(f"\nRecords: {len(df_national):,}")
    print(f"Columns: {df_national.columns.tolist()}")
    print(f"Saved: {FINAL_DIR}/national.parquet")
    print()
    print(df_national.describe().round(4).to_string())


def main():
    args = parse_args()
    province_ids = resolve_province_ids(args.provinces)

    if args.dry_run:
        return dry_run(province_ids)

    build(province_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
