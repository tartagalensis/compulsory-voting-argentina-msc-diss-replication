"""Run the full Stage-B replication (01->07), then verify completeness.

--build-data prepends Stage A: 00 (build the parquet) and 00b (the Appendix A
spatial-join figure). Both require the restricted raw data, which is why 00b is
not in the default chain even though the thesis references its figure.
--fix-date prepends 00c, the one-shot 2025-10-25 -> 2025-10-26 repair of the
stored parquets (already applied; kept for the audit trail and refuses to
double-apply).
--only/--skip take comma-separated script numbers (e.g. --only 02,04).

Each script is a thin driver over src/; heavy rdrobust loops dominate the
runtime (full Stage-B run ~3-4 h of compute). Scripts are independently
resumable, so a --only re-run of a single stage is cheap.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["01_descriptives.py", "02_main_rdd.py", "03_gradient.py",
         "04_gender_gap.py", "05_triple.py", "06_provincial.py",
         "07_robustness.py", "08_geography.py"]
# 08 runs after 05: it reads the estimation window off triple_interaction_T18.csv
# so its unconditional row is the same estimand as the published gradient. It
# also needs the circuit GeoJSON layer, but only to build circuit_geography.csv,
# which is committed — so 08 stays in Stage B and is safe without the raw inputs.

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--build-data", action="store_true",
               help="prepend Stage A (00_build_data.py); needs restricted raw inputs")
p.add_argument("--fix-date", action="store_true",
               help="prepend 00c_fix_election_date.py (one-shot 2025-10-25 -> "
                    "2025-10-26 repair of the stored parquets)")
p.add_argument("--only", default="", help="comma-separated script numbers to run")
p.add_argument("--skip", default="", help="comma-separated script numbers to skip")
a = p.parse_args()

# 00b is Stage A too (it reads the restricted census/circuit layers, not
# national.parquet), so it rides with --build-data and stays out of the default
# chain: Stage B must keep running for someone who has only the parquet.
scripts = (["00_build_data.py", "00b_spatial_illustration.py"] if a.build_data else []) \
    + (["00c_fix_election_date.py"] if a.fix_date else []) + ORDER
if a.only:
    keep = {s.strip().zfill(2) for s in a.only.split(",")}
    scripts = [s for s in scripts if s[:2] in keep]
if a.skip:
    drop = {s.strip().zfill(2) for s in a.skip.split(",")}
    scripts = [s for s in scripts if s[:2] not in drop]

for s in scripts:
    t0 = time.time()
    print(f"=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)])
    print(f"=== {s}: exit {r.returncode} in {time.time() - t0:,.0f}s ===", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

print("=== verify.py ===", flush=True)
sys.exit(subprocess.run([sys.executable, os.path.join(HERE, "verify.py")]).returncode)
