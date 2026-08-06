"""Run the full Stage-B replication (01->07), then verify completeness.

--build-data prepends 00 (Stage A, requires the restricted raw data).
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
         "07_robustness.py"]

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--build-data", action="store_true",
               help="prepend Stage A (00_build_data.py); needs restricted raw inputs")
p.add_argument("--only", default="", help="comma-separated script numbers to run")
p.add_argument("--skip", default="", help="comma-separated script numbers to skip")
a = p.parse_args()

scripts = (["00_build_data.py"] if a.build_data else []) + ORDER
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
