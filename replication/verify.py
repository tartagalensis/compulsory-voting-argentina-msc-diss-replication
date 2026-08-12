"""Completeness check: every artifact the thesis references must exist in
replication/outputs/. This is the package's permanent guarantee — see the
spec (docs/superpowers/specs/2026-07-21-replication-package-design.md)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replication._common import REP_FIGURES, REP_TABLES, _REPO_ROOT

THESIS = os.path.join(_REPO_ROOT, "thesis")


def referenced_artifacts():
    """(figures, tables) referenced by main.tex + chapters/*.tex."""
    figs, tabs = set(), set()
    tex_files = [os.path.join(THESIS, "main.tex")] + [
        os.path.join(THESIS, "chapters", f)
        for f in sorted(os.listdir(os.path.join(THESIS, "chapters")))
        if f.endswith(".tex")]
    for path in tex_files:
        src = open(path).read()
        src = re.sub(r"(?<!\\)%.*", "", src)          # strip comments
        figs |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src))
        tabs |= set(re.findall(r"\\input\{\\tabdir/([^}]+)\}", src))
    figs = {os.path.basename(f) for f in figs}
    return sorted(figs), sorted(tabs)


# Artifacts the thesis references that are NOT produced by the Stage B chain
# run_all.py executes by default. Emptying replication/outputs/ and re-running
# run_all.py therefore does not restore them, and the bare filename gives no clue
# which script owns them — so name the owner in the failure message.
STAGE_A_OWNED = {
    "spatial_join_illustration.png":
        "00b_spatial_illustration.py (Stage A: needs the restricted census "
        "and circuit layers; run it directly, or via run_all.py --build-data)",
}


def main():
    figs, tabs = referenced_artifacts()
    missing = [f"figures/{f}" for f in figs
               if not os.path.exists(os.path.join(REP_FIGURES, f))]
    missing += [f"tables/{t}" for t in tabs
                if not os.path.exists(os.path.join(REP_TABLES, t))]
    total = len(figs) + len(tabs)
    if missing:
        print(f"MISSING: {len(missing)} of {total} referenced artifacts")
        for m in missing:
            owner = STAGE_A_OWNED.get(os.path.basename(m))
            print(f"  {m}" + (f"\n      owned by {owner}" if owner else ""))
        sys.exit(1)
    print(f"OK: all {total} referenced artifacts present in replication/outputs/")


if __name__ == "__main__":
    main()
