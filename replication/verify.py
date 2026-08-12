"""Completeness check: every artifact the thesis displays must exist in
replication/outputs/. This is the package's permanent guarantee.

The manuscript's LaTeX sources are the authority on which artifacts those are,
but they are not distributed with this package. So the list is also recorded
here, in MANIFEST, and the check works either way:

  * with the sources present (the author's working repository), it parses them,
    and reports any disagreement with MANIFEST — which means MANIFEST has gone
    stale and needs updating in the same commit as the .tex change;
  * without them (a clone of this package), it checks MANIFEST directly.

Runs on the standard library alone, so a bare clone can verify itself.
"""
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS = os.path.join(_REPO_ROOT, "thesis")

# The 34 artifacts the submitted manuscript displays: 26 figures and 8 tables.
# Regenerate after any change to the .tex, with the sources present:
#   python replication/verify.py --print-manifest
MANIFEST_FIGURES = [
    "bandwidth_sensitivity.png",
    "bandwidth_sensitivity_nbi.png",
    "bandwidth_sensitivity_pca.png",
    "covariate_balance.png",
    "density_heaping_placebo_T18.png",
    "gap_jump_robustness_T18.png",
    "gender_gap_baseline_endpoint_nbi_T18.png",
    "gender_gap_baseline_endpoint_pca_index_T18.png",
    "placebo_cutpoints.png",
    "placebo_cutpoints_nbi.png",
    "placebo_cutpoints_pca.png",
    "rdd_gradient_gap_nbi.png",
    "rdd_gradient_gaprd_vs_decile_pca_index_T18.png",
    "rdd_gradient_marginal_nbi.png",
    "rdd_gradient_marginal_pca_index.png",
    "rdd_gradient_per_sex_nbi.png",
    "rdd_gradient_per_sex_pca_index.png",
    "rdd_threshold_18.png",
    "rdd_threshold_70.png",
    "spatial_join_illustration.png",
    "triple_tau_cells_T70.png",
    "turnout_by_age_gender.png",
    "turnout_vs_nbi_circuits.png",
    "turnout_vs_nbi_gender_lowess.png",
    "turnout_vs_pca_circuits.png",
    "turnout_vs_pca_gender_lowess.png",
]
MANIFEST_TABLES = [
    "covariate_balance_latex.tex",
    "descriptive_stats_latex.tex",
    "gradient_cluster_robustness_T18.tex",
    "gradient_geography_T18.tex",
    "rdd_results_latex.tex",
    "triple_cells_T18.tex",
    "triple_interaction_T18.tex",
    "triple_interaction_T70.tex",
]

# Artifacts the thesis references that are NOT produced by the Stage B chain
# run_all.py executes by default. Emptying replication/outputs/ and re-running
# run_all.py therefore does not restore them, and the bare filename gives no clue
# which script owns them — so name the owner in the failure message.
STAGE_A_OWNED = {
    "spatial_join_illustration.png":
        "00b_spatial_illustration.py (Stage A: needs the restricted census "
        "and circuit layers; run it directly, or via run_all.py --build-data)",
}


def output_dirs():
    """(figures, tables), from config.py if it is present and stdlib-importable,
    otherwise repo-relative — which is what config_template.py resolves to."""
    default = os.path.join(_REPO_ROOT, "replication", "outputs")
    try:
        sys.path.insert(0, _REPO_ROOT)
        from config import REPLICATION_OUTPUTS as base
    except Exception:
        base = default
    return os.path.join(base, "figures"), os.path.join(base, "tables")


def referenced_artifacts():
    """(figures, tables) referenced by main.tex + chapters/*.tex, or None if the
    manuscript sources are not distributed alongside this package."""
    chapters = os.path.join(THESIS, "chapters")
    if not os.path.isdir(chapters):
        return None
    figs, tabs = set(), set()
    tex_files = [os.path.join(THESIS, "main.tex")] + [
        os.path.join(chapters, f)
        for f in sorted(os.listdir(chapters)) if f.endswith(".tex")]
    for path in tex_files:
        src = open(path).read()
        src = re.sub(r"(?<!\\)%.*", "", src)          # strip comments
        figs |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src))
        tabs |= set(re.findall(r"\\input\{\\tabdir/([^}]+)\}", src))
    figs = {os.path.basename(f) for f in figs}
    return sorted(figs), sorted(tabs)


def report_drift(parsed):
    """Print how the manuscript's references differ from MANIFEST. True if any."""
    drift = False
    for kind, found, recorded in (("figures", parsed[0], MANIFEST_FIGURES),
                                  ("tables", parsed[1], MANIFEST_TABLES)):
        for name in sorted(set(found) - set(recorded)):
            print(f"DRIFT: {kind}/{name} is referenced but not in MANIFEST")
            drift = True
        for name in sorted(set(recorded) - set(found)):
            print(f"DRIFT: {kind}/{name} is in MANIFEST but no longer referenced")
            drift = True
    if drift:
        print("       MANIFEST is stale. Refresh it with --print-manifest.")
    return drift


def print_manifest(parsed):
    for var, names in (("MANIFEST_FIGURES", parsed[0]), ("MANIFEST_TABLES", parsed[1])):
        print(f"{var} = [")
        for n in names:
            print(f'    "{n}",')
        print("]")


def main():
    parsed = referenced_artifacts()

    if "--print-manifest" in sys.argv:
        if parsed is None:
            print("--print-manifest needs the manuscript sources in thesis/")
            sys.exit(1)
        print_manifest(parsed)
        return

    if parsed is None:
        figs, tabs = MANIFEST_FIGURES, MANIFEST_TABLES
        source = "MANIFEST"
        drift = False
    else:
        figs, tabs = parsed
        source = "the manuscript sources"
        drift = report_drift(parsed)

    fig_dir, tab_dir = output_dirs()
    missing = [f"figures/{f}" for f in figs
               if not os.path.exists(os.path.join(fig_dir, f))]
    missing += [f"tables/{t}" for t in tabs
                if not os.path.exists(os.path.join(tab_dir, t))]
    total = len(figs) + len(tabs)

    if missing:
        print(f"MISSING: {len(missing)} of {total} artifacts, per {source}")
        for m in missing:
            owner = STAGE_A_OWNED.get(os.path.basename(m))
            print(f"  {m}" + (f"\n      owned by {owner}" if owner else ""))
    elif not drift:
        print(f"OK: all {total} artifacts present in replication/outputs/, "
              f"per {source}")

    sys.exit(1 if (missing or drift) else 0)


if __name__ == "__main__":
    main()
