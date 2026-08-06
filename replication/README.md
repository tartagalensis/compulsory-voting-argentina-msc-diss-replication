# Replication package

Code that reproduces every figure and table in the dissertation *Who Votes
Because They Have To?* (heterogeneous effects of compulsory voting on turnout,
Argentina 2025; sharp RDD at the age-18 and age-70 thresholds).

The thesis reads its figures and tables **only** from `replication/outputs/`,
which is written **only** by the numbered scripts here. The exploratory
`notebooks/` are not part of replication; the `src/` package holds the heavy
logic that both call.

## Requirements

- Conda environment from `../environment.yml` (`thesis_lse_env`, or the working
  `franco_env`). Key packages: `pandas`, `pyarrow`, `rdrobust`, `rddensity`,
  `geopandas`, `scikit-learn`, `statsmodels`, `matplotlib`.
- Run scripts with the environment's interpreter, e.g.
  `/opt/anaconda3/envs/franco_env/bin/python replication/run_all.py`.
- Paths come from `../config.py` (repo-relative, not tracked by git — copy
  `../config_template.py` to `../config.py` and fill in the local paths).

## Data: two stages

- **Stage A (`00_build_data.py`, optional, restricted data).** Builds
  `data/final/national.parquet` from the raw padrón, INDEC census, and circuit
  geometries (reached via symlinks under `data/raw/`; not distributable). Thin
  driver over `src/pipeline.py`: fits the national PCA deprivation index once,
  then runs the per-province spatial join + NBI/PCA assignment, then
  consolidates with validation. `--dry-run` resolves and reports the input
  paths without building; `--provinces` filters. **You do not need this to
  replicate the analysis** — it documents how the input was constructed.
- **Stage B (`01`–`07`, the replication proper).** Takes
  `data/final/national.parquet` (35.9M voters × 16 columns) and produces every
  thesis artifact. This is what `run_all.py` runs by default.

## Run order

```
python replication/run_all.py                 # Stage B: 01->07, then verify
python replication/run_all.py --build-data     # prepend Stage A
python replication/run_all.py --only 02,04     # re-run selected stages
python replication/run_all.py --skip 07        # skip a stage
python replication/verify.py                   # completeness check only
```

`verify.py` parses the thesis `.tex` and confirms every referenced figure and
table exists in `replication/outputs/`. A clean run ends with
`OK: all 47 referenced artifacts present`.

## Script -> outputs

| Script | Produces (thesis artifacts) | Approx. runtime |
|---|---|---|
| `01_descriptives.py` | Table 1 (`descriptive_stats_latex`) + 15 descriptive figures (age/gender/NBI/PCA distributions, turnout-vs-SES, provincial turnout map, threshold density) | ~5 min |
| `02_main_rdd.py` | Table 2 (`rdd_results_latex`, 30 rows, all circuit-clustered), main RD figures, covariate balance, density test | ~50 min |
| `03_gradient.py` | SES-gradient figures (per-sex / gap / marginal x PCA,NBI) + Table 5 (`gradient_cluster_robustness_T18`) | ~40 min |
| `04_gender_gap.py` | Canonical F-M gap (`gender_gap_global_rd_T18.csv`, the §6.3 number), Figure 4, gap-jump forest plot (Figure 7) | ~10 min |
| `05_triple.py` | Tables 3/4/6 (triple interaction, T18/T70), tau-cell + baseline-endpoint figures | ~12 min |
| `06_provincial.py` | 48 province-level RDDs (Table 7), choropleth map + coefplot | ~2 min |
| `07_robustness.py` | Bandwidth sweeps (main + NBI/PCA), placebo cutpoints, rdplots, bw diagnostic | ~36 min |

Full Stage-B compute is ~2.5 h (dominated by the rdrobust loops in `02`, `03`,
`07`). Every script is independently resumable (per-block checkpoints), so an
interrupted run resumes cheaply and `--only NN` re-runs one stage.

## Expected key results

The reference for correctness is the **thesis document itself**, not the legacy
`outputs/`. A correct run reproduces the numbers the manuscript prints:

| Quantity | Value | Where in thesis |
|---|---|---|
| First stage, age 18 (Total) | **+18.25pp**, h = 47.5d, robust 95% CI [16.8, 19.2] | §6.1, Table 2 |
| First stage by sex, age 18 | ~20.0pp men / ~18.6pp women | §6.3 |
| Age-70 effect (Total) | **-3.68pp** | §6.1, Table 2 |
| SES gradient, PCA decile slope | **-1.7pp/decile** men (p=.002), **-2.1** women (p<.001) | §6.2 |
| Delta-tau (p10->p90), PCA | ~ -10.9pp men / -14.7pp women | §6.2 |
| Change in F-M gap at 18 | **-0.59pp**, p=.70, common h 47.4d, 95% CI [-3.6, +2.4], MDE80 4.3pp | §6.3 |
| Triple interaction theta1 (n.s.) | NBI p=.18, PCA p=.11 | §6.4, Table 3 |
| Provincial estimates | 48 rows; La Pampa T70 = -6.67 [-13.58, -0.43] | Table 7 |

## Ownership

`replication/outputs/` is the single sanctioned source of thesis artifacts.
To change a published number: edit the logic (usually in `src/`), re-run the
owning stage (`run_all.py --only NN`), and recompile the thesis. Do **not** rely
on the notebooks or the legacy top-level `outputs/` for anything the document
reads.
