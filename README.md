# Who Votes Because They Have To? — replication package

Code and generated output for the MSc dissertation *Who Votes Because They Have
To? Heterogeneous Effects of Compulsory Voting on Turnout by Socioeconomic
Status and Gender — Argentina 2025* (MSc Applied Social Data Science, London
School of Economics).

The design is a sharp regression discontinuity at the two age thresholds in
Argentine electoral law: voting becomes compulsory at 18 and voluntary again at
70. The running variable is the exact distance in days from each threshold on
election day, 26 October 2025. Estimation uses `rdrobust` with MSE-optimal
bandwidths and the bias-corrected robust confidence intervals of Calonico,
Cattaneo and Titiunik (2014), clustered by electoral circuit.

## What you can do with this repository

**You can read the code and inspect the output it produced.** Every figure and
table the manuscript displays — 26 figures and 8 LaTeX tables — is committed
under `replication/outputs/`, byte-identical to the files that compiled the
submitted PDF. Each one is written by a numbered script in `replication/`, so
any published number can be traced to the code behind it. Nothing else is
committed: the scripts write further intermediate CSVs, but only the manuscript's
artifacts are kept here.

**You cannot re-run the analysis.** It reads the individual-level electoral
register — 35,987,634 records with date of birth, sex, polling place and a
validated turnout flag. The permission under which the work was done covers
analysing that register, not redistributing it, so it is not in this repository
and neither stage runs without it. The table below documents how it was
obtained.

What you can run without the register is `verify.py`, which confirms all 34
artifacts are present. It needs nothing but the standard library:

```bash
python replication/verify.py
```

One thing that may look like a missing file but is not: several docstrings carry
provenance comments pointing at the exploratory notebooks the scripts were
transcribed from (`transcribed from notebooks/04_rdd.ipynb, cell 9`). Those
notebooks are not part of the replication.

## Contents

```
replication/            the numbered scripts and their outputs
  run_all.py            orchestrator: Stage B, then verify
  verify.py             checks all 34 artifacts are present; stdlib only
  00_build_data.py      Stage A: raw sources -> data/final/national.parquet
  00b_spatial_illustration.py   Stage A: the Appendix A join figure
  00c_fix_election_date.py      one-shot date migration, already applied
  01_descriptives.py .. 08_geography.py    Stage B
  outputs/figures/      the 26 figures the manuscript displays
  outputs/tables/       the 8 LaTeX tables it prints, plus circuit_geography.csv
src/                    the analysis library the scripts call
  pipeline.py           spatial join, NBI, PCA deprivation index
  analysis.py           RD estimation and plotting helpers
  format_tables.py      LaTeX table formatting
  lapop_triangulation.py   the LAPOP cross-check
config_template.py      copy to config.py; paths are repo-relative
environment.yml         conda environment
```

`circuit_geography.csv` is the one committed file that is an input rather than
an output. `08_geography.py` reads it so that its estimation stage runs from
the parquet alone, without the electoral-circuit GeoJSON layer.

## Data sources

| Source | What it provides | How to obtain it |
|---|---|---|
| National electoral register, 2025 general election | Individual records: date of birth, sex, electoral circuit, validated turnout | Freedom-of-information request (*Solicitud de Acceso a la Información Pública*) to the Cámara Nacional Electoral, channelled through the Consejo de la Magistratura. Supplied anonymised. |
| INDEC Census 2022, household indicators by census tract | NBI and the five deprivation variables behind the PCA index | Public. [INDEC census results](https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165) |
| Electoral circuit boundaries, all 24 districts | The geometry the census is joined to | Public, compiled by the author: [`tartagalensis/circuitos_electorales_AR`](https://github.com/tartagalensis/circuitos_electorales_AR) |
| LAPOP AmericasBarometer Argentina 2023 | Self-reported turnout by sex, used once for triangulation | Public, registration required. [LAPOP](https://www.vanderbilt.edu/lapop/) |

## Running it

```bash
conda env create -f environment.yml
conda activate thesis_lse_env
cp config_template.py config.py
```

`config.py` is repo-relative and needs no editing. The large raw inputs are
reached through symlinks placed in `data/raw/`:

```bash
ln -s /path/to/circuitos_electorales_AR/2025   data/raw/circuitos_2025
ln -s /path/to/registro_infractores_2025       data/raw/padron_2025
ln -s /path/to/Censos_Argentina/2022           data/raw/censo_2022
```

The pipeline has two stages, and they differ in what they need.

**Stage A** (`00_build_data.py`, `00b_spatial_illustration.py`) builds
`data/final/national.parquet` from the raw register, the census and the circuit
layer, and draws the Appendix A join figure. It reads all three raw sources, so
only someone holding the register can run it.

**Stage B** (`01`–`08`) produces every figure and table in the thesis from that
parquet. About 2.5 hours; each script is checkpointed per block and resumable,
so an interrupted run picks up cheaply.

```bash
python replication/run_all.py                # Stage B, then verify
python replication/run_all.py --build-data   # prepend Stage A
python replication/run_all.py --only 02,04   # selected stages
python replication/run_all.py --skip 07      # skip a stage
python replication/verify.py                 # completeness check only
```

`00c_fix_election_date.py` is a one-shot migration, kept for the audit trail and
written to refuse a second application. The pipeline was originally built with
`ELECTION_DATE = 2025-10-25`; the election was held on Sunday the 26th. Because
that constant enters `src/pipeline.py` only as a minuend, the defect was a
uniform one-day shift in `age_days`, `days_from_18` and `days_from_70`, and
`00c` applies the correction to a stored parquet after checking it against a
live Stage-A recomputation of one province. Run it with
`run_all.py --fix-date` only against a parquet built before the fix.

## Scripts and what they produce

Runtimes are measured, from the full rebuild of 7 August 2026.

| Script | Manuscript artifacts | Runtime |
|---|---|---|
| `01_descriptives.py` | Table 1 (`descriptive_stats_latex`), descriptive figures | 1 min |
| `02_main_rdd.py` | Table 2 (`rdd_results_latex`, 30 rows), main RD figures, covariate balance, density test | 33 min |
| `03_gradient.py` | SES-gradient figures (per-sex, gap, marginal × PCA/NBI), Table 5 | 32 min |
| `04_gender_gap.py` | Canonical F−M gap (the §6.3 estimate), gap-jump forest plot | 13 min |
| `05_triple.py` | Tables 3, 4 and 6 (triple interaction, T18/T70), τ-cell and baseline-endpoint figures | 13 min |
| `06_provincial.py` | 48 province-level RDDs (Table 7) | 2 min |
| `07_robustness.py` | Bandwidth sweeps, placebo cutpoints, rdplots | 55 min |
| `08_geography.py` | Appendix geography table (`gradient_geography_T18`) | not in the timed run |

Most of these scripts also write intermediate CSVs and exploratory figures the
manuscript does not use. A run puts them in `replication/outputs/` alongside the
rest; they are not committed here.

## Headline estimates

| | |
|---|---|
| First stage, age 18 | **+21.66pp** (robust 95% CI [20.3, 22.6]), h = 66.0 days |
| Age 70, compulsion lifted | **−4.00pp** |
| SES gradient, p10 to p90 of the PCA index | **−12.6pp** for men, **−17.2pp** for women |
| Change in the female−male gap at 18, common bandwidth | **−1.22pp**, 95% CI [−3.8, +1.4], p = .36, MDE₈₀ 3.7pp |

Socioeconomic status is measured at the electoral circuit, from the census: the
official NBI rate and a principal-component deprivation index over five
tract-level indicators. Sex is measured on the individual. The manuscript
discusses the asymmetry.

## Limitations

**Province filter in `src/pipeline.py:load_censo`.** Census departments are
selected by a string prefix test against the province's INDEC code. Those codes
are four digits for single-digit provinces and five for two-digit ones, so the
prefixes collide and four provinces load some tracts belonging to others. Those
tracts fail the containment test against that province's circuits and are then
attached by the nearest-circuit fallback, which concentrates them in the handful
of border circuits nearest the foreign province. The affected share of voters is
small enough not to move the reported estimates, which is why the published
outputs were not rebuilt. The fix is one line — replace the prefix test with
`df['Código de departamentos/comuna'].astype(str).str[:-3].astype(int) ==
indec_code` — but it invalidates the parquet and so requires re-running both
stages. The code here is the code that produced these outputs, and has been left
that way deliberately.

**Spatial join coverage.** 99.4% of census tracts are assigned by containment;
407 of 66,422 go through the nearest-circuit fallback. These are rural: the
census partitions the territory exhaustively and the circuit layer does not, so
a tract whose centroid falls in unmapped country has no polygon to sit in.

## Citation

> Galeano, F. (2026). *Who Votes Because They Have To? Heterogeneous Effects of
> Compulsory Voting on Turnout by Socioeconomic Status and Gender — Argentina
> 2025.* MSc dissertation, London School of Economics and Political Science.

## Licence

MIT, see `LICENSE`.
