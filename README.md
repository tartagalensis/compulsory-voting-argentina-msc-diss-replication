# Who Votes Because They Have To? — replication package

Code and generated output for the MSc dissertation *Who Votes Because They Have
To? Heterogeneous Effects of Compulsory Voting on Turnout by Socioeconomic
Status and Gender — Argentina 2025* (MSc Applied Social Data Science, London
School of Economics).

Every figure and table the manuscript displays is written by a script in
`replication/` into `replication/outputs/`, and both the scripts and their
output are here, so each published number can be traced to the code that
produced it. The manuscript itself will be added once the dissertation has been
marked.

---

## What is here, and what is not

```
replication/            the numbered scripts and their outputs
  run_all.py            orchestrator: Stage B (01-07), then verify
  verify.py             asserts every artifact the thesis references exists
  00_build_data.py      Stage A: raw sources -> data/final/national.parquet
  00b_spatial_illustration.py   Stage A: the Appendix A join figure
  01_descriptives.py .. 07_robustness.py    Stage B
  outputs/figures/      what the PDF shows
  outputs/tables/       what the PDF prints, plus the CSVs behind them
src/                    the analysis library the scripts call
  pipeline.py           spatial join, NBI, PCA deprivation index
  analysis.py           RD estimation and plotting helpers
config_template.py      copy to config.py; paths are repo-relative
environment.yml         conda environment
```

**The data are not here.** The analysis runs on the individual-level electoral
register: 35,987,634 records carrying date of birth, sex, polling place and a
validated turnout flag. Redistributing that is a different act from analysing
it, and it is not covered by the permission under which the work was done, so
the register is not in this repository. The section below documents how it was
obtained.

The exploratory notebooks from which these scripts were transcribed are also not
here; they live in the author's working repository. Several docstrings still
carry provenance comments referring to them ("transcribed from
`notebooks/04_rdd.ipynb`, cell 9"). Those are internal notes, not files you are
missing.

---

## Data sources

| Source | What it provides | How to obtain it |
|---|---|---|
| National electoral register, 2025 general election | Individual records: date of birth, sex, electoral circuit, validated turnout | Freedom-of-information request (*Solicitud de Acceso a la Información Pública*) to the Cámara Nacional Electoral, channelled through the Consejo de la Magistratura. Supplied anonymised. |
| INDEC Census 2022, household indicators by census tract | NBI and the five deprivation variables behind the PCA index | Public. [INDEC census results](https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165) |
| Electoral circuit boundaries, all 24 districts | The geometry the census is joined to | Public, compiled by the author: [`tartagalensis/circuitos_electorales_AR`](https://github.com/tartagalensis/circuitos_electorales_AR) |
| LAPOP AmericasBarometer Argentina 2023 | Self-reported turnout by sex, used once for triangulation | Public, registration required. [LAPOP](https://www.vanderbilt.edu/lapop/) |

---

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

The pipeline has two stages, and they differ in what they need:

**Stage A** (`00_build_data.py`) builds `data/final/national.parquet` from the
raw register, the census and the circuit layer. It needs all three, so it can be
run only by someone who has obtained the register.

**Stage B** (`01`–`07`) produces every figure and table in the thesis from that
parquet. About 2.5 hours; each stage is checkpointed and resumable.

```bash
python replication/run_all.py                # Stage B, then verify
python replication/run_all.py --build-data   # prepend Stage A
python replication/run_all.py --only 02,04   # selected stages
python replication/verify.py                 # completeness check only
```

**So: without the register, neither stage runs.** What this repository lets you
check without it is the code itself and the outputs it produced, both of which
are here in full. Note that `verify.py` parses the manuscript's LaTeX sources to
learn which artifacts to look for, so it runs only in the author's working
repository, where those sources live.

---

## Design and headline estimates

Sharp regression discontinuity at the two age thresholds in Argentine electoral
law: voting becomes compulsory at 18 and voluntary again at 70. The running
variable is the exact distance in days from each threshold on election day
(25 October 2025). Estimation uses `rdrobust` with MSE-optimal bandwidths and
the bias-corrected robust confidence intervals of Calonico, Cattaneo and
Titiunik (2014), clustered by electoral circuit.

| | |
|---|---|
| First stage, age 18 | **18.25pp** (robust 95% CI [16.8, 19.2]), h = 47.5 days |
| Age 70, compulsion lifted | −3.68pp |
| SES gradient, p10 to p90 of the PCA index | **−10.9pp** for men, **−14.7pp** for women (SE 1.7, p < .001) |
| Change in the female−male gap at 18, common bandwidth | **−0.59pp**, 95% CI [−3.6, +2.4], MDE₈₀ 4.3pp |

Socioeconomic status is measured at the electoral circuit, from the census: the
official NBI rate and a principal-component deprivation index over five
tract-level indicators. Sex is measured on the individual. The asymmetry is
discussed in the manuscript.

---

## Known issues

**Province filter in `src/pipeline.py:load_censo` (present in the published
outputs).** Census departments are selected by a string prefix test against the
province's INDEC code. Those codes are four digits for single-digit provinces
(CABA `2xxx`, Buenos Aires `6xxx`) and five for two-digit ones (Chaco `22xxx`,
Chubut `26xxx`, Río Negro `62xxx`, Salta `66xxx`), so the prefixes collide and
four provinces load tracts belonging to others. Those tracts fail the
containment test against that province's circuits and are then attached by the
nearest-circuit fallback.

Measured footprint: the fallback concentrates them into the handful of border
circuits nearest the foreign province rather than spreading them, so **22
circuits of about 5,900** shift by more than 0.05pp, and **84,426 voters of
35.99M (0.235%)** sit in those circuits — **325 of the 154,781** inside the
age-18 estimation window. That is too small to move any estimate above, which is
why the published outputs were not rebuilt.

The fix is one line: replace the prefix test with
`df['Código de departamentos/comuna'].astype(str).str[:-3].astype(int) ==
indec_code`. It invalidates the parquet and therefore requires re-running both
stages. **The code here is the code that produced these outputs**; it has been
left that way deliberately, since a repository whose code and outputs disagree
is worse than a documented defect.

**Spatial join coverage.** 99.39% of census tracts are assigned by containment;
407 of 66,422 (0.61%) go through the nearest-circuit fallback. These are rural:
the census partitions the territory exhaustively and the circuit layer does not,
so a tract whose centroid falls in unmapped country has no polygon to sit in.

---

## Citation

> Galeano, F. (2026). *Who Votes Because They Have To? Heterogeneous Effects of
> Compulsory Voting on Turnout by Socioeconomic Status and Gender — Argentina
> 2025.* MSc dissertation, London School of Economics and Political Science.

---

## Licence

MIT, see `LICENSE`.
