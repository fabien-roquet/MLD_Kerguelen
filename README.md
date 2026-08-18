# MLD Kerguelen

Reproducible analysis workflow for Kerguelen mixed-layer-depth variability.

The code is organized as the three-step analysis formerly developed in the
notebooks. The Python and R scripts are now the canonical workflow:

1. `1_data_processing/`: one Python script per processed dataset.
2. `2_compute_fPCA_R/`: one R script per fPCA dataset.
3. `3_generate_figures/`: one Python script per figure, plus the trend-table
   generator used by the manuscript.

Large NetCDF inputs and generated outputs are intentionally ignored by Git. Place the required local inputs under `data/` before running the pipeline.

## MLD Definition

All MLD products in the workflow use the same density threshold: the depth where
potential-density anomaly differs from its 10 m value by
`0.03 kg m-3`. This applies to CMA, full GLORYS, co-located GLORYS
(`GLORYS_CL`), the GLORYS section product used for Figure 9, and the KERFIX
profiles used for Figure 10. Park et al. (1998) reported KERFIX values with a
`0.02` criterion; those values are used only as historical context.

## Inputs

Expected local files:

- `data/CORA_MEOP_ARGO_2026.nc`
- `data/MEOP_2026.nc` for Figure 1
- `data/CORA_2026.nc` for Figure 1
- `data/ARGO_2026.nc` for Figure 1
- `data/GLORYS_2026.nc`
- `data/GEBCO_ker_large.nc`
- `data/fronts_62985.nc`
- `data/GLORYS_1000m_section_timemean.nc` for Figure 9
- `data/kerfix.csv` for Figure 10

The Figure 9 time-mean section file is generated once from the large temporary
source `data/GLORYS_1000m_section_plot.nc`:

```bash
uv run python 1_data_processing/process_GLORYS_section.py
```

After `data/GLORYS_1000m_section_timemean.nc` exists, the large source file is
not required by the figure script.

## Running

Install Python dependencies:

```bash
uv sync
```

`uv` installs the Python environment. The R interpreter itself still needs to
exist on the machine as `Rscript` because `uv` cannot install R. R packages are
listed in `pyproject.toml` under `[tool.mld-kerguelen.r]`; `run_analysis.py`
checks/installs them into the local `.r-lib/` folder before the fPCA stage.

Bootstrap only the R package library:

```bash
uv run python run_analysis.py --stage setup
```

Run the whole modular pipeline:

```bash
uv run python run_analysis.py
```

By default, the figure stage runs Figure 1, Figures 2-10, and supplementary
Figures S1-S2.
Appendix Figures A1-A5 are opt-in because they depend on sampling outputs.

Each `run_analysis.py` launch prints an estimated total runtime at startup
and the actual wall-clock runtime at the end.

Run only one stage:

```bash
uv run python run_analysis.py --stage data
uv run python run_analysis.py --stage fpca
uv run python run_analysis.py --stage sampling
uv run python run_analysis.py --stage figures --figures 2 3 4
```

When Figure 7 is included in the figure stage, `run_analysis.py` also writes
`processed/3_figures/Table_1_trends.tex` from the same trend calculation.
The figure stage also refreshes `statistics.md` and `summary.md` from the
current outputs.

Refresh statistics files without rerunning figures:

```bash
uv run python scripts/update_statistics_markdown.py
```

Appendix Figures A1-A2 use the GLORYS MLD-anomaly random space-time sampling
PACE sensitivity experiment. The sampling stage defaults to 30 pseudo-random
replicates at 5%, 10%, and 20% sampling:

```bash
uv run python run_analysis.py --stage sampling
uv run python run_analysis.py --stage figures --figures A1 A2
```

For a faster smoke test, reduce the number of replicates:

```bash
uv run python run_analysis.py --stage sampling --sampling-replicates 2 --sampling-levels 5,10
```

Figures A3-A5 use GLORYS cp1 = 0 and the SAF to reproduce the Figure 7 trend
panels within regional masks. A3 is the cp1-positive deep region south of 50S,
A4 is the cp1-negative shallow region, and A5 is the cp1-positive deep region
north of the SAF:

```bash
uv run python run_analysis.py --stage figures --figures A3 A4 A5
```

Run individual scripts directly:

```bash
uv run python 1_data_processing/process_CMA.py
uv run python run_analysis.py --stage fpca --datasets CMA
uv run python 3_generate_figures/Figure_4_MU_modes.py
uv run python 3_generate_figures/Table_1_trends.py
Rscript 2_compute_fPCA_R/script_PCA_GLORYS_random_sampling_2026.R
```

## Reference Outputs

`processed_reference/` is a local, ignored snapshot of the current `processed/` folder. Use it to check whether regenerated outputs still mostly match the reference:

```bash
uv run python scripts/compare_processed_reference.py
```

Figure 6 is generated only as `Figure_6_RMSE_mask.png`; the unmasked RMSE figure is obsolete.
