# Method Summary

This repository reconstructs monthly mixed-layer-depth (MLD) anomalies around the Kerguelen Plateau from sparse observations and evaluates the reconstruction against GLORYS. The workflow has three scripted stages: Python data processing in `1_data_processing/`, R/fdapace fPCA computation in `2_compute_fPCA_R/`, and Python figure/table generation in `3_generate_figures/`.

## Inputs and Domain

- Main inputs: `data/CORA_MEOP_ARGO_2026.nc` and `data/GLORYS_2026.nc`.
- Auxiliary figure inputs: `data/GEBCO_ker_large.nc`, `data/fronts_62985.nc`, `data/GLORYS_1000m_section_timemean.nc`, and `data/kerfix.csv`.
- Analysis period: January 2007 to December 2023, i.e. 204 monthly fields.
- Horizontal domain: the observation-defined Kerguelen box, binned to a `39 x 39` regular grid (`1521` grid cells).
- Total monthly grid-cell slots: `204 x 39 x 39 = 310284`.
- Kerguelen island mask: longitude `68.25-70.75 E`, latitude `50.0-48.0 S`; this removes 20 grid cells from spatial analyses.

## MLD and Gridding

- Observational product: merged CORA, MEOP, and additional Argo profiles, referred to as CMA.
- Current source file contains `97772` finite MLD profile values with `time <= 2023-12-31`.
- CMA profile MLDs are binned by month, longitude bin, and latitude bin using mean MLD inside each bin.
- Figure 1 uses the de-duplicated combined CMA profile set for both the map and the source bars. MEOP and CORA+ARGO bars are assigned by exact matching each combined profile on `(time, longitude, latitude, mld)` against the source files, which avoids double-counting across datasets.
- MLD is defined consistently for every product with a density threshold relative to 10 m:
  - `Delta sigma0 = 0.03 kg m-3`.
  - This applies to CMA profile MLDs, full GLORYS, co-located GLORYS (`GLORYS_CL`), the GLORYS section product used for Figure 9, and KERFIX.
  - Park et al. (1998) reported KERFIX values with a `0.02` criterion; those values are treated only as historical context, not mixed into the manuscript statistics.
- GLORYS MLD is processed in 12-month blocks, then binned and interpolated to the CMA grid.
- `GLORYS_CL` is produced by keeping GLORYS values only where CMA has observations.
- Before fPCA, the domain-mean monthly seasonal cycle is removed from each product. The R input files are anomaly fields in `processed/1_gridded_data/r_input/`.

## fPCA/PACE Setup

All fPCA runs use `fdapace::FPCA`, `set.seed(1)`, `nRegGrid = 204`, `methodSelectK = "FVE"`, and `FVEthreshold = 1`, so all selected modes needed to reach the full available variance threshold are retained.

Sparse products (`CMA` and `GLORYS_CL`) use PACE with conditional-expectation scores:

- `dataType = "Sparse"`.
- `methodMuCovEst = "smooth"`.
- `kernel = "epan"`.
- `userBwMu = 0.05`.
- `userBwCov = 0.25`.
- `methodXi = "CE"`.
- `maxK = 100`.

Full GLORYS uses dense fPCA inside fdapace:

- `dataType = "DenseWithMV"`.
- `methodMuCovEst = "cross-sectional"`.
- `methodXi = "IN"`.
- `maxK = min(nmonth - 2, ncol(donmat) - 1)`.
- `error = FALSE`.

The reconstructed anomaly field is

```text
MLD'(p,t) = mu(t) + sum_k c_pk * xi_k(t)
```

where `mu(t)` is the fPCA mean anomaly, `xi_k(t)` are temporal modes, and `c_pk` are spatial scores. For plotting consistency, mode 1 is sign-flipped for all products, and mode 2 is sign-flipped for `GLORYS_CL`; both the temporal mode and spatial score are flipped together, so the reconstruction is unchanged.

<!-- BEGIN AUTO:CURRENT_DATA_STATS -->
## Current Data Statistics

All values below are computed from the current `processed/` outputs.

| Product | Finite monthly grid values | Mean MLD (m) | Std MLD (m) |
|---|---:|---:|---:|
| GLORYS | 302,362 | 107.15 | 63.16 |
| GLORYS_CL | 26,978 | 102.82 | 57.15 |
| CMA | 27,148 | 109.21 | 68.94 |

The CMA gridded observational coverage is `27148 / 310284 = 8.75%`. The R anomaly input for CMA has 31229 non-missing values (`10.06%`) because the Kerguelen island mask is encoded as zero to preserve the regular grid shape.

Validated Figure 1 counts after the Kerguelen mask are `96216` combined profiles on the map, split into `76558` MEOP profiles and `19658` CORA+ARGO profiles in the bars. Those bar counts now sum exactly to the map total because they are derived from the same de-duplicated combined dataset.

Figure 2 spatial climatology correlations (time-mean maps after masking the Kerguelen island, with each pair evaluated on its own common valid cells):

- GLORYS vs CMA: `0.662`.
- GLORYS_CL vs CMA: `0.817`.
- GLORYS vs GLORYS_CL: `0.729`.

For reference, monthly pooled pointwise correlation at observed cells (not the Figure 2 metric) is `0.795` for `GLORYS_CL vs CMA`.

These values are not computed on exactly the same missing-data mask across every product pair. In particular, `GLORYS_CL` is created by retaining only GLORYS values where CMA has observations, so `GLORYS_CL vs CMA` is the most direct comparison of the co-located products.
<!-- END AUTO:CURRENT_DATA_STATS -->

## fPCA Variance Statistics

| Product | Retained modes | Mode 1 (%) | Mode 2 (%) | First 2 modes (%) | First 5 modes (%) |
|---|---:|---:|---:|---:|---:|
| GLORYS | 125 | 52.78 | 14.81 | 67.58 | 75.53 |
| GLORYS_CL | 86 | 47.92 | 10.85 | 58.76 | 76.48 |
| CMA | 88 | 42.75 | 12.08 | 54.84 | 72.36 |

Temporal-mode summary:

| Product | mean xi1 | std xi1 | mean xi2 | std xi2 |
|---|---:|---:|---:|---:|
| GLORYS | 0.179 | 0.174 | -0.139 | 0.209 |
| GLORYS_CL | 0.217 | 0.123 | -0.023 | 0.249 |
| CMA | 0.207 | 0.141 | -0.087 | 0.235 |

## Reconstruction Error

RMSE is evaluated on the co-located `GLORYS_CL` sampling mask for all three products. Values are spatial means and spatial standard deviations of the cellwise temporal RMSE maps.

| Product | Mean RMSE (m) | Std RMSE (m) |
|---|---:|---:|
| GLORYS | 9.32 | 7.53 |
| GLORYS_CL | 16.50 | 8.48 |
| CMA | 28.79 | 17.09 |

Current Figure 6 plotting parameters:

- RMSE map range: `0-80 m`.
- Monthly RMSE y-axis: `0-60 m`.
- Only `Figure_6_RMSE_mask.png` is retained.

## Trend Statistics

Trends are computed from domain-mean reconstructed MLD anomaly time series using annual means, summer means (January-March), and winter means (July-September). Slopes are in `m yr-1`; p-values are in parentheses.

| Period | GLORYS | GLORYS_CL | CMA |
|---|---:|---:|---:|
| Annual | -0.20 (0.21) | -0.41 (0.05) | -0.30 (0.14) |
| Summer | 0.17 (0.33) | 0.25 (0.33) | 0.23 (0.43) |
| Winter | 0.06 (0.85) | -0.91 (0.05) | -0.63 (0.37) |

The table is regenerated by `3_generate_figures/Table_1_trends.py` and written to `processed/3_figures/Table_1_trends.tex` when Figure 7 is included in `run_analysis.py`.

## Figure-Specific Notes

- Figure 4 shows only the first two temporal modes; the domain-mean panel was removed because it is redundant with Figure 3.
- Figure 5 uses `cmo.balance`; `cp1` is plotted over `[-250, 250]`, and `cp2` over `[-50, 50]`.
- Figure 9 is generated from `data/GLORYS_1000m_section_timemean.nc`; the larger temporary file `data/GLORYS_1000m_section_plot.nc` is not required once the time mean exists.
