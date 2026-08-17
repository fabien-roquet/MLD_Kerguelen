# Statistics Audit

This file collects the main numbers currently used to update `latex/manuscript_doriot_v3.tex`. Values are from the local `data/` and `processed/` outputs unless stated otherwise.

## Source Data and Gridding

| Quantity | Value | Source |
|---|---:|---|
| CMA source MLD profiles | 97,772 | finite `mld` values with `time <= 2023-12-31` in `data/CORA_MEOP_ARGO_2026.nc` |
| Time span | Jan 2007-Dec 2023 | source data and processed grids |
| Monthly fields | 204 | `processed/1_gridded_data/*.nc` |
| Horizontal grid | 39 x 39 | processing scripts |
| Grid cells | 1,521 | processing scripts |
| Total monthly grid-cell slots | 310,284 | 204 x 39 x 39 |
| Kerguelen island mask | 20 cells | lon 68.25-70.75 E, lat 50-48 S |
| CMA finite monthly grid values | 27,148 | `CMA_gridded.nc` |
| CMA gridded coverage | 8.75% | 27,148 / 310,284 |
| CMA R-input non-missing values | 31,229 | includes island-mask zeros |
| CMA R-input coverage | 10.06% | 31,229 / 310,284 |

For Figure 1, source bars are now reconstructed safely from the de-duplicated combined CMA file by exact matching on `(time, longitude, latitude, mld)` against the MEOP, CORA, and ARGO source files. This avoids double-counting while keeping the bars on the same profile universe as the map.

Validated Figure 1 counts after the Kerguelen mask:

| Quantity | Value |
|---|---:|
| Combined profiles on map | 96,216 |
| MEOP profiles in bars | 76,558 |
| CORA+ARGO profiles in bars | 19,658 |
| Occupied monthly CMA cells | 27,148 |

The de-duplicated source split in the combined file is 78,108 MEOP profiles and 19,664 CORA+ARGO profiles before the Kerguelen mask. The standalone CORA, MEOP, and ARGO files sum to 97,783 finite profiles over the period, but 11 CORA records are not present in the merged CMA file.

Profile counts by year in `data/CORA_MEOP_ARGO_2026.nc`:

| Year | Profiles |
|---:|---:|
| 2007 | 616 |
| 2008 | 2,323 |
| 2009 | 3,246 |
| 2010 | 2,335 |
| 2011 | 6,629 |
| 2012 | 6,710 |
| 2013 | 8,293 |
| 2014 | 11,187 |
| 2015 | 6,261 |
| 2016 | 3,982 |
| 2017 | 6,415 |
| 2018 | 5,370 |
| 2019 | 10,758 |
| 2020 | 5,866 |
| 2021 | 7,302 |
| 2022 | 7,075 |
| 2023 | 3,447 |

The average over 2011-2021 is 7,161 profiles per year, rounded to about 7,160 in the manuscript.

Profile counts by calendar month:

| Month | Profiles |
|---:|---:|
| 1 | 14,000 |
| 2 | 11,801 |
| 3 | 8,817 |
| 4 | 7,150 |
| 5 | 8,070 |
| 6 | 7,512 |
| 7 | 7,386 |
| 8 | 6,789 |
| 9 | 7,463 |
| 10 | 6,032 |
| 11 | 6,547 |
| 12 | 6,248 |

## Gridded MLD Statistics

Mean and standard deviation are over all finite monthly grid cells.

| Product | Valid values | Mean MLD (m) | Std MLD (m) |
|---|---:|---:|---:|
| GLORYS | 307,470 | 98.45 | 60.43 |
| GLORYS_CL | 27,339 | 102.61 | 57.03 |
| CMA | 27,148 | 109.21 | 68.94 |

Pointwise GLORYS-CMA difference at observed monthly grid cells:

| Difference | n | Mean (m) | Std (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| GLORYS - CMA | 27,041 | -13.67 | 40.47 | 42.72 |
| GLORYS_CL - CMA | 26,978 | -4.45 | 39.17 | 39.42 |

Pointwise correlations at observed monthly grid cells:

| Pair | Correlation | n |
|---|---:|---:|
| GLORYS vs CMA | 0.786 | 27,041 |
| GLORYS_CL vs CMA | 0.795 | 26,978 |
| GLORYS vs GLORYS_CL | 0.986 | 27,339 |

Spatial climatology correlations:

| Pair | Correlation | n |
|---|---:|---:|
| GLORYS vs CMA | 0.668 | 1,501 |
| GLORYS_CL vs CMA | 0.817 | 1,501 |
| GLORYS vs GLORYS_CL | 0.739 | 1,521 |

## Seasonal Cycle and Domain-Mean Anomalies

Statistics below use the products plotted in Figure 3: monthly climatology files and domain-mean anomaly files.

| Product | Climatology min (m) | Climatology max (m) | Seasonal amplitude (m) | Anomaly mean (m) | Anomaly std (m) |
|---|---:|---:|---:|---:|---:|
| GLORYS | 49.77 | 178.58 | 128.81 | -0.96 | 8.57 |
| GLORYS_CL | 53.46 | 177.21 | 123.74 | -0.11 | 10.78 |
| CMA | 62.54 | 184.61 | 122.08 | -0.49 | 12.77 |

Domain-mean anomaly differences:

| Difference | Mean (m) | Std (m) | Min (m) | Max (m) |
|---|---:|---:|---:|---:|
| GLORYS_CL - GLORYS | 0.84 | 7.84 | -28.69 | 24.67 |
| CMA - GLORYS | 0.47 | 11.01 | -37.23 | 68.71 |

## fPCA/PACE Statistics

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

Figure 6 RMSE values are evaluated on the co-located `GLORYS_CL` sampling mask.

| Product | Mean RMSE (m) | Std RMSE (m) |
|---|---:|---:|
| GLORYS | 9.32 | 7.53 |
| GLORYS_CL | 16.50 | 8.48 |
| CMA | 28.79 | 17.09 |

## Domain-Mean Trends

Slopes are in `m yr-1`, with p-values in parentheses.

| Period | GLORYS | GLORYS_CL | CMA |
|---|---:|---:|---:|
| Annual | -0.20 (0.21) | -0.41 (0.05) | -0.30 (0.14) |
| Summer (JFM) | 0.17 (0.33) | 0.25 (0.33) | 0.23 (0.43) |
| Winter (JAS) | 0.06 (0.85) | -0.91 (0.05) | -0.63 (0.37) |

## KERFIX / Figure 10

KERFIX station position used in Figure 10: 50.6667 S, 68.4167 E. The GLORYS grid point selected by nearest-neighbor interpolation is 50.5125 S, 68.4615 E.

KERFIX MLD is recomputed from `data/kerfix.csv` with the same `0.03 kg m-3` density threshold relative to 10 m used for CMA, GLORYS, GLORYS_CL, and the GLORYS section product.

| Product | Period | n | Mean MLD (m) | Std MLD (m) | Min (m) | Max (m) |
|---|---|---:|---:|---:|---:|---:|
| KERFIX | Dec 1990-Dec 1994 | 45 | 139.62 | 53.60 | 49.56 | 247.69 |
| GLORYS at KERFIX | Jan 2007-Dec 2023 | 204 | 107.07 | 45.91 | 34.05 | 229.94 |

Seasonal means:

| Product | Annual mean (m) | Summer JFM (m) | Winter JAS (m) |
|---|---:|---:|---:|
| KERFIX | 139.62 | 92.51 | 192.40 |
| GLORYS at KERFIX | 107.07 | 64.11 | 159.59 |

Simple monthly linear fits at the station:

| Product | Slope (m yr-1) | p-value |
|---|---:|---:|
| KERFIX, 1990-1994 | 15.23 | 0.037 |
| GLORYS at KERFIX, 2007-2023 | 0.13 | 0.840 |

Park et al. (1998) reported KERFIX MLD with a `0.02 sigma_theta` density-difference criterion. Those values are useful context for the historical station record, but they are not used in the manuscript statistics or Figure 10 comparison.

## Park et al. (1998) Context

ScienceDirect metadata/abstract for Park et al. (1998) states that monthly hydrographic data were collected at KERFIX from May 1991 to December 1994, at about 50 deg 40 min S, 68 deg 25 min E. MLD was determined in that paper using a `0.02 sigma_theta` density-difference criterion. The paper reported a strong seasonal cycle, warm/low-salinity anomalies in 1992, cold/high-salinity anomalies in 1994, and significantly deeper mixed layers in the later years of the short record, with possible links to ENSO.

Bibliographic check: the DOI `10.1016/S0924-7963(98)00065-7` corresponds to Journal of Marine Systems volume 17, issues 1-4, pages 571-586, not pages 233-247.
