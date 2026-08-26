#!/usr/bin/env python3
"""Refresh all computed statistics sections in statistics.md.

This script recomputes all values that depend on current data/processed outputs,
including Figure 1 profile counts, seasonal-cycle diagnostics, fPCA summaries,
reconstruction RMSE, domain-mean trends, and KERFIX/Figure 10 station stats.

It also refreshes the auto-managed gridded block via update_audit_markdown.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
STATISTICS_PATH = ROOT / "statistics.md"

sys.path.insert(0, str(ROOT / "3_generate_figures"))

from Figure_1_prof_dist_year import (  # type: ignore
    observation_grid_bins,
    profile_counts_on_template_grid,
    profile_dataframe,
    split_combined_sources,
    yearly_monthly_counts,
)
from Figure_6_quadratic_error import N_RECONSTRUCTION_MODES, compute_qe  # type: ignore
from Table_1_trends import trend_stats  # type: ignore
from create_rec_datasets import r_analysis_df  # type: ignore
from figure_common import kerguelen_mask, load_fpca, open_gridded  # type: ignore
from update_audit_markdown import main as update_gridded_blocks


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_f2(value: float) -> str:
    return f"{value:.2f}"


def _fmt_f3(value: float) -> str:
    return f"{value:.3f}"


def _source_block() -> str:
    ds_total = xr.open_dataset(ROOT / "data" / "CORA_MEOP_ARGO_2026.nc")
    df_total = profile_dataframe(ds_total)

    ds_cma_grid = xr.open_dataset(ROOT / "processed" / "1_gridded_data" / "CMA_gridded.nc")
    ds_cora = xr.open_dataset(ROOT / "data" / "CORA_2026.nc")
    ds_argo = xr.open_dataset(ROOT / "data" / "ARGO_2026.nc")
    ds_meop = xr.open_dataset(ROOT / "data" / "MEOP_2026.nc")

    lon_bins, lat_bins, time_bins = observation_grid_bins(ds_total, ds_cma_grid)
    total_df, meop_df, other_df = split_combined_sources(ds_total, ds_meop, ds_cora, ds_argo)

    total_count = profile_counts_on_template_grid(total_df, ds_cma_grid, lon_bins, lat_bins, time_bins)
    meop_count = profile_counts_on_template_grid(meop_df, ds_cma_grid, lon_bins, lat_bins, time_bins)
    other_count = profile_counts_on_template_grid(other_df, ds_cma_grid, lon_bins, lat_bins, time_bins)

    mask = kerguelen_mask(ds_cma_grid)
    total_count = total_count.where(~mask)
    meop_count = meop_count.where(~mask)
    other_count = other_count.where(~mask)
    ds_cma_masked = ds_cma_grid.where(~mask)

    hist_other_y, hist_other_m = yearly_monthly_counts(other_count)
    hist_meop_y, hist_meop_m = yearly_monthly_counts(meop_count)
    all_years = np.arange(2007, 2024)
    months_idx = np.arange(1, 13)
    hist_other_y = hist_other_y.reindex(year=all_years, fill_value=0)
    hist_meop_y = hist_meop_y.reindex(year=all_years, fill_value=0)
    hist_other_m = hist_other_m.reindex(month=months_idx, fill_value=0)
    hist_meop_m = hist_meop_m.reindex(month=months_idx, fill_value=0)
    hist_total_y = hist_other_y + hist_meop_y
    hist_total_m = hist_other_m + hist_meop_m

    standalone_total = 0
    for fname in ["CORA_2026.nc", "MEOP_2026.nc", "ARGO_2026.nc"]:
        standalone_total += int(profile_dataframe(xr.open_dataset(ROOT / "data" / fname)).shape[0])

    cma_r = pd.read_csv(ROOT / "processed" / "1_gridded_data" / "r_input" / "CMA_masked.txt", sep=r"\s+")
    cma_r_nonmissing = int(cma_r["mld"].notna().sum())

    year_rows = "\n".join(f"| {y} | {_fmt_int(int(v))} |" for y, v in zip(all_years, hist_total_y.values))
    month_rows = "\n".join(f"| {m} | {_fmt_int(int(v))} |" for m, v in zip(months_idx, hist_total_m.values))

    total_slots = int(ds_cma_grid.sizes["time"] * ds_cma_grid.sizes["latitude"] * ds_cma_grid.sizes["longitude"])
    cma_finite = int(ds_cma_grid.mld.count().item())

    return f"""## Source Data and Gridding

| Quantity | Value | Source |
|---|---:|---|
| CMA source MLD profiles | {_fmt_int(int(df_total.shape[0]))} | finite `mld` values with `time <= 2023-12-31` in `data/CORA_MEOP_ARGO_2026.nc` |
| Time span | Jan 2007-Dec 2023 | source data and processed grids |
| Monthly fields | {_fmt_int(int(ds_cma_grid.sizes['time']))} | `processed/1_gridded_data/*.nc` |
| Horizontal grid | {int(ds_cma_grid.sizes['latitude'])} x {int(ds_cma_grid.sizes['longitude'])} | processing scripts |
| Grid cells | {_fmt_int(int(ds_cma_grid.sizes['latitude'] * ds_cma_grid.sizes['longitude']))} | processing scripts |
| Total monthly grid-cell slots | {_fmt_int(total_slots)} | 204 x 39 x 39 |
| Kerguelen island mask | {_fmt_int(int(mask.sum().item()))} cells | lon 68.25-70.75 E, lat 50-48 S |
| CMA finite monthly grid values | {_fmt_int(cma_finite)} | `CMA_gridded.nc` |
| CMA gridded coverage | {100 * cma_finite / total_slots:.2f}% | {_fmt_int(cma_finite)} / {_fmt_int(total_slots)} |
| CMA R-input non-missing values | {_fmt_int(cma_r_nonmissing)} | includes island-mask zeros |
| CMA R-input coverage | {100 * cma_r_nonmissing / total_slots:.2f}% | {_fmt_int(cma_r_nonmissing)} / {_fmt_int(total_slots)} |

For Figure 1, source bars are now reconstructed safely from the de-duplicated combined CMA file by exact matching on `(time, longitude, latitude, mld)` against the MEOP, CORA, and ARGO source files. This avoids double-counting while keeping the bars on the same profile universe as the map.

Validated Figure 1 counts after the Kerguelen mask:

| Quantity | Value |
|---|---:|
| Combined profiles on map | {_fmt_int(int(total_count.sum().item()))} |
| MEOP profiles in bars | {_fmt_int(int(meop_count.sum().item()))} |
| CORA+ARGO profiles in bars | {_fmt_int(int(other_count.sum().item()))} |
| Occupied monthly CMA cells | {_fmt_int(int(ds_cma_masked.mld.count().item()))} |

The de-duplicated source split in the combined file is {_fmt_int(int(meop_df.shape[0]))} MEOP profiles and {_fmt_int(int(other_df.shape[0]))} CORA+ARGO profiles before the Kerguelen mask. The standalone CORA, MEOP, and ARGO files sum to {_fmt_int(standalone_total)} finite profiles over the period, but {standalone_total - int(total_df.shape[0])} CORA records are not present in the merged CMA file.

Profile counts by year in `data/CORA_MEOP_ARGO_2026.nc`:

| Year | Profiles |
|---:|---:|
{year_rows}

The full-period mean is {_fmt_int(int(round(float(hist_total_y.mean().item()))))} ± {_fmt_int(int(round(float(hist_total_y.std().item()))))} profiles per year over 2007-2023 (mean ± std).

Profile counts by calendar month:

| Month | Profiles |
|---:|---:|
{month_rows}
"""


def _seasonal_fpca_rmse_trends_kerfix_block() -> str:
    # Seasonal cycle and anomaly stats
    seasonal = {}
    for name in ["GLORYS", "GLORYS_CL", "CMA"]:
        ds_anom = open_gridded(ROOT, f"{name}_anom.nc")
        ds_clim = open_gridded(ROOT, f"{name}_clim.nc")
        clim_ts = ds_clim.mld.mean(dim=["latitude", "longitude"])
        anom_ts = ds_anom.mld.mean(dim=["latitude", "longitude"])
        seasonal[name] = {
            "clim_min": float(clim_ts.min().item()),
            "clim_max": float(clim_ts.max().item()),
            "amp": float((clim_ts.max() - clim_ts.min()).item()),
            "anom_mean": float(anom_ts.mean().item()),
            "anom_std": float(anom_ts.std().item()),
        }

    anom_g = open_gridded(ROOT, "GLORYS_anom.nc").mld.mean(dim=["latitude", "longitude"])
    anom_cl = open_gridded(ROOT, "GLORYS_CL_anom.nc").mld.mean(dim=["latitude", "longitude"])
    anom_cma = open_gridded(ROOT, "CMA_anom.nc").mld.mean(dim=["latitude", "longitude"])
    d_cl_g = (anom_cl - anom_g).values
    d_cma_g = (anom_cma - anom_g).values

    # fPCA summaries
    fpca = load_fpca(ROOT)
    fpca_stats = {}
    for name in ["GLORYS", "GLORYS_CL", "CMA"]:
        ds, _, _, _, _, _, df_lambda = fpca[name]
        lam = df_lambda.iloc[:, 0].to_numpy(dtype=float)
        ve = 100 * lam / lam.sum()
        fpca_stats[name] = {
            "retained_modes": int(len(lam)),
            "mode1": float(ve[0]),
            "mode2": float(ve[1]),
            "first2": float(ve[:2].sum()),
            "first5": float(ve[:5].sum()),
            "phi1_mean": float(ds["phi1"].mean().item()),
            "phi1_std": float(ds["phi1"].std().item()),
            "phi2_mean": float(ds["phi2"].mean().item()),
            "phi2_std": float(ds["phi2"].std().item()),
        }

    # RMSE summaries (Figure 6 definition)
    fpca_50 = load_fpca(ROOT, max_modes=N_RECONSTRUCTION_MODES)
    ds_g = fpca_50["GLORYS"][0]
    ds_cl = fpca_50["GLORYS_CL"][0]
    ds_cma = fpca_50["CMA"][0]

    def align_original_anomaly(filename: str, template: xr.Dataset) -> xr.Dataset:
        ds = open_gridded(ROOT, filename)
        ds = ds.where(ds.time < pd.to_datetime("2024-01-01"), drop=True)
        ds = ds.rename({"longitude": "long", "latitude": "lat"})
        ds["time"] = template.time
        ds["lat"] = template.lat
        ds["long"] = template.long
        return ds

    ds_g_og = align_original_anomaly("GLORYS_anom.nc", ds_g)
    ds_cl_og = align_original_anomaly("GLORYS_CL_anom.nc", ds_g)
    ds_cma_og = align_original_anomaly("CMA_anom.nc", ds_g)
    evaluation_mask = ds_cl_og["mld"].notnull()
    qe_g, _ = compute_qe(ds_g, ds_g_og, mask=evaluation_mask)
    qe_cl, _ = compute_qe(ds_cl, ds_cl_og, mask=evaluation_mask)
    qe_cma, _ = compute_qe(ds_cma, ds_cma_og, mask=evaluation_mask)

    # Domain-mean trends
    trends = {}
    periods = {"Annual": list(range(1, 13)), "Summer (JFM)": [1, 2, 3], "Winter (JAS)": [7, 8, 9]}
    ds_map = {"GLORYS": fpca_50["GLORYS"][0], "GLORYS_CL": fpca_50["GLORYS_CL"][0], "CMA": fpca_50["CMA"][0]}
    for period_name, months in periods.items():
        trends[period_name] = {}
        for name in ["GLORYS", "GLORYS_CL", "CMA"]:
            tr = trend_stats(ds_map[name], months)
            trends[period_name][name] = (float(tr.slope), float(tr.pvalue))

    # KERFIX / Figure 10 stats
    try:
        import gsw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("update_statistics_markdown.py requires gsw for KERFIX statistics.") from exc

    df_ker = pd.read_csv(ROOT / "data" / "kerfix.csv", sep=";")
    profile_id = df_ker["profileID"].astype(str)
    date_prefix = profile_id.str[:2].apply(lambda value: "19" + value if int(value) > 50 else "20" + value)
    df_ker["TIME"] = pd.to_datetime(date_prefix + profile_id.str[2:], format="%Y%m")
    df_ker["DEPTH"] = -gsw.conversions.z_from_p(df_ker["PRES"], lat=-50.6667, geo_strf_dyn_height=0, sea_surface_geopotential=0)
    df_ker["SA"] = gsw.SA_from_SP(df_ker["PSAL"], df_ker["PRES"], lon=68.4167, lat=-50.6667)
    df_ker["CT"] = gsw.CT_from_t(df_ker["SA"], df_ker["TEMP"], df_ker["PRES"])
    df_ker["sigma0"] = gsw.density.sigma0(df_ker["SA"], df_ker["CT"])
    ds_ker = df_ker.reset_index(drop=True).set_index(["TIME", "DEPTH"]).to_xarray()
    sigma0_10m = ds_ker.sel(DEPTH=10, method="nearest").sigma0
    hit = ((ds_ker.sigma0 - sigma0_10m) >= 0.03) & (ds_ker.DEPTH >= 10)
    has = hit.any(dim="DEPTH")
    ker_da = ds_ker.DEPTH.isel(DEPTH=hit.argmax(dim="DEPTH")).where(has)
    ker = xr.DataArray(ker_da.data, coords={"time": ds_ker["TIME"].values}, dims=["time"]).dropna(dim="time")

    lon_ker = 68.4167
    lat_ker = -50.6667
    ds_g_ker = open_gridded(ROOT, "GLORYS_gridded.nc").sel(longitude=lon_ker, latitude=lat_ker, method="nearest")
    glo = ds_g_ker["mld"].dropna(dim="time")

    ds_cma_rec = r_analysis_df("CMA_masked", project_root=ROOT)[0]
    ds_cma_clim = open_gridded(ROOT, "CMA_clim.nc").rename({"longitude": "long", "latitude": "lat"})
    cma_anom = ds_cma_rec["mld"].sel(long=lon_ker, lat=lat_ker, method="nearest")
    cma_clim = ds_cma_clim["mld"].sel(long=lon_ker, lat=lat_ker, method="nearest")
    cma_rec = (cma_anom.groupby("time.month") + cma_clim).dropna(dim="time")

    ds_cma_obs = open_gridded(ROOT, "CMA_gridded.nc").sel(longitude=lon_ker, latitude=lat_ker, method="nearest")
    cma_obs = ds_cma_obs["mld"].dropna(dim="time")

    def station_stats(da: xr.DataArray) -> dict[str, float | int | str]:
        t = pd.to_datetime(da[da.dims[0]].values)
        v = da.values.astype(float)
        s = pd.Series(v, index=t)
        x = t.year + (t.month - 0.5) / 12.0
        fit = linregress(x, v)
        return {
            "period": f"{t.min().strftime('%b %Y')}-{t.max().strftime('%b %Y')}",
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "annual": float(s.mean()),
            "jfm": float(s[s.index.month.isin([1, 2, 3])].mean()),
            "jas": float(s[s.index.month.isin([7, 8, 9])].mean()),
            "slope": float(fit.slope),
            "p": float(fit.pvalue),
        }

    st_ker = station_stats(ker)
    st_g = station_stats(glo)
    st_cma_rec = station_stats(cma_rec)
    st_cma_obs = station_stats(cma_obs)

    return f"""## Seasonal Cycle and Domain-Mean Anomalies

Statistics below use the products plotted in Figure 3: monthly climatology files and domain-mean anomaly files.

| Product | Climatology min (m) | Climatology max (m) | Seasonal amplitude (m) | Anomaly mean (m) | Anomaly std (m) |
|---|---:|---:|---:|---:|---:|
| GLORYS | {_fmt_f2(seasonal['GLORYS']['clim_min'])} | {_fmt_f2(seasonal['GLORYS']['clim_max'])} | {_fmt_f2(seasonal['GLORYS']['amp'])} | {_fmt_f2(seasonal['GLORYS']['anom_mean'])} | {_fmt_f2(seasonal['GLORYS']['anom_std'])} |
| GLORYS_CL | {_fmt_f2(seasonal['GLORYS_CL']['clim_min'])} | {_fmt_f2(seasonal['GLORYS_CL']['clim_max'])} | {_fmt_f2(seasonal['GLORYS_CL']['amp'])} | {_fmt_f2(seasonal['GLORYS_CL']['anom_mean'])} | {_fmt_f2(seasonal['GLORYS_CL']['anom_std'])} |
| CMA | {_fmt_f2(seasonal['CMA']['clim_min'])} | {_fmt_f2(seasonal['CMA']['clim_max'])} | {_fmt_f2(seasonal['CMA']['amp'])} | {_fmt_f2(seasonal['CMA']['anom_mean'])} | {_fmt_f2(seasonal['CMA']['anom_std'])} |

Domain-mean anomaly differences:

| Difference | Mean (m) | Std (m) | Min (m) | Max (m) |
|---|---:|---:|---:|---:|
| GLORYS_CL - GLORYS | {_fmt_f2(float(np.nanmean(d_cl_g)))} | {_fmt_f2(float(np.nanstd(d_cl_g)))} | {_fmt_f2(float(np.nanmin(d_cl_g)))} | {_fmt_f2(float(np.nanmax(d_cl_g)))} |
| CMA - GLORYS | {_fmt_f2(float(np.nanmean(d_cma_g)))} | {_fmt_f2(float(np.nanstd(d_cma_g)))} | {_fmt_f2(float(np.nanmin(d_cma_g)))} | {_fmt_f2(float(np.nanmax(d_cma_g)))} |

## fPCA/PACE Statistics

| Product | Retained modes | Mode 1 (%) | Mode 2 (%) | First 2 modes (%) | First 5 modes (%) |
|---|---:|---:|---:|---:|---:|
| GLORYS | {fpca_stats['GLORYS']['retained_modes']} | {_fmt_f2(fpca_stats['GLORYS']['mode1'])} | {_fmt_f2(fpca_stats['GLORYS']['mode2'])} | {_fmt_f2(fpca_stats['GLORYS']['first2'])} | {_fmt_f2(fpca_stats['GLORYS']['first5'])} |
| GLORYS_CL | {fpca_stats['GLORYS_CL']['retained_modes']} | {_fmt_f2(fpca_stats['GLORYS_CL']['mode1'])} | {_fmt_f2(fpca_stats['GLORYS_CL']['mode2'])} | {_fmt_f2(fpca_stats['GLORYS_CL']['first2'])} | {_fmt_f2(fpca_stats['GLORYS_CL']['first5'])} |
| CMA | {fpca_stats['CMA']['retained_modes']} | {_fmt_f2(fpca_stats['CMA']['mode1'])} | {_fmt_f2(fpca_stats['CMA']['mode2'])} | {_fmt_f2(fpca_stats['CMA']['first2'])} | {_fmt_f2(fpca_stats['CMA']['first5'])} |

Temporal-mode summary:

| Product | mean xi1 | std xi1 | mean xi2 | std xi2 |
|---|---:|---:|---:|---:|
| GLORYS | {_fmt_f3(fpca_stats['GLORYS']['phi1_mean'])} | {_fmt_f3(fpca_stats['GLORYS']['phi1_std'])} | {_fmt_f3(fpca_stats['GLORYS']['phi2_mean'])} | {_fmt_f3(fpca_stats['GLORYS']['phi2_std'])} |
| GLORYS_CL | {_fmt_f3(fpca_stats['GLORYS_CL']['phi1_mean'])} | {_fmt_f3(fpca_stats['GLORYS_CL']['phi1_std'])} | {_fmt_f3(fpca_stats['GLORYS_CL']['phi2_mean'])} | {_fmt_f3(fpca_stats['GLORYS_CL']['phi2_std'])} |
| CMA | {_fmt_f3(fpca_stats['CMA']['phi1_mean'])} | {_fmt_f3(fpca_stats['CMA']['phi1_std'])} | {_fmt_f3(fpca_stats['CMA']['phi2_mean'])} | {_fmt_f3(fpca_stats['CMA']['phi2_std'])} |

## Reconstruction Error

Figure 6 RMSE values are evaluated on the co-located `GLORYS_CL` sampling mask.

| Product | Mean RMSE (m) | Std RMSE (m) |
|---|---:|---:|
| GLORYS | {_fmt_f2(float(qe_g.mean().item()))} | {_fmt_f2(float(qe_g.std().item()))} |
| GLORYS_CL | {_fmt_f2(float(qe_cl.mean().item()))} | {_fmt_f2(float(qe_cl.std().item()))} |
| CMA | {_fmt_f2(float(qe_cma.mean().item()))} | {_fmt_f2(float(qe_cma.std().item()))} |

## Domain-Mean Trends

Slopes are in `m yr-1`, with p-values in parentheses.

| Period | GLORYS | GLORYS_CL | CMA |
|---|---:|---:|---:|
| Annual | {_fmt_f2(trends['Annual']['GLORYS'][0])} ({_fmt_f2(trends['Annual']['GLORYS'][1])}) | {_fmt_f2(trends['Annual']['GLORYS_CL'][0])} ({_fmt_f2(trends['Annual']['GLORYS_CL'][1])}) | {_fmt_f2(trends['Annual']['CMA'][0])} ({_fmt_f2(trends['Annual']['CMA'][1])}) |
| Summer (JFM) | {_fmt_f2(trends['Summer (JFM)']['GLORYS'][0])} ({_fmt_f2(trends['Summer (JFM)']['GLORYS'][1])}) | {_fmt_f2(trends['Summer (JFM)']['GLORYS_CL'][0])} ({_fmt_f2(trends['Summer (JFM)']['GLORYS_CL'][1])}) | {_fmt_f2(trends['Summer (JFM)']['CMA'][0])} ({_fmt_f2(trends['Summer (JFM)']['CMA'][1])}) |
| Winter (JAS) | {_fmt_f2(trends['Winter (JAS)']['GLORYS'][0])} ({_fmt_f2(trends['Winter (JAS)']['GLORYS'][1])}) | {_fmt_f2(trends['Winter (JAS)']['GLORYS_CL'][0])} ({_fmt_f2(trends['Winter (JAS)']['GLORYS_CL'][1])}) | {_fmt_f2(trends['Winter (JAS)']['CMA'][0])} ({_fmt_f2(trends['Winter (JAS)']['CMA'][1])}) |

## KERFIX / Figure 10

KERFIX station position used in Figure 10: 50.6667 S, 68.4167 E. The GLORYS grid point selected by nearest-neighbor interpolation is 50.5125 S, 68.4615 E.

KERFIX MLD is recomputed from `data/kerfix.csv` with the same `0.03 kg m-3` density threshold relative to 10 m used for CMA, GLORYS, GLORYS_CL, and the GLORYS section product.

| Product | Period | n | Mean MLD (m) | Std MLD (m) | Min (m) | Max (m) |
|---|---|---:|---:|---:|---:|---:|
| KERFIX | {st_ker['period']} | {st_ker['n']} | {_fmt_f2(st_ker['mean'])} | {_fmt_f2(st_ker['std'])} | {_fmt_f2(st_ker['min'])} | {_fmt_f2(st_ker['max'])} |
| GLORYS at KERFIX | {st_g['period']} | {st_g['n']} | {_fmt_f2(st_g['mean'])} | {_fmt_f2(st_g['std'])} | {_fmt_f2(st_g['min'])} | {_fmt_f2(st_g['max'])} |
| CMA reconstruction at KERFIX | {st_cma_rec['period']} | {st_cma_rec['n']} | {_fmt_f2(st_cma_rec['mean'])} | {_fmt_f2(st_cma_rec['std'])} | {_fmt_f2(st_cma_rec['min'])} | {_fmt_f2(st_cma_rec['max'])} |
| CMA observed at KERFIX grid cell | {st_cma_obs['period']} | {st_cma_obs['n']} | {_fmt_f2(st_cma_obs['mean'])} | {_fmt_f2(st_cma_obs['std'])} | {_fmt_f2(st_cma_obs['min'])} | {_fmt_f2(st_cma_obs['max'])} |

Seasonal means:

| Product | Annual mean (m) | Summer JFM (m) | Winter JAS (m) |
|---|---:|---:|---:|
| KERFIX | {_fmt_f2(st_ker['annual'])} | {_fmt_f2(st_ker['jfm'])} | {_fmt_f2(st_ker['jas'])} |
| GLORYS at KERFIX | {_fmt_f2(st_g['annual'])} | {_fmt_f2(st_g['jfm'])} | {_fmt_f2(st_g['jas'])} |
| CMA reconstruction at KERFIX | {_fmt_f2(st_cma_rec['annual'])} | {_fmt_f2(st_cma_rec['jfm'])} | {_fmt_f2(st_cma_rec['jas'])} |
| CMA observed at KERFIX grid cell | {_fmt_f2(st_cma_obs['annual'])} | {_fmt_f2(st_cma_obs['jfm'])} | {_fmt_f2(st_cma_obs['jas'])} |

Simple monthly linear fits at the station:

| Product | Slope (m yr-1) | p-value |
|---|---:|---:|
| KERFIX, 1990-1994 | {_fmt_f2(st_ker['slope'])} | {_fmt_f3(st_ker['p'])} |
| GLORYS at KERFIX, 2007-2023 | {_fmt_f2(st_g['slope'])} | {_fmt_f3(st_g['p'])} |
| CMA reconstruction at KERFIX, 2007-2023 | {_fmt_f2(st_cma_rec['slope'])} | {_fmt_f3(st_cma_rec['p'])} |
| CMA observed at KERFIX grid cell, 2008-2023 | {_fmt_f2(st_cma_obs['slope'])} | {_fmt_f3(st_cma_obs['p'])} |

Park et al. (1998) reported KERFIX MLD with a `0.02 sigma_theta` density-difference criterion. Those values are useful context for the historical station record, but they are not used in the manuscript statistics or Figure 10 comparison.
"""


def _replace_section(text: str, pattern: str, replacement: str) -> str:
    new_text, n = re.subn(pattern, replacement.strip() + "\n\n", text, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"Failed to replace section matching pattern: {pattern}")
    return new_text


def main() -> None:
    # First refresh the auto-managed gridded block and summary.
    update_gridded_blocks()

    text = STATISTICS_PATH.read_text(encoding="utf-8")
    text = _replace_section(
        text,
        r"## Source Data and Gridding\n.*?(?=<!-- BEGIN AUTO:GRIDDLED_MLD_STATS -->)",
        _source_block(),
    )
    text = _replace_section(
        text,
        r"## Seasonal Cycle and Domain-Mean Anomalies\n.*?(?=## Park et al\. \(1998\) Context)",
        _seasonal_fpca_rmse_trends_kerfix_block(),
    )
    STATISTICS_PATH.write_text(text, encoding="utf-8")
    print("Updated all computed sections in statistics.md.")


if __name__ == "__main__":
    main()
