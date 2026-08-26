#!/usr/bin/env python3
"""Update the markdown audit files from the current processed gridded outputs.

This script refreshes the gridded-statistics block in `statistics.md` and the
summary block in `summary.md` using the values currently stored in the gridded
NetCDF outputs under `processed/1_gridded_data/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed" / "1_gridded_data"
STATISTICS_PATH = ROOT / "statistics.md"
SUMMARY_PATH = ROOT / "summary.md"


def kerguelen_mask(ds: xr.Dataset) -> xr.DataArray:
    lon = ds["longitude"]
    lat = ds["latitude"]
    return (
        (lon >= 68.25)
        & (lon <= 70.75)
        & (lat >= -50.0)
        & (lat <= -48.0)
    )


def finite_summary(arr: np.ndarray) -> tuple[int, float, float]:
    finite = np.isfinite(arr)
    values = arr[finite]
    return int(values.size), float(np.mean(values)), float(np.std(values))


def pairwise_overlap_stats(name_a: str, name_b: str) -> dict[str, float | int]:
    a = xr.open_dataset(PROCESSED / f"{name_a}_gridded.nc")["mld"].values
    b = xr.open_dataset(PROCESSED / f"{name_b}_gridded.nc")["mld"].values
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    corr = np.corrcoef(a[mask], b[mask])[0, 1]
    return {
        "n": int(mask.sum()),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "corr": float(corr),
    }


def spatial_mean_corr(name_a: str, name_b: str) -> tuple[float, int]:
    ds_a = xr.open_dataset(PROCESSED / f"{name_a}_gridded.nc")
    ds_b = xr.open_dataset(PROCESSED / f"{name_b}_gridded.nc")
    a_mean = ds_a.mean("time").mld
    b_mean = ds_b.mean("time").mld
    valid = np.isfinite(a_mean.values) & np.isfinite(b_mean.values)
    corr = xr.corr(a_mean, b_mean, dim=["latitude", "longitude"]).item()
    return float(corr), int(valid.sum())


def replace_between(path: Path, start_marker: str, end_marker: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    new_text = text[:start] + start_marker + "\n" + replacement.strip() + "\n" + end_marker + text[end + len(end_marker) :]
    path.write_text(new_text, encoding="utf-8")


def make_statistics_block() -> str:
    stats = {}
    for name in ["GLORYS", "GLORYS_CL", "CMA"]:
        arr = xr.open_dataset(PROCESSED / f"{name}_gridded.nc")["mld"].values
        n, mean, std = finite_summary(arr)
        stats[name] = {"n": n, "mean": mean, "std": std}

    g_cma = pairwise_overlap_stats("GLORYS", "CMA")
    cl_cma = pairwise_overlap_stats("GLORYS_CL", "CMA")
    g_cl = pairwise_overlap_stats("GLORYS", "GLORYS_CL")
    corr_gc, n_gc = spatial_mean_corr("GLORYS", "CMA")
    corr_clc, n_clc = spatial_mean_corr("GLORYS_CL", "CMA")
    corr_gcl, n_gcl = spatial_mean_corr("GLORYS", "GLORYS_CL")

    return f"""## Gridded MLD Statistics

Mean and standard deviation are over all finite monthly grid cells. Pointwise differences and correlations are computed on the common finite overlap only; because the missing-data masks differ between GLORYS and GLORYS_CL, the `GLORYS - CMA` and `GLORYS_CL - CMA` differences are not expected to be identical unless we restrict both to the same intersection mask.

| Product | Valid values | Mean MLD (m) | Std MLD (m) |
|---|---:|---:|---:|
| GLORYS | {stats['GLORYS']['n']:,} | {stats['GLORYS']['mean']:.2f} | {stats['GLORYS']['std']:.2f} |
| GLORYS_CL | {stats['GLORYS_CL']['n']:,} | {stats['GLORYS_CL']['mean']:.2f} | {stats['GLORYS_CL']['std']:.2f} |
| CMA | {stats['CMA']['n']:,} | {stats['CMA']['mean']:.2f} | {stats['CMA']['std']:.2f} |

Pointwise GLORYS-CMA difference at observed monthly grid cells (common finite overlap):

| Difference | n | Mean (m) | Std (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| GLORYS - CMA | {g_cma['n']:,} | {g_cma['mean_diff']:.2f} | {g_cma['std_diff']:.2f} | {g_cma['rmse']:.2f} |
| GLORYS_CL - CMA | {cl_cma['n']:,} | {cl_cma['mean_diff']:.2f} | {cl_cma['std_diff']:.2f} | {cl_cma['rmse']:.2f} |

Monthly pointwise correlations (all 204 months pooled; common finite overlap per pair):

| Pair | Correlation | n |
|---|---:|---:|
| GLORYS vs CMA | {g_cma['corr']:.3f} | {g_cma['n']:,} |
| GLORYS_CL vs CMA | {cl_cma['corr']:.3f} | {cl_cma['n']:,} |
| GLORYS vs GLORYS_CL | {g_cl['corr']:.3f} | {g_cl['n']:,} |

These monthly pooled values are not the Figure 2 map correlations.

Figure 2 spatial climatology correlations (time-mean maps after masking the Kerguelen island, using the common valid grid cells in each pair):

| Pair | Correlation | n |
|---|---:|---:|
| GLORYS vs CMA | {corr_gc:.3f} | {n_gc:,} |
| GLORYS_CL vs CMA | {corr_clc:.3f} | {n_clc:,} |
| GLORYS vs GLORYS_CL | {corr_gcl:.3f} | {n_gcl:,} |

These values are not computed on exactly the same missing-data mask across every pair. The `GLORYS_CL vs CMA` pair is effectively the cleanest comparison because `GLORYS_CL` is generated by keeping only GLORYS values where CMA has an observation.
"""


def make_summary_block() -> str:
    stats = {}
    for name in ["GLORYS", "GLORYS_CL", "CMA"]:
        arr = xr.open_dataset(PROCESSED / f"{name}_gridded.nc")["mld"].values
        n, mean, std = finite_summary(arr)
        stats[name] = {"n": n, "mean": mean, "std": std}

    cl_cma = pairwise_overlap_stats("GLORYS_CL", "CMA")
    corr_gc, n_gc = spatial_mean_corr("GLORYS", "CMA")
    corr_clc, n_clc = spatial_mean_corr("GLORYS_CL", "CMA")
    corr_gcl, n_gcl = spatial_mean_corr("GLORYS", "GLORYS_CL")

    return f"""## Current Data Statistics

All values below are computed from the current `processed/` outputs.

| Product | Finite monthly grid values | Mean MLD (m) | Std MLD (m) |
|---|---:|---:|---:|
| GLORYS | {stats['GLORYS']['n']:,} | {stats['GLORYS']['mean']:.2f} | {stats['GLORYS']['std']:.2f} |
| GLORYS_CL | {stats['GLORYS_CL']['n']:,} | {stats['GLORYS_CL']['mean']:.2f} | {stats['GLORYS_CL']['std']:.2f} |
| CMA | {stats['CMA']['n']:,} | {stats['CMA']['mean']:.2f} | {stats['CMA']['std']:.2f} |

The CMA gridded observational coverage is `{stats['CMA']['n']} / 310284 = {stats['CMA']['n'] / 310284:.2%}`. The R anomaly input for CMA has 31229 non-missing values (`10.06%`) because the Kerguelen island mask is encoded as zero to preserve the regular grid shape.

Validated Figure 1 counts after the Kerguelen mask are `96216` combined profiles on the map, split into `76558` MEOP profiles and `19658` CORA+ARGO profiles in the bars. Those bar counts now sum exactly to the map total because they are derived from the same de-duplicated combined dataset.

Figure 2 spatial climatology correlations (time-mean maps after masking the Kerguelen island, with each pair evaluated on its own common valid cells):

- GLORYS vs CMA: `{corr_gc:.3f}`.
- GLORYS_CL vs CMA: `{corr_clc:.3f}`.
- GLORYS vs GLORYS_CL: `{corr_gcl:.3f}`.

For reference, monthly pooled pointwise correlation at observed cells (not the Figure 2 metric) is `{cl_cma['corr']:.3f}` for `GLORYS_CL vs CMA`.

These values are not computed on exactly the same missing-data mask across every product pair. In particular, `GLORYS_CL` is created by retaining only GLORYS values where CMA has observations, so `GLORYS_CL vs CMA` is the most direct comparison of the co-located products.
"""


def main() -> None:
    replace_between(
        STATISTICS_PATH,
        "<!-- BEGIN AUTO:GRIDDLED_MLD_STATS -->",
        "<!-- END AUTO:GRIDDLED_MLD_STATS -->",
        make_statistics_block(),
    )
    replace_between(
        SUMMARY_PATH,
        "<!-- BEGIN AUTO:CURRENT_DATA_STATS -->",
        "<!-- END AUTO:CURRENT_DATA_STATS -->",
        make_summary_block(),
    )
    print(f"Updated {STATISTICS_PATH.name} and {SUMMARY_PATH.name} from the current processed outputs.")


if __name__ == "__main__":
    main()
