# Data Inputs

Expected local input files:

- `CORA_MEOP_ARGO_2026.nc`
- `MEOP_2026.nc` for Figure 1
- `CORA_2026.nc` for Figure 1
- `ARGO_2026.nc` for Figure 1
- `GLORYS_2026.nc`
- `GEBCO_ker_large.nc`
- `fronts_62985.nc`
- `kerfix.csv` for the KERFIX figure workflow
- `GLORYS_1000m_section_timemean.nc` for Figure 9

The large raw NetCDF files are ignored by Git. Keep them locally in this directory.
`GLORYS_1000m_section_timemean.nc` is generated from `GLORYS_1000m_section_plot.nc`
with `1_data_processing/process_GLORYS_section.py`; the large source file is not
needed after the time-mean file has been created.

All derived MLD products use the common `0.03 kg m-3` density threshold relative
to 10 m, including GLORYS, GLORYS_CL, the GLORYS section product, CMA, and
KERFIX.

Appendix Figures A1-A5 require sampling outputs from
`2_compute_fPCA_R/script_PCA_GLORYS_random_sampling_2026.R` before running
their figure scripts.
