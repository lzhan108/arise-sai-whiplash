#!/usr/bin/env python
# step5_table2_v2.py
#
# Compute global mean whiplash frequency for Table 2 (v2):
# CTL baseline (2015-2034), CTL future (2035-2064),
# SAI-1.5, SAI-1.0, Delayed-2045.
#
# v2 changes vs the original compute_table2_global_means.py:
#   - Land mask applied (LANDFRAC>0.5, desert<80%, glacier<80%),
#     excluding ocean, hyperarid desert, glaciers, and Antarctica
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#   - All four scenarios (including CTL baseline) now use SPEI3
#     computed with FAO-56 PET and each member's own CTL baseline
#
# Usage:
#   conda activate arise_spei
#   python step5_table2_v2.py

import xarray as xr
import numpy as np
from utils_v2 import load_land_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"

land_mask = load_land_mask()

# ------------------------------------------------------------------ #
# Load pre-aggregated ensemble whiplash frequency files
# (produced by step4_ensemble_whiplash_v2.py)
# ------------------------------------------------------------------ #
ens_base   = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_baseline2015_v2_allMembers.nc')
ens_ctl    = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_v2_allMembers.nc')
ens_sai15  = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI15_v2_allMembers.nc')
ens_sai10  = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI1p0_v2_allMembers.nc')
ens_del    = xr.open_dataarray(spei_dir + 'whiplash_freq_delayed2045_v2_allMembers.nc')

baseline_mean = float(area_weighted_mean(ens_base, land_mask).mean()) * 100
ctl_mean      = float(area_weighted_mean(ens_ctl, land_mask).mean()) * 100
sai15_mean    = float(area_weighted_mean(ens_sai15, land_mask).mean()) * 100
sai10_mean    = float(area_weighted_mean(ens_sai10, land_mask).mean()) * 100
del_mean      = float(area_weighted_mean(ens_del, land_mask).mean()) * 100

# ------------------------------------------------------------------ #
# Offset ratios
# ------------------------------------------------------------------ #
denom = ctl_mean - baseline_mean

off15  = (ctl_mean - sai15_mean) / denom * 100
off10  = (ctl_mean - sai10_mean) / denom * 100
offdel = (ctl_mean - del_mean)   / denom * 100

# ------------------------------------------------------------------ #
# Print Table 2 (v2)
# ------------------------------------------------------------------ #
print(f"{'Scenario':<25} {'SAI freq (%)':>12} {'CTL freq (%)':>12} {'Diff (pp)':>10} {'Offset (%)':>10}")
print("-" * 75)
print(f"{'CTL baseline (2015-2034)':<25} {baseline_mean:>12.2f} {'-':>12} {'-':>10} {'-':>10}")
print(f"{'CTL future (2035-2064)':<25} {ctl_mean:>12.2f} {'-':>12} {ctl_mean-baseline_mean:>+10.2f} {'-':>10}")
print(f"{'ARISE-SAI-1.5':<25} {sai15_mean:>12.2f} {ctl_mean:>12.2f} {sai15_mean-ctl_mean:>+10.2f} {off15:>9.0f}%")
print(f"{'ARISE-SAI-1.0':<25} {sai10_mean:>12.2f} {ctl_mean:>12.2f} {sai10_mean-ctl_mean:>+10.2f} {off10:>9.0f}%")
print(f"{'ARISE-SAI-2045':<25} {del_mean:>12.2f} {ctl_mean:>12.2f} {del_mean-ctl_mean:>+10.2f} {offdel:>9.0f}%")
