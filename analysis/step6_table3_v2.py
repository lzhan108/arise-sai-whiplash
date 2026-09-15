#!/usr/bin/env python
# step6_table3_v2.py
#
# Compute regional whiplash frequency and offset ratios for Table 3 (v2).
#
# v2 changes vs the original compute_table3_regional.py:
#   - Land mask applied within each region box (excludes ocean,
#     hyperarid desert, glaciers; in practice only W N America has
#     any desert grid cells among the six regions, ~9.5% of its land
#     cells, so the other five regions are essentially unaffected)
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#   - CTL baseline (2015-2034) computed per-region with the same
#     land mask + area weighting, replacing the old approach of
#     re-deriving whiplash from raw per-member SPEI3 files
#
# Usage:
#   conda activate arise_spei
#   python step6_table3_v2.py

import xarray as xr
import numpy as np
from scipy import stats
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()


def get_sig(sai, ctl):
    _, p = stats.ttest_1samp(np.array(sai) - np.array(ctl), popmean=0)
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


# ------------------------------------------------------------------ #
# Load pre-aggregated ensemble whiplash frequency files
# ------------------------------------------------------------------ #
ens_ctl     = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_v2_allMembers.nc')
ens_sai15   = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI15_v2_allMembers.nc')
ens_sai10   = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI1p0_v2_allMembers.nc')
ens_delayed = xr.open_dataarray(spei_dir + 'whiplash_freq_delayed2045_v2_allMembers.nc')

# ------------------------------------------------------------------ #
# CTL baseline (2015-2034), per region, per member
# ------------------------------------------------------------------ #
base_list = {r: [] for r in REGIONS}
for mem in members:
    ctl = xr.open_dataarray(spei_dir + f'SPEI3_CTL_v2_member{mem}_CTLbaseline.nc')
    d   = ctl.diff(dim='time')
    wh  = (np.abs(d.sel(time=slice('2015', '2034'))) >= 2.0).mean(dim='time')
    for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
        rmask = region_mask(wh, lat_s, lat_n, lon_w, lon_e) & land_mask
        base_list[rname].append(float(area_weighted_mean(wh, rmask)) * 100)
    ctl.close()

# ------------------------------------------------------------------ #
# Print Table 3 (v2)
# ------------------------------------------------------------------ #
print("\nTable 3 (v2). Regional whiplash frequency and offset ratios.")
print(f"{'Region':<18} {'CTL':>6} {'S15':>6} {'Diff15':>8} {'Agr15':>6} {'Off15':>7} "
      f"{'S10':>6} {'Diff10':>8} {'Agr10':>6} {'Off10':>7} "
      f"{'Del':>6} {'DiffD':>8} {'AgrD':>6} {'OffD':>7}")
print("-" * 130)

for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
    rmask  = region_mask(ens_ctl, lat_s, lat_n, lon_w, lon_e) & land_mask
    ctl_v  = area_weighted_mean(ens_ctl, rmask).values * 100
    s15_v  = area_weighted_mean(ens_sai15, rmask).values * 100
    s10_v  = area_weighted_mean(ens_sai10, rmask).values * 100
    del_v  = area_weighted_mean(ens_delayed, rmask).values * 100
    base   = np.mean(base_list[rname])
    denom  = np.mean(ctl_v) - base
    if abs(denom) < 1e-6:
        denom = 1e-6

    diff15 = np.mean(s15_v) - np.mean(ctl_v)
    diff10 = np.mean(s10_v) - np.mean(ctl_v)
    diffd  = np.mean(del_v) - np.mean(ctl_v)

    off15  = (np.mean(ctl_v) - np.mean(s15_v)) / denom * 100
    off10  = (np.mean(ctl_v) - np.mean(s10_v)) / denom * 100
    offd   = (np.mean(ctl_v) - np.mean(del_v)) / denom * 100

    sig15  = get_sig(s15_v, ctl_v)
    sig10  = get_sig(s10_v, ctl_v)
    sigd   = get_sig(del_v, ctl_v)

    agr15  = np.sum(s15_v - ctl_v < 0)
    agr10  = np.sum(s10_v - ctl_v < 0)
    agrd   = np.sum(del_v - ctl_v < 0)

    print(f"{rname:<18} {np.mean(ctl_v):>6.2f} {np.mean(s15_v):>6.2f} {diff15:>+7.2f}{sig15:>3} "
          f"{agr15:>5}/10 {off15:>6.0f}% {np.mean(s10_v):>6.2f} {diff10:>+7.2f}{sig10:>3} "
          f"{agr10:>5}/10 {off10:>6.0f}% {np.mean(del_v):>6.2f} {diffd:>+7.2f}{sigd:>3} "
          f"{agrd:>5}/10 {offd:>6.0f}%")
