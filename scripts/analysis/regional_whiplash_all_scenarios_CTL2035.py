#!/usr/bin/env python
# regional_whiplash_all_scenarios_CTL2035.py
#
# Regional whiplash breakdown for all 4 scenarios
# UNIFIED CTL baseline: 2035-2064 for all scenarios (Method B)
#
# Delayed-2045: SAI window = 2045-2064, CTL window = 2035-2064
# SAI-1.5, SAI-1.0: SAI window = 2035-2064, CTL window = 2035-2064
#
# Usage:
#   conda activate arise_spei
#   python regional_whiplash_all_scenarios_CTL2035.py

import xarray as xr
import numpy as np
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"

regions = {
    'W N America':   (30,  60,  230, 260),
    'NE Brazil':     (-15,  5,  315, 345),
    'Mediterranean': (30,  45,  350,  40),
    'W C Africa':    (-10, 15,    5,  30),
    'W Amazon':      (-15,  5,  280, 310),
    'N Australia':   (-25, -10, 120, 145),
}

def region_mask(da, lat_s, lat_n, lon_w, lon_e):
    if lon_w > lon_e:
        return ((da.lat >= lat_s) & (da.lat <= lat_n) &
                ((da.lon >= lon_w) | (da.lon <= lon_e)))
    else:
        return ((da.lat >= lat_s) & (da.lat <= lat_n) &
                (da.lon >= lon_w) & (da.lon <= lon_e))

# Load ensemble files — all using CTL 2035-2064
ens = {
    'CTL':     xr.open_dataarray(save_dir + 'whiplash_freq_CTL2035_allMembers_SPEI3.nc'),
    'SAI-1.5': xr.open_dataarray(save_dir + 'whiplash_freq_SAI_allMembers_SPEI3.nc'),
    'SAI-1.0': xr.open_dataarray(save_dir + 'whiplash_freq_SAI1p0_allMembers_SPEI3_CTL2035.nc'),
    'Delayed': xr.open_dataarray(save_dir + 'whiplash_freq_delayed2045_allMembers_SPEI3_CTL2035.nc'),
}

print("Method B — Unified CTL baseline: 2035-2064")
print(f"\n{'Region':<18} {'CTL':>7} {'SAI-1.5':>9} {'SAI-1.0':>9} {'Delayed':>9}   "
      f"{'Diff-1.5':>10} {'Diff-1.0':>10} {'Diff-Del':>10}")
print("-" * 100)

results = {}

for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
    row = {}
    for scenario, da in ens.items():
        mask = region_mask(da, lat_s, lat_n, lon_w, lon_e)
        reg  = da.where(mask).mean(dim=['lat', 'lon']) * 100
        row[scenario] = reg.values
    results[rname] = row

    ctl_mean = row['CTL'].mean()
    s15_mean = row['SAI-1.5'].mean()
    s10_mean = row['SAI-1.0'].mean()
    del_mean = row['Delayed'].mean()

    _, p15  = stats.ttest_1samp(row['SAI-1.5'] - row['CTL'], popmean=0)
    _, p10  = stats.ttest_1samp(row['SAI-1.0'] - row['CTL'], popmean=0)
    _, pdel = stats.ttest_1samp(row['Delayed']  - row['CTL'], popmean=0)

    def sig(p):
        if p < 0.001: return '***'
        elif p < 0.01:  return '**'
        elif p < 0.05:  return '*'
        else:           return 'ns'

    print(f"{rname:<18} {ctl_mean:>7.2f}% {s15_mean:>8.2f}% {s10_mean:>8.2f}% {del_mean:>8.2f}%   "
          f"{s15_mean-ctl_mean:>+8.2f}%{sig(p15):>3}  "
          f"{s10_mean-ctl_mean:>+8.2f}%{sig(p10):>3}  "
          f"{del_mean-ctl_mean:>+8.2f}%{sig(pdel):>3}")

# Offset ratio
# (CTL_future - SAI) / (CTL_future - CTL_baseline) * 100
CTL_baseline = 5.35  # global CTL baseline 2015-2034

print(f"\n{'Region':<18} {'Offset-1.5':>12} {'Offset-1.0':>12} {'Offset-Del':>12}")
print("-" * 60)

for rname, row in results.items():
    ctl_mean = row['CTL'].mean()
    s15_mean = row['SAI-1.5'].mean()
    s10_mean = row['SAI-1.0'].mean()
    del_mean = row['Delayed'].mean()

    denom = ctl_mean - CTL_baseline
    if abs(denom) > 0.01:
        off15  = (ctl_mean - s15_mean) / denom * 100
        off10  = (ctl_mean - s10_mean) / denom * 100
        offdel = (ctl_mean - del_mean) / denom * 100
    else:
        off15 = off10 = offdel = float('nan')

    print(f"{rname:<18} {off15:>11.0f}%  {off10:>11.0f}%  {offdel:>11.0f}%")

print("\nDone.")
