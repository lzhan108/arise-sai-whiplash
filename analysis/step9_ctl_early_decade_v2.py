#!/usr/bin/env python
# step9_ctl_early_decade_v2.py
#
# CTL whiplash frequency change during 2035-2044 (the decade before
# ARISE-SAI-2045 deployment begins) for all six regions, using the
# same 10-year rolling median methodology as Figure 6, to support the
# Figure 6 discussion paragraph ("whiplash frequency follows the CTL
# trajectory during 2035-2044, increasing by X pp depending on region").
#
# Usage:
#   conda activate arise_spei
#   python step9_ctl_early_decade_v2.py

import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter1d
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()


def rolling_stats(data_2d, window=10):
    rolled = np.array([
        uniform_filter1d(data_2d[m], size=window, mode='reflect')
        for m in range(data_2d.shape[0])
    ])
    return np.median(rolled, axis=0)


print("Computing CTL annual whiplash frequency per region (2035-2064)...")

annual_data = {r: [] for r in REGIONS}

for mem in members:
    da = xr.open_dataarray(spei_dir + f'SPEI3_CTL_v2_member{mem}_CTLbaseline.nc')
    spei = da.sel(time=slice('2035', '2064'))
    wh = np.abs(spei.diff(dim='time')) >= 2.0
    years = np.unique(wh.time.dt.year.values)
    for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
        rmask = region_mask(wh, lat_s, lat_n, lon_w, lon_e) & land_mask
        annual_vals = []
        for yr in years:
            wh_yr = wh.sel(time=wh.time.dt.year == yr).mean(dim='time')
            annual_vals.append(float(area_weighted_mean(wh_yr, rmask)) * 100)
        annual_data[rname].append(annual_vals)

for rname in REGIONS:
    annual_data[rname] = np.array(annual_data[rname])

years_full = np.arange(2035, 2065)
idx_2035 = np.where(years_full == 2035)[0][0]
idx_2044 = np.where(years_full == 2044)[0][0]

print("\n=== CTL rolling median whiplash frequency: 2035 vs 2044 ===")
diffs = []
for rname in REGIONS:
    med = rolling_stats(annual_data[rname], window=10)
    v2035 = med[idx_2035]
    v2044 = med[idx_2044]
    diff = v2044 - v2035
    diffs.append(diff)
    print(f"  {rname:<18} 2035={v2035:.2f}%  2044={v2044:.2f}%  diff={diff:+.2f} pp")

print(f"\nRange across regions: {min(diffs):+.2f} to {max(diffs):+.2f} pp")
print("Done.")
