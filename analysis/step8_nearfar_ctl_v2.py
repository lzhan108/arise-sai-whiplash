#!/usr/bin/env python
# step8_nearfar_ctl_v2.py
#
# CTL whiplash frequency in the near-future (2035-2049) vs far-future
# (2050-2064) sub-periods for MED and WCA, to support the Figure 6
# discussion paragraph ("CTL frequency rises by approximately X pp
# from near-future to far-future").
#
# Usage:
#   conda activate arise_spei
#   python step8_nearfar_ctl_v2.py

import xarray as xr
import numpy as np
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

periods = {
    'Near-future (2035-2049)': slice('2035', '2049'),
    'Far-future (2050-2064)':  slice('2050', '2064'),
}

for rname in ['Mediterranean', 'W C Africa']:
    lat_s, lat_n, lon_w, lon_e = REGIONS[rname]
    print(f"\n=== {rname} (CTL) ===")
    vals = {}
    for period_label, tslice in periods.items():
        member_vals = []
        for mem in members:
            da = xr.open_dataarray(spei_dir + f'SPEI3_CTL_v2_member{mem}_CTLbaseline.nc')
            wh = (np.abs(da.diff(dim='time').sel(time=tslice)) >= 2.0).mean(dim='time')
            rmask = region_mask(wh, lat_s, lat_n, lon_w, lon_e) & land_mask
            member_vals.append(float(area_weighted_mean(wh, rmask)) * 100)
        vals[period_label] = np.mean(member_vals)
        print(f"  {period_label}: {vals[period_label]:.2f}%")
    diff = vals['Far-future (2050-2064)'] - vals['Near-future (2035-2049)']
    print(f"  Increase (far - near): {diff:+.2f} pp")

print("\nDone.")
