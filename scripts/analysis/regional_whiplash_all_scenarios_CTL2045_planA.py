#!/usr/bin/env python
# regional_whiplash_all_scenarios_CTL2045_planA.py
#
# Regional whiplash breakdown — Method A
# Delayed-2045: SAI window = 2045-2064, CTL window = 2045-2064
# SAI-1.5, SAI-1.0: SAI window = 2035-2064, CTL window = 2035-2064
#
# Usage:
#   conda activate arise_spei
#   python regional_whiplash_all_scenarios_CTL2045_planA.py

import xarray as xr
import numpy as np
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
members  = [f"{i:03d}" for i in range(1, 11)]

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

def calculate_whiplash(spei, threshold=2.0):
    delta = spei.diff(dim='time')
    return np.abs(delta) >= threshold

# ------------------------------------------------------------------ #
# Build per-member regional frequency for each scenario
# ------------------------------------------------------------------ #

# SAI-1.5 and SAI-1.0: CTL 2035-2064 (unchanged)
ens_sai15 = xr.open_dataarray(save_dir + 'whiplash_freq_SAI_allMembers_SPEI3.nc')
ens_sai10 = xr.open_dataarray(save_dir + 'whiplash_freq_SAI1p0_allMembers_SPEI3_CTL2035.nc')
ens_ctl35 = xr.open_dataarray(save_dir + 'whiplash_freq_CTL2035_allMembers_SPEI3.nc')

# Delayed-2045: recompute CTL using 2045-2064 window
print("Computing Delayed-2045 with CTL 2045-2064 (Method A)...")
wh_delayed_list  = []
wh_ctl45_list    = []

for mem in members:
    delayed = xr.open_dataarray(save_dir + f'SPEI3_delayed2045_member{mem}_CTLbaseline.nc')
    ctl     = xr.open_dataarray(save_dir + f'SPEI3_CTL_member{mem}_CTLbaseline.nc')

    wh_del   = calculate_whiplash(delayed.sel(time=slice('2045', '2064'))).mean(dim='time')
    wh_ctl45 = calculate_whiplash(ctl.sel(time=slice('2045', '2064'))).mean(dim='time')

    wh_delayed_list.append(wh_del)
    wh_ctl45_list.append(wh_ctl45)

ens_delayed_A = xr.concat(wh_delayed_list, dim='member')
ens_ctl45     = xr.concat(wh_ctl45_list,   dim='member')

print(f"Global: Delayed={float(ens_delayed_A.mean())*100:.2f}%  CTL(2045-2064)={float(ens_ctl45.mean())*100:.2f}%\n")

# ------------------------------------------------------------------ #
# Regional breakdown
# ------------------------------------------------------------------ #
def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

print("Method A — Delayed-2045 uses CTL 2045-2064; SAI-1.5/1.0 use CTL 2035-2064")
print(f"\n{'Region':<18} {'CTL-35':>8} {'CTL-45':>8} {'SAI-1.5':>9} {'SAI-1.0':>9} {'Delayed':>9}   "
      f"{'Diff-1.5':>10} {'Diff-1.0':>10} {'Diff-Del(A)':>12}")
print("-" * 110)

results_A = {}

for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
    mask = region_mask(ens_ctl35, lat_s, lat_n, lon_w, lon_e)

    ctl35_reg = ens_ctl35.where(mask).mean(dim=['lat','lon']).values * 100
    ctl45_reg = ens_ctl45.where(mask).mean(dim=['lat','lon']).values * 100
    s15_reg   = ens_sai15.where(mask).mean(dim=['lat','lon']).values * 100
    s10_reg   = ens_sai10.where(mask).mean(dim=['lat','lon']).values * 100
    del_reg   = ens_delayed_A.where(mask).mean(dim=['lat','lon']).values * 100

    results_A[rname] = {
        'CTL35': ctl35_reg, 'CTL45': ctl45_reg,
        'SAI-1.5': s15_reg, 'SAI-1.0': s10_reg, 'Delayed': del_reg
    }

    _, p15  = stats.ttest_1samp(s15_reg  - ctl35_reg, popmean=0)
    _, p10  = stats.ttest_1samp(s10_reg  - ctl35_reg, popmean=0)
    _, pdel = stats.ttest_1samp(del_reg  - ctl45_reg, popmean=0)  # Method A: Delayed vs CTL45

    print(f"{rname:<18} {ctl35_reg.mean():>8.2f}% {ctl45_reg.mean():>8.2f}% "
          f"{s15_reg.mean():>8.2f}% {s10_reg.mean():>8.2f}% {del_reg.mean():>8.2f}%   "
          f"{s15_reg.mean()-ctl35_reg.mean():>+8.2f}%{sig(p15):>3}  "
          f"{s10_reg.mean()-ctl35_reg.mean():>+8.2f}%{sig(p10):>3}  "
          f"{del_reg.mean()-ctl45_reg.mean():>+10.2f}%{sig(pdel):>3}")

# ------------------------------------------------------------------ #
# Offset ratio comparison: A vs B
# ------------------------------------------------------------------ #
CTL_baseline = 5.35

print(f"\n{'':=<110}")
print("Offset ratio comparison — Method A vs Method B")
print(f"{'Region':<18} {'Off-1.5':>9} {'Off-1.0':>9} {'Off-Del(A)':>12} {'Off-Del(B)':>12}  {'Diff(A-B)':>10}")
print("-" * 75)

for rname, row in results_A.items():
    ctl35 = row['CTL35'].mean()
    ctl45 = row['CTL45'].mean()
    s15   = row['SAI-1.5'].mean()
    s10   = row['SAI-1.0'].mean()
    delv  = row['Delayed'].mean()

    denom35 = ctl35 - CTL_baseline
    denom45 = ctl45 - CTL_baseline

    off15  = (ctl35 - s15)  / denom35 * 100 if abs(denom35) > 0.01 else float('nan')
    off10  = (ctl35 - s10)  / denom35 * 100 if abs(denom35) > 0.01 else float('nan')
    offA   = (ctl45 - delv) / denom45 * 100 if abs(denom45) > 0.01 else float('nan')  # Method A
    offB   = (ctl35 - delv) / denom35 * 100 if abs(denom35) > 0.01 else float('nan')  # Method B

    print(f"{rname:<18} {off15:>9.0f}%  {off10:>9.0f}%  {offA:>11.0f}%  {offB:>11.0f}%  {offA-offB:>+9.0f}%")

print("\nDone.")
