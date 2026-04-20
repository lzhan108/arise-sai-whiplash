#!/usr/bin/env python
# seasonal_whiplash_all_scenarios.py
#
# Seasonal whiplash frequency for all scenarios
# Consistent with ensemble_analysis.ipynb Cell 9
# Method A for Delayed-2045: CTL window = 2045-2064
#
# Usage:
#   conda activate arise_spei
#   python seasonal_whiplash_all_scenarios.py

import xarray as xr
import numpy as np
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
members  = [f"{i:03d}" for i in range(1, 11)]

seasons = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

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

def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

# ------------------------------------------------------------------ #
# Compute seasonal whiplash per member
# ------------------------------------------------------------------ #
scenarios = {
    'SAI-1.5': ('SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064')),
    'Delayed': ('SPEI3_delayed2045_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045','2064')),
    'CTL_35':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'CTL_45':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2045','2064')),
}

# seasonal_data[scenario][season] = list of (lat,lon) arrays, one per member
seasonal_data = {sc: {s: [] for s in seasons} for sc in scenarios}

print("Computing seasonal whiplash...")
for sc_name, (fname_tmpl, varname, tslice) in scenarios.items():
    for mem in members:
        ds   = xr.open_dataset(save_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        wh   = np.abs(spei.diff(dim='time')) >= 2.0  # (time, lat, lon)

        for season, months in seasons.items():
            wh_s = wh.sel(time=wh.time.dt.month.isin(months)).mean(dim='time')
            seasonal_data[sc_name][season].append(wh_s.values)
        ds.close()
    print(f"  {sc_name} done.")

# Stack to (10, lat, lon) per scenario per season
for sc in scenarios:
    for s in seasons:
        seasonal_data[sc][s] = np.stack(seasonal_data[sc][s], axis=0)

# Load lat/lon
ds_ref = xr.open_dataset(save_dir + 'SPEI3_CTL_member001_CTLbaseline.nc')
lats = ds_ref.lat.values
lons = ds_ref.lon.values
ds_ref.close()

def to_xr(arr):
    return xr.DataArray(arr, dims=['member','lat','lon'],
                        coords={'member': np.arange(10), 'lat': lats, 'lon': lons})

# ------------------------------------------------------------------ #
# Global seasonal summary
# ------------------------------------------------------------------ #
print("\n=== Global mean whiplash frequency by season ===")
print(f"{'Scenario':<14}", end='')
for s in seasons:
    print(f"  {s:>7}", end='')
print()
print("-" * 50)

for sc in ['CTL_35', 'SAI-1.5', 'SAI-1.0', 'CTL_45', 'Delayed']:
    label = sc.replace('_35','(35-64)').replace('_45','(45-64)')
    print(f"{label:<16}", end='')
    for season in seasons:
        mean_val = seasonal_data[sc][season].mean() * 100
        print(f"  {mean_val:>6.2f}%", end='')
    print()

# ------------------------------------------------------------------ #
# Regional seasonal breakdown — SAI - CTL diff
# ------------------------------------------------------------------ #
for sc, ctl in [('SAI-1.5', 'CTL_35'), ('SAI-1.0', 'CTL_35'), ('Delayed', 'CTL_45')]:
    print(f"\n=== {sc} vs CTL — Regional seasonal diff (%) ===")
    print(f"{'Region':<18}", end='')
    for s in seasons:
        print(f"  {s:>10}", end='')
    print()
    print("-" * 62)

    for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
        # Build a dummy xarray to get the mask
        dummy = to_xr(seasonal_data[sc]['DJF'])
        mask  = region_mask(dummy, lat_s, lat_n, lon_w, lon_e)

        print(f"{rname:<18}", end='')
        for season in seasons:
            sai_arr = to_xr(seasonal_data[sc][season])
            ctl_arr = to_xr(seasonal_data[ctl][season])

            sai_reg = sai_arr.where(mask).mean(dim=['lat','lon']).values * 100
            ctl_reg = ctl_arr.where(mask).mean(dim=['lat','lon']).values * 100
            diff    = sai_reg - ctl_reg

            _, p = stats.ttest_1samp(diff, popmean=0)
            print(f"  {diff.mean():>+6.2f}%{sig(p):>3}", end='')
        print()

# ------------------------------------------------------------------ #
# Dominant season per region (season with largest SAI suppression)
# ------------------------------------------------------------------ #
print("\n=== Dominant season of SAI suppression per region ===")
print(f"{'Region':<18} {'SAI-1.5':>12} {'SAI-1.0':>12} {'Delayed':>12}")
print("-" * 58)

for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
    dummy = to_xr(seasonal_data['SAI-1.5']['DJF'])
    mask  = region_mask(dummy, lat_s, lat_n, lon_w, lon_e)

    dominant = {}
    for sc, ctl in [('SAI-1.5','CTL_35'), ('SAI-1.0','CTL_35'), ('Delayed','CTL_45')]:
        diffs = {}
        for season in seasons:
            sai_reg = to_xr(seasonal_data[sc][season]).where(mask).mean(dim=['lat','lon']).values.mean() * 100
            ctl_reg = to_xr(seasonal_data[ctl][season]).where(mask).mean(dim=['lat','lon']).values.mean() * 100
            diffs[season] = sai_reg - ctl_reg
        # Most negative = strongest suppression
        dominant[sc] = min(diffs, key=diffs.get) + f"({diffs[min(diffs, key=diffs.get)]:+.2f}%)"

    print(f"{rname:<18} {dominant['SAI-1.5']:>12} {dominant['SAI-1.0']:>12} {dominant['Delayed']:>12}")

print("\nDone.")
