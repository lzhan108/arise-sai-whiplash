#!/usr/bin/env python
# directional_whiplash_all_scenarios.py
#
# D->W and W->D whiplash frequency for all scenarios
# Consistent with ensemble_analysis.ipynb calc_dw_wd_correct()
#
# Method A for Delayed-2045: CTL window = 2045-2064
#
# Usage:
#   conda activate arise_spei
#   python directional_whiplash_all_scenarios.py

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

def calc_dw_wd(spei_da, time_slice, threshold=2.0):
    """
    D->W: ΔSPEI >= +2.0
    W->D: ΔSPEI <= -2.0
    Returns frequency as fraction (not %)
    """
    sp    = spei_da.sel(time=time_slice)
    delta = sp.diff(dim='time').values
    total = delta.shape[0]
    dw = (delta >=  threshold).sum(axis=0) / total
    wd = (delta <= -threshold).sum(axis=0) / total
    return dw, wd  # (lat, lon)

def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

# ------------------------------------------------------------------ #
# Compute per-member D->W and W->D for each scenario
# ------------------------------------------------------------------ #
print("Computing directional whiplash for all members...")

scenarios = {
    'SAI-1.5'  : ('SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    'SAI-1.0'  : ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064')),
    'Delayed'  : ('SPEI3_delayed2045_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045','2064')),
    'CTL_35'   : ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'CTL_45'   : ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2045','2064')),
}

ens_dw = {}
ens_wd = {}

for sc_name, (fname_template, varname, tslice) in scenarios.items():
    dw_list, wd_list = [], []
    for mem in members:
        ds  = xr.open_dataset(save_dir + fname_template.format(mem))
        spei = ds[varname]
        dw, wd = calc_dw_wd(spei, tslice)
        dw_list.append(dw)
        wd_list.append(wd)
        ds.close()
    ens_dw[sc_name] = np.stack(dw_list, axis=0)  # (10, lat, lon)
    ens_wd[sc_name] = np.stack(wd_list, axis=0)
    print(f"  {sc_name} done.")

# Load lat/lon from a reference file
ds_ref = xr.open_dataset(save_dir + 'SPEI3_CTL_member001_CTLbaseline.nc')
lats = ds_ref.lat.values
lons = ds_ref.lon.values
ds_ref.close()

# ------------------------------------------------------------------ #
# Save ensemble arrays as xarray for regional analysis
# ------------------------------------------------------------------ #
def to_xarray(arr, lats, lons):
    return xr.DataArray(arr, dims=['member','lat','lon'],
                        coords={'member': np.arange(10), 'lat': lats, 'lon': lons})

xr_dw = {k: to_xarray(v, lats, lons) for k, v in ens_dw.items()}
xr_wd = {k: to_xarray(v, lats, lons) for k, v in ens_wd.items()}

# ------------------------------------------------------------------ #
# Global mean summary
# ------------------------------------------------------------------ #
print("\n=== Global mean directional whiplash frequency ===")
print(f"{'Scenario':<12} {'D->W':>8} {'W->D':>8} {'Total':>8}")
print("-" * 42)

for sc in ['CTL_35', 'SAI-1.5', 'SAI-1.0', 'CTL_45', 'Delayed']:
    dw_mean = float(xr_dw[sc].mean()) * 100
    wd_mean = float(xr_wd[sc].mean()) * 100
    label = sc.replace('_35','(2035-64)').replace('_45','(2045-64)')
    print(f"{label:<16} {dw_mean:>7.2f}%  {wd_mean:>7.2f}%  {dw_mean+wd_mean:>7.2f}%")

# ------------------------------------------------------------------ #
# Regional breakdown
# ------------------------------------------------------------------ #
print("\n=== Regional D->W frequency ===")
print(f"{'Region':<18} {'CTL':>7} {'SAI-1.5':>9} {'SAI-1.0':>9} {'Delayed':>9}  "
      f"{'Diff-1.5':>10} {'Diff-1.0':>10} {'Diff-Del':>10}")
print("-" * 100)

for direction, xr_dict in [('D->W', xr_dw), ('W->D', xr_wd)]:
    print(f"\n--- {direction} ---")
    print(f"{'Region':<18} {'CTL':>7} {'SAI-1.5':>9} {'SAI-1.0':>9} {'Delayed':>9}  "
          f"{'Diff-1.5':>10} {'Diff-1.0':>10} {'Diff-Del(A)':>12}")
    print("-" * 105)

    for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
        mask = region_mask(xr_dict['CTL_35'], lat_s, lat_n, lon_w, lon_e)

        def reg_mean(sc):
            return xr_dict[sc].where(mask).mean(dim=['lat','lon']).values * 100

        ctl35 = reg_mean('CTL_35')
        ctl45 = reg_mean('CTL_45')
        s15   = reg_mean('SAI-1.5')
        s10   = reg_mean('SAI-1.0')
        deld  = reg_mean('Delayed')

        _, p15  = stats.ttest_1samp(s15   - ctl35, popmean=0)
        _, p10  = stats.ttest_1samp(s10   - ctl35, popmean=0)
        _, pdel = stats.ttest_1samp(deld  - ctl45, popmean=0)  # Method A

        print(f"{rname:<18} {ctl35.mean():>7.2f}% {s15.mean():>8.2f}% "
              f"{s10.mean():>8.2f}% {deld.mean():>8.2f}%  "
              f"{s15.mean()-ctl35.mean():>+8.2f}%{sig(p15):>3}  "
              f"{s10.mean()-ctl35.mean():>+8.2f}%{sig(p10):>3}  "
              f"{deld.mean()-ctl45.mean():>+10.2f}%{sig(pdel):>3}")

# ------------------------------------------------------------------ #
# Symmetry check: D->W vs W->D
# ------------------------------------------------------------------ #
print("\n=== Symmetry check: D->W vs W->D (SAI - CTL difference) ===")
print(f"{'Scenario':<12} {'Global D->W diff':>18} {'Global W->D diff':>18} {'Symmetric?':>12}")
print("-" * 65)

for sc, ctl in [('SAI-1.5', 'CTL_35'), ('SAI-1.0', 'CTL_35'), ('Delayed', 'CTL_45')]:
    dw_diff = float((xr_dw[sc] - xr_dw[ctl]).mean()) * 100
    wd_diff = float((xr_wd[sc] - xr_wd[ctl]).mean()) * 100
    sym = 'Yes' if abs(abs(dw_diff) - abs(wd_diff)) < 0.05 else 'No'
    print(f"{sc:<12} {dw_diff:>+17.3f}%  {wd_diff:>+17.3f}%  {sym:>12}")

print("\nDone.")
