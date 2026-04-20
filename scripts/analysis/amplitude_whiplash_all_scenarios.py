#!/usr/bin/env python
# amplitude_whiplash_all_scenarios.py
#
# Whiplash amplitude analysis: mean |ΔSPEI| conditional on whiplash events
# Consistent with ensemble_analysis.ipynb Cell 11
# Method A for Delayed-2045: CTL window = 2045-2064
#
# Usage:
#   conda activate arise_spei
#   python amplitude_whiplash_all_scenarios.py

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

def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

# ------------------------------------------------------------------ #
# Compute amplitude per member
# amplitude = mean |ΔSPEI| only when |ΔSPEI| >= 2.0
# ------------------------------------------------------------------ #
scenarios = {
    'SAI-1.5': ('SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064')),
    'Delayed': ('SPEI3_delayed2045_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045','2064')),
    'CTL_35':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'CTL_45':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2045','2064')),
}

amp_data = {sc: [] for sc in scenarios}

print("Computing whiplash amplitude...")
for sc_name, (fname_tmpl, varname, tslice) in scenarios.items():
    for mem in members:
        ds   = xr.open_dataset(save_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = np.abs(spei.diff(dim='time'))
        # Conditional amplitude: only whiplash events
        amp = delta.where(delta >= 2.0).mean(dim='time')
        amp_data[sc_name].append(amp.values)
        ds.close()
    print(f"  {sc_name} done.")

# Stack to (10, lat, lon)
for sc in scenarios:
    amp_data[sc] = np.stack(amp_data[sc], axis=0)

# Load lat/lon
ds_ref = xr.open_dataset(save_dir + 'SPEI3_CTL_member001_CTLbaseline.nc')
lats = ds_ref.lat.values
lons = ds_ref.lon.values
ds_ref.close()

def to_xr(arr):
    return xr.DataArray(arr, dims=['member','lat','lon'],
                        coords={'member': np.arange(10), 'lat': lats, 'lon': lons})

xr_amp = {k: to_xr(v) for k, v in amp_data.items()}

# ------------------------------------------------------------------ #
# Global mean amplitude
# ------------------------------------------------------------------ #
print("\n=== Global mean whiplash amplitude (mean |ΔSPEI| | whiplash) ===")
print(f"{'Scenario':<16} {'Amplitude':>10}")
print("-" * 30)
for sc in ['CTL_35', 'SAI-1.5', 'SAI-1.0', 'CTL_45', 'Delayed']:
    label = sc.replace('_35','(35-64)').replace('_45','(45-64)')
    # nanmean across all grid points and members
    val = float(np.nanmean(amp_data[sc]))
    print(f"{label:<18} {val:>9.4f}")

# ------------------------------------------------------------------ #
# Regional amplitude breakdown
# ------------------------------------------------------------------ #
print("\n=== Regional mean whiplash amplitude ===")
print(f"{'Region':<18} {'CTL':>8} {'SAI-1.5':>9} {'SAI-1.0':>9} {'Delayed':>9}  "
      f"{'Diff-1.5':>10} {'Diff-1.0':>10} {'Diff-Del(A)':>12}")
print("-" * 100)

for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
    mask = region_mask(xr_amp['CTL_35'], lat_s, lat_n, lon_w, lon_e)

    def reg_amp(sc):
        return xr_amp[sc].where(mask).mean(dim=['lat','lon']).values

    ctl35 = reg_amp('CTL_35')
    ctl45 = reg_amp('CTL_45')
    s15   = reg_amp('SAI-1.5')
    s10   = reg_amp('SAI-1.0')
    deld  = reg_amp('Delayed')

    _, p15  = stats.ttest_1samp(s15  - ctl35, popmean=0)
    _, p10  = stats.ttest_1samp(s10  - ctl35, popmean=0)
    _, pdel = stats.ttest_1samp(deld - ctl45, popmean=0)

    print(f"{rname:<18} {ctl35.mean():>8.4f} {s15.mean():>9.4f} {s10.mean():>9.4f} {deld.mean():>9.4f}  "
          f"{s15.mean()-ctl35.mean():>+8.4f}{sig(p15):>3}  "
          f"{s10.mean()-ctl35.mean():>+8.4f}{sig(p10):>3}  "
          f"{deld.mean()-ctl45.mean():>+10.4f}{sig(pdel):>3}")

# ------------------------------------------------------------------ #
# Key question: does SAI reduce amplitude or only frequency?
# ------------------------------------------------------------------ #
print("\n=== Frequency vs Amplitude: SAI effect summary ===")
print(f"{'Scenario':<12} {'Freq diff (%)':>14} {'Amp diff':>12} {'SAI reduces amp?':>18}")
print("-" * 60)

# Load frequency data for comparison
freq_data = {
    'SAI-1.5': float(xr.open_dataarray(save_dir + 'whiplash_freq_SAI_allMembers_SPEI3.nc').mean()) * 100,
    'SAI-1.0': float(xr.open_dataarray(save_dir + 'whiplash_freq_SAI1p0_allMembers_SPEI3_CTL2035.nc').mean()) * 100,
    'Delayed': float(xr.open_dataarray(save_dir + 'whiplash_freq_delayed2045_allMembers_SPEI3_CTL2035.nc').mean()) * 100,
}
ctl_freq = float(xr.open_dataarray(save_dir + 'whiplash_freq_CTL2035_allMembers_SPEI3.nc').mean()) * 100

ctl35_amp_global = float(np.nanmean(amp_data['CTL_35']))
ctl45_amp_global = float(np.nanmean(amp_data['CTL_45']))

for sc, ctl_amp, ctl_f in [
    ('SAI-1.5', ctl35_amp_global, ctl_freq),
    ('SAI-1.0', ctl35_amp_global, ctl_freq),
    ('Delayed', ctl45_amp_global, ctl_freq),
]:
    freq_diff = freq_data[sc] - ctl_f
    amp_diff  = float(np.nanmean(amp_data[sc])) - ctl_amp
    reduces   = 'Yes' if amp_diff < -0.001 else ('Negligible' if abs(amp_diff) <= 0.001 else 'No')
    print(f"{sc:<12} {freq_diff:>+13.3f}%  {amp_diff:>+11.4f}  {reduces:>18}")

print("\nDone.")
