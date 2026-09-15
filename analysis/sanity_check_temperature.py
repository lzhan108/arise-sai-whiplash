#!/usr/bin/env python
# sanity_check_temperature.py
#
# Independent physical sanity check: compare raw global-mean TREFHT
# trajectories between SAI-1.5 and Delayed-2045, using ONLY raw
# temperature data (no SPEI, no PET, no gamma-fitting) to verify
# whether Delayed-2045's temperature converges to SAI-1.5 by the
# 2050s-2060s, as expected from its ~20% stronger injection rate.
#
# This is a cross-check unrelated to the whiplash pipeline itself.

import xarray as xr
import numpy as np
import glob

sai15_base  = "/disk/dtouma/ARISE-SAI/SAI"
delayed_dir = "/disk/dtouma/lzhang/ARISE_delayed2045/monthly_concat"


def find_files(directory, pattern):
    files = sorted(glob.glob(directory + f"/*{pattern}*"))
    return files


print("Loading SAI-1.5 TREFHT (member001, raw)...")
sai15_files = find_files(f"{sai15_base}/member001/monthly", "TREFHT")
ds_sai15 = xr.open_mfdataset(sai15_files, compat='override', data_vars='minimal', coords='minimal')['TREFHT']

print("Loading Delayed-2045 TREFHT (member001, raw)...")
ds_delayed = xr.open_dataarray(f"{delayed_dir}/TREFHT_delayed2045_member001.nc")

sai15_annual   = ds_sai15.sel(time=slice('2035', '2064')).resample(time='YE').mean().mean(dim=['lat', 'lon']) - 273.15
delayed_annual = ds_delayed.sel(time=slice('2045', '2064')).resample(time='YE').mean().mean(dim=['lat', 'lon']) - 273.15

print("\n=== Global mean surface temperature (degC), member001 ===")
print("Year   SAI-1.5   Delayed-2045   Diff")
sai15_vals   = sai15_annual.compute().values
delayed_vals = delayed_annual.compute().values
sai15_years   = sai15_annual.time.dt.year.values
delayed_years = delayed_annual.time.dt.year.values

for yr in range(2045, 2065):
    s15_v = sai15_vals[np.where(sai15_years == yr)[0][0]] if yr in sai15_years else np.nan
    del_v = delayed_vals[np.where(delayed_years == yr)[0][0]] if yr in delayed_years else np.nan
    print(f"{yr}   {s15_v:.3f}    {del_v:.3f}    {del_v - s15_v:+.3f}")
