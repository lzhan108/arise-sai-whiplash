#!/usr/bin/env python
# step4_ensemble_whiplash_v2.py
#
# Aggregate per-member SPEI3 files into 10-member whiplash frequency
# ensembles for all four scenarios, using a unified CTL reference
# period (2035-2064) for all comparisons (Method B, confirmed with
# Danielle: "stick to Method B").
#
# This is the v2 equivalent of ensemble_whiplash.py and
# ensemble_whiplash_new_scenarios_CTL2035.py combined into one script.
#
# Usage:
#   conda activate arise_spei
#   nohup python step4_ensemble_whiplash_v2.py > ../logs/step4_ensemble_whiplash_v2.log 2>&1 &

import xarray as xr
import numpy as np
import os

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
out_dir  = "/disk/dtouma/lzhang/SPEI_v2/"
os.makedirs(out_dir, exist_ok=True)

MEMBERS = [f"{i:03d}" for i in range(1, 11)]


def calculate_whiplash(spei, threshold=2.0):
    delta = spei.diff(dim='time')
    return np.abs(delta) >= threshold


# (scenario name, file prefix, variable name, time slice, output file)
SCENARIOS = {
    'CTL':      ('SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
    'SAI15':    ('SPEI3_SAI15_v2_member{}_CTLbaseline.nc',       'SPEI-3', slice('2035', '2064')),
    'SAI1p0':   ('SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035', '2064')),
    'delayed2045': ('SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3', slice('2045', '2064')),
}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    print(f"\n=== Processing {sc_name} ===", flush=True)
    wh_list = []
    for mem in MEMBERS:
        da = xr.open_dataarray(spei_dir + fname_tmpl.format(mem))
        wh = calculate_whiplash(da.sel(time=tslice)).mean(dim='time')
        wh_list.append(wh)
        print(f"  member {mem}: whiplash freq = {float(wh.mean())*100:.2f}%", flush=True)

    ens = xr.concat(wh_list, dim='member')
    out_file = out_dir + f"whiplash_freq_{sc_name}_v2_allMembers.nc"
    ens.to_netcdf(out_file)
    print(f"  Saved: {out_file}", flush=True)
    print(f"  Ensemble mean: {float(ens.mean())*100:.2f}%", flush=True)

# ------------------------------------------------------------------ #
# CTL baseline (2015-2034), used as the reference for offset ratios
# ------------------------------------------------------------------ #
print("\n=== Processing CTL baseline (2015-2034) ===", flush=True)
wh_base_list = []
for mem in MEMBERS:
    da = xr.open_dataarray(spei_dir + f'SPEI3_CTL_v2_member{mem}_CTLbaseline.nc')
    wh = calculate_whiplash(da.sel(time=slice('2015', '2034'))).mean(dim='time')
    wh_base_list.append(wh)
    print(f"  member {mem}: baseline freq = {float(wh.mean())*100:.2f}%", flush=True)

ens_base = xr.concat(wh_base_list, dim='member')
out_file = out_dir + "whiplash_freq_CTL_baseline2015_v2_allMembers.nc"
ens_base.to_netcdf(out_file)
print(f"  Saved: {out_file}", flush=True)
print(f"  Ensemble mean: {float(ens_base.mean())*100:.2f}%", flush=True)

print("\nAll scenarios done.", flush=True)
