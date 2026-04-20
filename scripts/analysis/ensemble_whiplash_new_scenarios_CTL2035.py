#!/usr/bin/env python
# ensemble_whiplash_new_scenarios_CTL2035.py
#
# Whiplash frequency for SAI-1.0 and Delayed-2045
# All scenarios compared against UNIFIED CTL baseline: 2035-2064
# (Method B — enables direct cross-scenario comparison)
#
# Output files include "_CTL2035" suffix to distinguish from Method A
#
# Usage:
#   conda activate arise_spei
#   python ensemble_whiplash_new_scenarios_CTL2035.py

import xarray as xr
import numpy as np

save_dir = "/disk/dtouma/lzhang/SPEI/"
members  = [f"{i:03d}" for i in range(1, 11)]

def calculate_whiplash(spei, threshold=2.0):
    delta = spei.diff(dim='time')
    return np.abs(delta) >= threshold

# ============================================================
# PART 1: ARISE-SAI-1.0  (SAI window: 2035-2064, CTL: 2035-2064)
# ============================================================
print("=" * 50)
print("Processing ARISE-SAI-1.0  [CTL: 2035-2064]")
print("=" * 50)

wh_sai1p0_list = []
wh_ctl_list    = []

for mem in members:
    print(f"Member {mem}...", flush=True)

    sai1p0 = xr.open_dataarray(save_dir + f"SPEI3_SAI1p0_member{mem}_CTLbaseline.nc")
    ctl    = xr.open_dataarray(save_dir + f"SPEI3_CTL_member{mem}_CTLbaseline.nc")

    wh_sai1p0 = calculate_whiplash(sai1p0.sel(time=slice('2035', '2064'))).mean(dim='time')
    wh_ctl    = calculate_whiplash(ctl.sel(time=slice('2035', '2064'))).mean(dim='time')

    wh_sai1p0_list.append(wh_sai1p0)
    wh_ctl_list.append(wh_ctl)
    print(f"  SAI-1.0={float(wh_sai1p0.mean())*100:.2f}%  CTL={float(wh_ctl.mean())*100:.2f}%", flush=True)

ens_sai1p0 = xr.concat(wh_sai1p0_list, dim='member')
ens_ctl    = xr.concat(wh_ctl_list,    dim='member')

ens_sai1p0.to_netcdf(save_dir + "whiplash_freq_SAI1p0_allMembers_SPEI3_CTL2035.nc")
ens_ctl.to_netcdf(   save_dir + "whiplash_freq_CTL2035_allMembers_SPEI3.nc")

print(f"\nEnsemble mean SAI-1.0 : {float(ens_sai1p0.mean())*100:.2f}%")
print(f"Ensemble mean CTL     : {float(ens_ctl.mean())*100:.2f}%")
print(f"Ensemble mean diff    : {float((ens_sai1p0 - ens_ctl).mean())*100:.2f}%")
print("SAI-1.0 DONE!\n")

# ============================================================
# PART 2: Delayed-2045  (SAI window: 2045-2064, CTL: 2035-2064)
# ============================================================
print("=" * 50)
print("Processing Delayed-2045  [CTL: 2035-2064]")
print("=" * 50)

wh_delayed_list = []
wh_ctl_B_list   = []

for mem in members:
    print(f"Member {mem}...", flush=True)

    delayed = xr.open_dataarray(save_dir + f"SPEI3_delayed2045_member{mem}_CTLbaseline.nc")
    ctl     = xr.open_dataarray(save_dir + f"SPEI3_CTL_member{mem}_CTLbaseline.nc")

    # SAI effect: Delayed SAI 2045-2064
    wh_delayed = calculate_whiplash(delayed.sel(time=slice('2045', '2064'))).mean(dim='time')
    # Unified CTL baseline: 2035-2064 (same as SAI-1.5 and SAI-1.0)
    wh_ctl_B   = calculate_whiplash(ctl.sel(time=slice('2035', '2064'))).mean(dim='time')

    wh_delayed_list.append(wh_delayed)
    wh_ctl_B_list.append(wh_ctl_B)
    print(f"  Delayed={float(wh_delayed.mean())*100:.2f}%  CTL={float(wh_ctl_B.mean())*100:.2f}%", flush=True)

ens_delayed = xr.concat(wh_delayed_list, dim='member')
ens_ctl_B   = xr.concat(wh_ctl_B_list,  dim='member')

ens_delayed.to_netcdf(save_dir + "whiplash_freq_delayed2045_allMembers_SPEI3_CTL2035.nc")
# ens_ctl_B same as CTL2035 already saved above

print(f"\nEnsemble mean Delayed-2045    : {float(ens_delayed.mean())*100:.2f}%")
print(f"Ensemble mean CTL (2035-2064) : {float(ens_ctl_B.mean())*100:.2f}%")
print(f"Ensemble mean diff            : {float((ens_delayed - ens_ctl_B).mean())*100:.2f}%")
print("Delayed-2045 DONE!")
