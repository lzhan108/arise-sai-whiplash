#!/usr/bin/env python
# step1_ctl_baseline_v2.py
#
# Re-extract each member's own CTL water balance baseline (2015-2034)
# Fixes the bug where all members shared member001's baseline
# PET method: FAO-56 Penman-Monteith (consistent with run_SPEI_all_members.py)
#
# Usage:
#   conda activate arise_spei
#   nohup python step1_ctl_baseline_v2.py > /home/staff/lzhang/arise_whiplash/logs/step1_ctl_baseline_v2.log 2>&1 &

import xarray as xr
import numpy as np
import glob
import os

ctl_base_dir = "/disk/dtouma/ARISE-SAI"
out_dir = "/disk/dtouma/lzhang/SPEI_v2/"
os.makedirs(out_dir, exist_ok=True)

MEMBERS = [f"{i:03d}" for i in range(1, 11)]


def find_files(directory, pattern):
    files = sorted(glob.glob(directory + f"*{pattern}*"))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {directory}")
    return files


def load_var(files, var):
    if len(files) == 1:
        return xr.open_dataset(files[0])[var]
    return xr.open_mfdataset(files, compat='override', data_vars='minimal', coords='minimal')[var]


def fao56_pet(T, RH, Rs, U2, lat, time):
    """FAO-56 Penman-Monteith PET (mm/month), identical to run_SPEI_all_members.py"""
    delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3)**2
    P_atm = 101.325
    gamma_ = 0.000665 * P_atm
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    ea = es * RH / 100.0
    Rns = (1 - 0.23) * Rs
    lat_rad = np.deg2rad(lat)
    J = 30 * (time.dt.month - 1) + 15
    delta_sun = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    cos_ws = (-np.tan(lat_rad) * np.tan(delta_sun)).clip(-1, 1)
    ws = np.arccos(cos_ws)
    Ra = (24/np.pi) * 4.92 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta_sun) +
        np.cos(lat_rad) * np.cos(delta_sun) * np.sin(ws)
    )
    Ra = Ra.clip(min=0)
    Rso = 0.75 * Ra
    sigma = 4.903e-9
    T_K = T + 273.15
    ratio = (Rs / Rso.where(Rso > 0, other=1)).clip(0.25, 1.0)
    Rnl = sigma * T_K**4 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * ratio - 0.35)
    Rn = Rns - Rnl
    G = 0.07 * Rn
    numerator = (0.408 * delta * (Rn - G) +
                 gamma_ * (900 / (T + 273)) * U2 * (es - ea))
    denominator = delta + gamma_ * (1 + 0.34 * U2)
    PET_day = (numerator / denominator).clip(min=0)
    return PET_day * time.dt.days_in_month


for mem in MEMBERS:
    print(f"\n=== CTL baseline, member {mem} ===", flush=True)
    ctl_dir = f"{ctl_base_dir}/CTL/member{mem}/monthly/"

    T_ctl  = load_var(find_files(ctl_dir, 'TREFHT'), 'TREFHT') - 273.15
    days_c = xr.open_mfdataset(find_files(ctl_dir, 'PRECT'), compat='override', data_vars='minimal', coords='minimal').time.dt.days_in_month
    P_ctl  = load_var(find_files(ctl_dir, 'PRECT'), 'PRECT') * 86400 * days_c * 1000
    RH_ctl = load_var(find_files(ctl_dir, 'RHREFHT'), 'RHREFHT').clip(0, 100)
    Rs_ctl = load_var(find_files(ctl_dir, 'FSDS'), 'FSDS') * 86400 / 1e6
    U2_ctl = load_var(find_files(ctl_dir, 'U10'), 'U10') * (4.87 / np.log(67.8*10 - 5.42))

    print("  Calculating PET (FAO-56)...", flush=True)
    PET_ctl = fao56_pet(T_ctl, RH_ctl, Rs_ctl, U2_ctl, T_ctl.lat, T_ctl.time)
    D_ctl = P_ctl - PET_ctl

    D_baseline = D_ctl.sel(time=slice('2015', '2034'))
    D_baseline = D_baseline.compute()
    D_baseline.name = 'water_balance'

    out_file = out_dir + f"D_CTL_baseline_v2_member{mem}.nc"
    D_baseline.to_netcdf(out_file)
    print(f"  Saved: {out_file}  shape={D_baseline.shape}", flush=True)

print("\nAll 10 members' CTL baseline (2015-2034) saved.")
