#!/usr/bin/env python
# step0_ctl_sai15_v2.py
#
# Recompute SPEI-3 for CTL and ARISE-SAI-1.5 using FAO-56 Penman-Monteith
# PET, each member standardized against its own CTL baseline (2015-2034).
#
# This is a v2 rewrite of run_SPEI_all_members.py, kept for consistency
# so the entire v2 pipeline is self-contained and reproducible from one
# set of scripts. CTL/SAI-1.5 were already computed correctly in the
# original pipeline (see /disk/dtouma/lzhang/SPEI/SPEI3_CTL_member*.nc
# and SPEI3_SAI_member*.nc, copied to _v2 naming without modification).
# Running this script is OPTIONAL - only needed to verify byte-for-byte
# (or numerically identical) reproducibility, not required for the
# actual v2 pipeline to function.
#
# Usage:
#   conda activate arise_spei
#   nohup python step0_ctl_sai15_v2.py > ../logs/step0_ctl_sai15_v2.log 2>&1 &

import xarray as xr
import numpy as np
from scipy.stats import gamma, norm
import glob
import os

base_dir = "/disk/dtouma/ARISE-SAI"
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


def calculate_spei_with_baseline(D_target, D_baseline, scale=3):
    """Fit gamma distribution on CTL baseline (2015-2034), standardize target."""
    D_roll_target   = D_target.rolling(time=scale, min_periods=scale).sum()
    D_roll_baseline = D_baseline.rolling(time=scale, min_periods=scale).sum()

    spei_vals = np.full(D_roll_target.shape, np.nan)
    months_target   = D_roll_target.time.dt.month.values
    months_baseline = D_roll_baseline.time.dt.month.values

    for m in range(1, 13):
        idx_base = np.where(months_baseline == m)[0]
        data_base = D_roll_baseline.values[idx_base, :, :]
        idx_tgt = np.where(months_target == m)[0]
        data_tgt = D_roll_target.values[idx_tgt, :, :]

        for i in range(data_base.shape[1]):
            for j in range(data_base.shape[2]):
                base_series = data_base[:, i, j]
                tgt_series  = data_tgt[:, i, j]
                valid_base  = base_series[np.isfinite(base_series)]
                if len(valid_base) < 4:
                    continue
                try:
                    shift = valid_base.min() - 0.01
                    params = gamma.fit(valid_base - shift, floc=0)
                    cdf = gamma.cdf(tgt_series - shift, *params)
                    cdf = np.clip(cdf, 0.001, 0.999)
                    spei_vals[idx_tgt, i, j] = norm.ppf(cdf)
                except Exception:
                    pass

    result = D_roll_target.copy(data=spei_vals)
    result.name = f'SPEI-{scale}'
    return result


def process_member(mem):
    print(f"\n{'='*50}\nProcessing Member {mem}\n{'='*50}", flush=True)

    sai_dir = f"{base_dir}/SAI/member{mem}/monthly/"
    ctl_dir = f"{base_dir}/CTL/member{mem}/monthly/"

    T_sai  = load_var(find_files(sai_dir, 'TREFHT'), 'TREFHT') - 273.15
    days_s = xr.open_mfdataset(find_files(sai_dir, 'PRECT'), compat='override', data_vars='minimal', coords='minimal').time.dt.days_in_month
    P_sai  = load_var(find_files(sai_dir, 'PRECT'), 'PRECT') * 86400 * days_s * 1000
    RH_sai = load_var(find_files(sai_dir, 'RHREFHT'), 'RHREFHT').clip(0, 100)
    Rs_sai = load_var(find_files(sai_dir, 'FSDS'), 'FSDS') * 86400 / 1e6
    U2_sai = load_var(find_files(sai_dir, 'U10'), 'U10') * (4.87 / np.log(67.8*10 - 5.42))

    T_ctl  = load_var(find_files(ctl_dir, 'TREFHT'), 'TREFHT') - 273.15
    days_c = xr.open_mfdataset(find_files(ctl_dir, 'PRECT'), compat='override', data_vars='minimal', coords='minimal').time.dt.days_in_month
    P_ctl  = load_var(find_files(ctl_dir, 'PRECT'), 'PRECT') * 86400 * days_c * 1000
    RH_ctl = load_var(find_files(ctl_dir, 'RHREFHT'), 'RHREFHT').clip(0, 100)
    Rs_ctl = load_var(find_files(ctl_dir, 'FSDS'), 'FSDS') * 86400 / 1e6
    U2_ctl = load_var(find_files(ctl_dir, 'U10'), 'U10') * (4.87 / np.log(67.8*10 - 5.42))

    print("Calculating PET...", flush=True)
    PET_sai = fao56_pet(T_sai, RH_sai, Rs_sai, U2_sai, T_sai.lat, T_sai.time)
    PET_ctl = fao56_pet(T_ctl, RH_ctl, Rs_ctl, U2_ctl, T_ctl.lat, T_ctl.time)
    D_sai = P_sai - PET_sai
    D_ctl = P_ctl - PET_ctl
    D_baseline = D_ctl.sel(time=slice('2015', '2034'))

    out_sai = out_dir + f"SPEI3_SAI15_v2_member{mem}_CTLbaseline.nc"
    out_ctl = out_dir + f"SPEI3_CTL_v2_member{mem}_CTLbaseline.nc"

    print("Calculating SPEI-3...", flush=True)
    SPEI_sai = calculate_spei_with_baseline(D_sai, D_baseline, scale=3)
    SPEI_ctl = calculate_spei_with_baseline(D_ctl, D_baseline, scale=3)
    SPEI_sai.to_netcdf(out_sai)
    SPEI_ctl.to_netcdf(out_ctl)
    print(f"Saved: {out_sai}", flush=True)
    print(f"Saved: {out_ctl}", flush=True)
    print(f"Member {mem} DONE!", flush=True)


for m in MEMBERS:
    process_member(m)

print("\nALL MEMBERS COMPLETE!", flush=True)
