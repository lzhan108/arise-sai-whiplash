#!/usr/bin/env python
# step2_sai10_delayed_v2.py
#
# Recompute SPEI-3 for ARISE-SAI-1.0 and Delayed-2045 using:
#   1. FAO-56 Penman-Monteith PET (previously these used a simplified
#      Thornthwaite PET, inconsistent with CTL/SAI-1.5)
#   2. Each member's own CTL baseline (previously all 10 members were
#      standardized against a single shared member001 baseline)
#
# Input: monthly_concat files already preprocessed (PRECT, TREFHT,
#        RHREFHT, FSDS, U10), no need to rerun preprocess_monthly.py
# Baseline: D_CTL_baseline_v2_member{mem}.nc (from step1_ctl_baseline_v2.py)
#
# Usage:
#   conda activate arise_spei
#   nohup python step2_sai10_delayed_v2.py > ../logs/step2_sai10_delayed_v2.log 2>&1 &

import xarray as xr
import numpy as np
from scipy.stats import gamma, norm
import os

sai1p0_dir  = '/disk/dtouma/lzhang/ARISE_1p0/monthly_concat/'
delayed_dir = '/disk/dtouma/lzhang/ARISE_delayed2045/monthly_concat/'
baseline_dir = '/disk/dtouma/lzhang/SPEI_v2/'
out_dir = '/disk/dtouma/lzhang/SPEI_v2/'

MEMBERS = [f'{i:03d}' for i in range(1, 11)]


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
    Ra = (24 / np.pi) * 4.92 * dr * (
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


def compute_water_balance(data_dir, prefix, mem):
    """Load 5 monthly variables, compute D = P - PET (mm/month) using FAO-56."""
    ds_p  = xr.open_dataset(data_dir + f'PRECT_{prefix}_member{mem}.nc')
    ds_t  = xr.open_dataset(data_dir + f'TREFHT_{prefix}_member{mem}.nc')
    ds_rh = xr.open_dataset(data_dir + f'RHREFHT_{prefix}_member{mem}.nc')
    ds_rs = xr.open_dataset(data_dir + f'FSDS_{prefix}_member{mem}.nc')
    ds_w  = xr.open_dataset(data_dir + f'U10_{prefix}_member{mem}.nc')

    days = ds_p.time.dt.days_in_month
    P  = ds_p['PRECT']  * 86400 * days * 1000
    T  = ds_t['TREFHT'] - 273.15
    RH = ds_rh['RHREFHT'].clip(0, 100)
    Rs = ds_rs['FSDS']   * 86400 / 1e6
    U2 = ds_w['U10']     * (4.87 / np.log(67.8 * 10 - 5.42))

    PET = fao56_pet(T, RH, Rs, U2, ds_t.lat, ds_t.time)
    D = P - PET
    D.name = 'water_balance'

    for ds in [ds_p, ds_t, ds_rh, ds_rs, ds_w]:
        ds.close()
    return D


def calculate_spei3_with_baseline(D_target, D_baseline):
    """SPEI-3: fit gamma distribution on this member's own CTL baseline,
    then standardize this member's target-scenario data against it."""
    D_roll_t = D_target.rolling(time=3, min_periods=3).sum()
    D_roll_b = D_baseline.rolling(time=3, min_periods=3).sum()

    spei_vals = np.full(D_roll_t.shape, np.nan)
    months_target   = D_roll_t.time.dt.month.values
    months_baseline = D_roll_b.time.dt.month.values

    for m in range(1, 13):
        if m % 3 == 1:
            print(f'  Fitting month {m}/12...', flush=True)
        idx_b = np.where(months_baseline == m)[0]
        idx_t = np.where(months_target   == m)[0]
        db = D_roll_b.values[idx_b]
        dt = D_roll_t.values[idx_t]

        nlat, nlon = db.shape[1], db.shape[2]
        for i in range(nlat):
            for j in range(nlon):
                base = db[:, i, j]
                tgt  = dt[:, i, j]
                valid = base[np.isfinite(base)]
                if len(valid) < 4:
                    continue
                try:
                    shift = valid.min() - 0.01
                    params = gamma.fit(valid - shift, floc=0)
                    cdf = np.clip(gamma.cdf(tgt - shift, *params), 0.001, 0.999)
                    spei_vals[idx_t, i, j] = norm.ppf(cdf)
                except Exception:
                    pass

    result = D_roll_t.copy(data=spei_vals)
    result.name = 'SPEI3'
    result.attrs['description'] = (
        'SPEI-3, FAO-56 PET, standardized against this member\'s own '
        'CTL baseline 2015-2034 (v2 fix: PET method + per-member baseline)'
    )
    return result


# ------------------------------------------------------------------ #
# Process SAI-1.0
# ------------------------------------------------------------------ #
print('=' * 50, flush=True)
print('Processing ARISE-SAI-1.0', flush=True)
print('=' * 50, flush=True)

for mem in MEMBERS:
    out_file = out_dir + f'SPEI3_SAI1p0_v2_member{mem}_CTLbaseline.nc'
    print(f'\n=== SAI-1.0 Member {mem} ===', flush=True)

    D_baseline = xr.open_dataarray(baseline_dir + f'D_CTL_baseline_v2_member{mem}.nc')

    print('  Computing water balance (FAO-56)...', flush=True)
    D = compute_water_balance(sai1p0_dir, 'SAI1p0', mem)
    D = D.sel(time=slice('2035', '2069'))

    print('  Computing SPEI-3...', flush=True)
    spei = calculate_spei3_with_baseline(D, D_baseline)

    spei.to_netcdf(out_file)
    print(f'  Saved: {out_file}', flush=True)

print('\nSAI-1.0 complete.', flush=True)

# ------------------------------------------------------------------ #
# Process Delayed-2045
# ------------------------------------------------------------------ #
print('=' * 50, flush=True)
print('Processing Delayed-2045', flush=True)
print('=' * 50, flush=True)

for mem in MEMBERS:
    out_file = out_dir + f'SPEI3_delayed2045_v2_member{mem}_CTLbaseline.nc'
    print(f'\n=== Delayed-2045 Member {mem} ===', flush=True)

    D_baseline = xr.open_dataarray(baseline_dir + f'D_CTL_baseline_v2_member{mem}.nc')

    print('  Computing water balance (FAO-56)...', flush=True)
    D = compute_water_balance(delayed_dir, 'delayed2045', mem)
    D = D.sel(time=slice('2045', '2069'))

    print('  Computing SPEI-3...', flush=True)
    spei = calculate_spei3_with_baseline(D, D_baseline)

    spei.to_netcdf(out_file)
    print(f'  Saved: {out_file}', flush=True)

print('\nDelayed-2045 complete.', flush=True)
print('\nAll done.', flush=True)
