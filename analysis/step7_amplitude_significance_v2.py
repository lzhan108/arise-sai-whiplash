#!/usr/bin/env python
# step7_amplitude_significance_v2.py
#
# Regional and global whiplash amplitude (mean |dSPEI| conditional on
# whiplash occurrence) for all four scenarios, with significance
# testing (SAI vs CTL), to support the Figure 3 discussion paragraph.
#
# This extends fig3_frequency_vs_amplitude_v2.py, which only computed
# amplitude for CTL and SAI-1.5. Here we add SAI-1.0 and Delayed-2045,
# plus a t-test comparing each SAI scenario against CTL.
#
# Usage:
#   conda activate arise_spei
#   python step7_amplitude_significance_v2.py

import xarray as xr
import numpy as np
from scipy import stats
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

SCENARIOS = {
    'CTL':     ('SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
    'SAI-1.5': ('SPEI3_SAI15_v2_member{}_CTLbaseline.nc',       'SPEI-3', slice('2035', '2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035', '2064')),
    'Delayed': ('SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045', '2064')),
}

print("Computing regional + global amplitude for all scenarios...")

# amp_results[sc][rname] = list of 10 per-member amplitudes
amp_results = {sc: {r: [] for r in REGIONS} for sc in SCENARIOS}
amp_global  = {sc: [] for sc in SCENARIOS}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    for mem in members:
        ds   = xr.open_dataset(spei_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = np.abs(spei.diff(dim='time'))
        wh_mask  = delta >= 2.0
        amp_cond = delta.where(wh_mask)
        amp_time_mean = amp_cond.mean(dim='time')

        # Global amplitude (land-masked, area-weighted)
        amp_global[sc_name].append(float(area_weighted_mean(amp_time_mean, land_mask)))

        # Regional amplitude
        for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
            rmask = region_mask(amp_time_mean, lat_s, lat_n, lon_w, lon_e) & land_mask
            amp_results[sc_name][rname].append(float(area_weighted_mean(amp_time_mean, rmask)))
        ds.close()
    print(f"  {sc_name} done.")


def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


# ------------------------------------------------------------------ #
# Global amplitude summary
# ------------------------------------------------------------------ #
print("\n=== Global mean whiplash amplitude ===")
ctl_amp = np.array(amp_global['CTL'])
print(f"{'Scenario':<12} {'Amplitude':>10} {'Diff':>10} {'Sig':>5}")
for sc in ['CTL', 'SAI-1.5', 'SAI-1.0', 'Delayed']:
    vals = np.array(amp_global[sc])
    mean_amp = vals.mean()
    if sc == 'CTL':
        print(f"{sc:<12} {mean_amp:>10.4f} {'-':>10} {'-':>5}")
    else:
        diff = vals - ctl_amp
        _, p = stats.ttest_1samp(diff, popmean=0)
        print(f"{sc:<12} {mean_amp:>10.4f} {diff.mean():>+10.4f} {sig(p):>5}")

# ------------------------------------------------------------------ #
# Regional amplitude summary
# ------------------------------------------------------------------ #
print("\n=== Regional whiplash amplitude (mean |dSPEI| | whiplash) ===")
print(f"{'Region':<18} {'CTL':>8} {'SAI-1.5':>10} {'sig':>5} {'SAI-1.0':>10} {'sig':>5} {'Delayed':>10} {'sig':>5}")
for rname in REGIONS:
    ctl_vals = np.array(amp_results['CTL'][rname])
    row = f"{rname:<18} {ctl_vals.mean():>8.4f}"
    for sc in ['SAI-1.5', 'SAI-1.0', 'Delayed']:
        sc_vals = np.array(amp_results[sc][rname])
        diff = sc_vals - ctl_vals
        _, p = stats.ttest_1samp(diff, popmean=0)
        row += f" {sc_vals.mean():>10.4f} {sig(p):>5}"
    print(row)

print("\nDone.")
