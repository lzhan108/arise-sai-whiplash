#!/usr/bin/env python
# fig6_rolling_timeseries_v2.py
#
# Rolling 10-year whiplash frequency time series per region, v2.
# Thick line = 10-year moving ensemble median. Shading = ensemble IQR.
# 6 subplots (one per region), 3 SAI scenarios + CTL.
#
# v2 changes vs fig6_rolling_timeseries_all_scenarios_v3.py:
#   - Land mask applied within each region box (region_mask & land_mask)
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#
# Usage:
#   conda activate arise_spei
#   python fig6_rolling_timeseries_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/v2/figures_output/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

SCENARIOS = {
    'CTL':     ('SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
    'SAI-1.5': ('SPEI3_SAI15_v2_member{}_CTLbaseline.nc',       'SPEI-3', slice('2035', '2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035', '2064')),
    'Delayed': ('SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045', '2064')),
}

colors = {
    'CTL':     '#4C72B0',
    'SAI-1.5': '#DD8452',
    'SAI-1.0': '#55A868',
    'Delayed': '#C44E52',
}

# ------------------------------------------------------------------ #
# Compute annual whiplash frequency per member per region
# ------------------------------------------------------------------ #
print("Computing annual whiplash frequency per region...")

annual_data = {sc: {r: [] for r in REGIONS} for sc in SCENARIOS}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    for mem in members:
        ds   = xr.open_dataset(spei_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        wh   = np.abs(spei.diff(dim='time')) >= 2.0

        years = np.unique(wh.time.dt.year.values)
        for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
            rmask = region_mask(wh, lat_s, lat_n, lon_w, lon_e) & land_mask
            annual_vals = []
            for yr in years:
                wh_yr = wh.sel(time=wh.time.dt.year == yr).mean(dim='time')
                val   = float(area_weighted_mean(wh_yr, rmask)) * 100
                annual_vals.append(val)
            annual_data[sc_name][rname].append(annual_vals)
        ds.close()
    print(f"  {sc_name} done.")

for sc in SCENARIOS:
    for rname in REGIONS:
        annual_data[sc][rname] = np.array(annual_data[sc][rname])

year_arrays = {
    'CTL':     np.arange(2035, 2065),
    'SAI-1.5': np.arange(2035, 2065),
    'SAI-1.0': np.arange(2035, 2065),
    'Delayed': np.arange(2045, 2065),
}


def rolling_stats(data_2d, window=10):
    n_members, n_years = data_2d.shape
    rolled = np.array([
        uniform_filter1d(data_2d[m], size=window, mode='reflect')
        for m in range(n_members)
    ])
    median = np.median(rolled, axis=0)
    q25    = np.percentile(rolled, 25, axis=0)
    q75    = np.percentile(rolled, 75, axis=0)
    return median, q25, q75


# ------------------------------------------------------------------ #
# Plot: 2 rows x 3 cols
# ------------------------------------------------------------------ #
region_names = list(REGIONS.keys())
fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                         gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
axes_flat = axes.flatten()

for ri, rname in enumerate(region_names):
    ax = axes_flat[ri]

    for sc_name in ['CTL', 'SAI-1.5', 'SAI-1.0', 'Delayed']:
        years  = year_arrays[sc_name]
        data2d = annual_data[sc_name][rname]
        med, q25, q75 = rolling_stats(data2d, window=10)

        color = colors[sc_name]
        lw    = 2.0 if sc_name != 'CTL' else 1.8
        ls    = '--' if sc_name == 'CTL' else '-'

        ax.plot(years, med, color=color, linewidth=lw,
                linestyle=ls, label=sc_name, zorder=3)
        ax.fill_between(years, q25, q75, color=color,
                        alpha=0.18, zorder=2)

    ax.axvspan(2035, 2049, alpha=0.07, color='steelblue', zorder=1)
    ax.axvspan(2050, 2064, alpha=0.07, color='firebrick', zorder=1)

    ax.set_title(rname, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Whiplash frequency (%)', fontsize=12)
    ax.set_xlim(2035, 2064)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ri, rname in enumerate(region_names):
    ax = axes_flat[ri]
    ymax = ax.get_ylim()[1]
    for txt, x, c in [('Near future', 2042, 'steelblue'),
                       ('Far future',  2057, 'firebrick')]:
        ax.text(x, ymax * 0.97, txt, ha='center', va='top',
                fontsize=10, color=c, alpha=0.8)

handles = [
    plt.Line2D([0],[0], color=colors['CTL'],     lw=1.8, ls='--', label='CTL (2035-2064)'),
    plt.Line2D([0],[0], color=colors['SAI-1.5'], lw=2.0, label='SAI-1.5'),
    plt.Line2D([0],[0], color=colors['SAI-1.0'], lw=2.0, label='SAI-1.0'),
    plt.Line2D([0],[0], color=colors['Delayed'],  lw=2.0, label='Delayed-2045'),
    mpatches.Patch(color='gray', alpha=0.2, label='Ensemble IQR'),
]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=12, bbox_to_anchor=(0.5, -0.02),
           framealpha=0.9, edgecolor='#CCCCCC')

fig.suptitle('Regional whiplash frequency: 10-year rolling median\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  (shading = IQR)',
             fontsize=14, y=1.03)

plt.subplots_adjust(top=0.93, bottom=0.1)

out_path = fig_dir + 'fig6_rolling_timeseries_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
