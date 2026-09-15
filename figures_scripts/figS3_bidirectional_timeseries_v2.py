#!/usr/bin/env python
# figS3_bidirectional_timeseries_v2.py
#
# Rolling 10-year D->W and W->D time series per region, v2.
# 6 regions in 2 rows x 3 cols, each with 2 stacked subplots
# (D->W top, W->D bottom).
#
# v2 changes vs figS2_bidirectional_timeseries_planB_v3.py:
#   - Land mask applied within each region box (region_mask & land_mask)
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#
# Usage:
#   conda activate arise_spei
#   python figS2_bidirectional_timeseries_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from matplotlib.gridspec import GridSpec
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

colors = {'CTL': '#4C72B0', 'SAI-1.5': '#DD8452', 'SAI-1.0': '#55A868', 'Delayed': '#C44E52'}
year_arrays = {'CTL': np.arange(2035,2065), 'SAI-1.5': np.arange(2035,2065),
               'SAI-1.0': np.arange(2035,2065), 'Delayed': np.arange(2045,2065)}

# ------------------------------------------------------------------ #
# Compute annual D->W and W->D per region
# ------------------------------------------------------------------ #
print("Computing annual D->W and W->D per region...")

annual = {sc: {r: {'dw': [], 'wd': []} for r in REGIONS} for sc in SCENARIOS}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    for mem in members:
        ds   = xr.open_dataset(spei_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = spei.diff(dim='time')
        dw = (delta >=  2.0)
        wd = (delta <= -2.0)
        years = np.unique(spei.time.dt.year.values)

        for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
            rmask = region_mask(spei, lat_s, lat_n, lon_w, lon_e) & land_mask
            dw_yr, wd_yr = [], []
            for yr in years:
                dw_yr.append(float(area_weighted_mean(dw.sel(time=dw.time.dt.year==yr).mean(dim='time'), rmask)) * 100)
                wd_yr.append(float(area_weighted_mean(wd.sel(time=wd.time.dt.year==yr).mean(dim='time'), rmask)) * 100)
            annual[sc_name][rname]['dw'].append(dw_yr)
            annual[sc_name][rname]['wd'].append(wd_yr)
        ds.close()
    print(f"  {sc_name} done.")

for sc in SCENARIOS:
    for r in REGIONS:
        annual[sc][r]['dw'] = np.array(annual[sc][r]['dw'])
        annual[sc][r]['wd'] = np.array(annual[sc][r]['wd'])


def rolling_stats(data_2d, window=10):
    rolled = np.array([uniform_filter1d(data_2d[m], size=window, mode='reflect')
                       for m in range(data_2d.shape[0])])
    return np.median(rolled,axis=0), np.percentile(rolled,25,axis=0), np.percentile(rolled,75,axis=0)


# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
region_names = list(REGIONS.keys())

fig = plt.figure(figsize=(15, 13))
gs = GridSpec(5, 3, figure=fig,
              height_ratios=[1, 1, 0.15, 1, 1],
              hspace=0.12, wspace=0.28,
              top=0.91, bottom=0.08, left=0.07, right=0.97)

axes_pairs = [
    (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])),
    (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])),
    (fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])),
    (fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[4, 0])),
    (fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[4, 1])),
    (fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[4, 2])),
]

for ri, rname in enumerate(region_names):
    ax_dw, ax_wd = axes_pairs[ri]

    for sc_name in ['CTL', 'SAI-1.5', 'SAI-1.0', 'Delayed']:
        years = year_arrays[sc_name]
        color = colors[sc_name]
        lw    = 1.8
        ls    = '--' if sc_name == 'CTL' else '-'

        for ax, data_key in [(ax_dw, 'dw'), (ax_wd, 'wd')]:
            med, q25, q75 = rolling_stats(annual[sc_name][rname][data_key])
            ax.plot(years, med, color=color, linewidth=lw, linestyle=ls, zorder=3)
            ax.fill_between(years, q25, q75, color=color, alpha=0.15, zorder=2)

    for ax in [ax_dw, ax_wd]:
        ax.axvspan(2035, 2049, alpha=0.06, color='steelblue', zorder=1)
        ax.axvspan(2050, 2064, alpha=0.06, color='firebrick',  zorder=1)
        ax.set_xlim(2035, 2064)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_dw.set_title(rname, fontsize=14, fontweight='bold', loc='left', pad=3)
    ax_dw.set_ylabel('D->W (%)', fontsize=12)
    ax_wd.set_ylabel('W->D (%)', fontsize=12)
    ax_dw.set_xticklabels([])
    ax_wd.set_xlabel('Year', fontsize=12)

    ax_dw.spines['bottom'].set_linestyle('--')
    ax_dw.spines['bottom'].set_alpha(0.4)

sc_handles = [plt.Line2D([0],[0], color=colors[sc], lw=2,
              ls='--' if sc=='CTL' else '-', label=sc)
              for sc in ['CTL','SAI-1.5','SAI-1.0','Delayed']]
sc_handles.append(mpatches.Patch(color='gray', alpha=0.2, label='Ensemble IQR'))
fig.legend(handles=sc_handles, loc='lower center', ncol=5,
           fontsize=12, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Regional whiplash: D->W and W->D - 10-year rolling median\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  (shading = IQR)',
             fontsize=14)

out_path = fig_dir + 'figS3_bidirectional_timeseries_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
