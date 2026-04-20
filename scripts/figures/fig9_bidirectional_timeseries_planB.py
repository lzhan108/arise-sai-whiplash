#!/usr/bin/env python
# fig9_bidirectional_timeseries_planB.py
#
# Rolling 10-year time series for D->W and W->D
# Per region: two stacked panels (D->W top, W->D bottom)
# 6 regions in 2 rows x 3 cols, each with 2 stacked subplots
#
# Usage:
#   conda activate arise_spei
#   python fig9_bidirectional_timeseries_planB.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

save_dir = "/disk/dtouma/lzhang/SPEI/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/figures/"
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

scenarios = {
    'CTL':     ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'SAI-1.5': ('SPEI3_SAI_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035','2064')),
    'Delayed': ('SPEI3_delayed2045_member{}_CTLbaseline.nc',  'SPEI3',  slice('2045','2064')),
}

colors = {'CTL': '#4C72B0', 'SAI-1.5': '#DD8452', 'SAI-1.0': '#55A868', 'Delayed': '#C44E52'}
year_arrays = {'CTL': np.arange(2035,2065), 'SAI-1.5': np.arange(2035,2065),
               'SAI-1.0': np.arange(2035,2065), 'Delayed': np.arange(2045,2065)}

# ------------------------------------------------------------------ #
# Reuse annual data from planA — compute same way
# ------------------------------------------------------------------ #
print("Computing annual D->W and W->D per region...")

annual = {sc: {r: {'dw': [], 'wd': []} for r in regions} for sc in scenarios}

for sc_name, (fname_tmpl, varname, tslice) in scenarios.items():
    for mem in members:
        ds   = xr.open_dataset(save_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = spei.diff(dim='time')
        dw = (delta >=  2.0)
        wd = (delta <= -2.0)
        years = np.unique(spei.time.dt.year.values)

        for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
            mask = region_mask(spei, lat_s, lat_n, lon_w, lon_e)
            dw_yr, wd_yr = [], []
            for yr in years:
                dw_yr.append(float(dw.sel(time=dw.time.dt.year==yr).where(mask).mean()) * 100)
                wd_yr.append(float(wd.sel(time=wd.time.dt.year==yr).where(mask).mean()) * 100)
            annual[sc_name][rname]['dw'].append(dw_yr)
            annual[sc_name][rname]['wd'].append(wd_yr)
        ds.close()
    print(f"  {sc_name} done.")

for sc in scenarios:
    for r in regions:
        annual[sc][r]['dw'] = np.array(annual[sc][r]['dw'])
        annual[sc][r]['wd'] = np.array(annual[sc][r]['wd'])

def rolling_stats(data_2d, window=10):
    rolled = np.array([uniform_filter1d(data_2d[m], size=window, mode='reflect')
                       for m in range(data_2d.shape[0])])
    return np.median(rolled,axis=0), np.percentile(rolled,25,axis=0), np.percentile(rolled,75,axis=0)

# ------------------------------------------------------------------ #
# Plan B: 2 stacked panels per region (D->W top, W->D bottom)
# Layout: 4 rows x 3 cols (row pairs = one region each)
# ------------------------------------------------------------------ #
region_names = list(regions.keys())

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15, 12))
# 4 data rows + 1 spacer row between group 1 and group 2
gs = GridSpec(5, 3, figure=fig,
              height_ratios=[1, 1, 0.15, 1, 1],
              hspace=0.12, wspace=0.28,
              top=0.91, bottom=0.06, left=0.07, right=0.97)

axes_pairs = [
    (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])),  # W N America
    (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])),  # NE Brazil
    (fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])),  # Mediterranean
    (fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[4, 0])),  # W C Africa
    (fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[4, 1])),  # W Amazon
    (fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[4, 2])),  # N Australia
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Region title: inside top panel at top-left
    ax_dw.set_title(rname, fontsize=11, fontweight='bold', loc='left', pad=3)

    # Direction labels
    ax_dw.set_ylabel('D→W (%)', fontsize=9)
    ax_wd.set_ylabel('W→D (%)', fontsize=9)

    # X labels only on bottom panel of each pair
    ax_dw.set_xticklabels([])
    ax_wd.set_xlabel('Year', fontsize=9)

    # Separator line between D->W and W->D
    ax_dw.spines['bottom'].set_linestyle('--')
    ax_dw.spines['bottom'].set_alpha(0.4)

# Legend inside figure at bottom
sc_handles = [plt.Line2D([0],[0], color=colors[sc], lw=2,
              ls='--' if sc=='CTL' else '-', label=sc)
              for sc in ['CTL','SAI-1.5','SAI-1.0','Delayed']]
sc_handles.append(mpatches.Patch(color='gray', alpha=0.2, label='Ensemble IQR'))
fig.legend(handles=sc_handles, loc='lower center', ncol=5,
           fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.03))

fig.suptitle('Regional whiplash: D→W and W→D — 10-year rolling median\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  (shading = IQR)',
             fontsize=12)

out_path = fig_dir + 'fig9_bidirectional_timeseries_planB.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
