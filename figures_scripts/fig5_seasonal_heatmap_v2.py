#!/usr/bin/env python
# fig5_seasonal_heatmap_v2.py
#
# Seasonal directional whiplash frequency heatmap, v2.
# 2 rows (D->W, W->D) x 3 cols (SAI-1.5, SAI-1.0, Delayed-2045).
# Each cell: 6 regions x 4 seasons, SAI - CTL diff with significance.
#
# v2 changes vs fig5_seasonal_heatmap_directional_all_scenarios_v3.py:
#   - Land mask applied within each region box (region_mask & land_mask)
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#
# Usage:
#   conda activate arise_spei
#   python fig5_seasonal_heatmap_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/v2/figures_output/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}

SCENARIOS = {
    'SAI-1.5': ('SPEI3_SAI15_v2_member{}_CTLbaseline.nc',       'SPEI-3', slice('2035', '2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035', '2064')),
    'Delayed': ('SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045', '2064')),
    'CTL_35':  ('SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
}

# ------------------------------------------------------------------ #
# Compute seasonal D->W and W->D per member per region
# ------------------------------------------------------------------ #
print("Computing directional seasonal whiplash...")

data = {sc: {'dw': {s: {r: [] for r in REGIONS} for s in seasons},
             'wd': {s: {r: [] for r in REGIONS} for s in seasons}}
        for sc in SCENARIOS}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    for mem in members:
        ds   = xr.open_dataset(spei_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = spei.diff(dim='time')
        dw = (delta >=  2.0)
        wd = (delta <= -2.0)

        for season, months in seasons.items():
            dw_s = dw.sel(time=dw.time.dt.month.isin(months)).mean(dim='time')
            wd_s = wd.sel(time=wd.time.dt.month.isin(months)).mean(dim='time')

            for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
                rmask = region_mask(dw_s, lat_s, lat_n, lon_w, lon_e) & land_mask
                data[sc_name]['dw'][season][rname].append(
                    float(area_weighted_mean(dw_s, rmask)) * 100)
                data[sc_name]['wd'][season][rname].append(
                    float(area_weighted_mean(wd_s, rmask)) * 100)
        ds.close()
    print(f"  {sc_name} done.")

# ------------------------------------------------------------------ #
# Build diff arrays
# ------------------------------------------------------------------ #
region_names = list(REGIONS.keys())
season_names = list(seasons.keys())

sc_ctl_pairs = [
    ('SAI-1.5', 'CTL_35', 'SAI-1.5  (2035-2064)'),
    ('SAI-1.0', 'CTL_35', 'SAI-1.0  (2035-2064)'),
    ('Delayed', 'CTL_35', 'Delayed-2045  (2045-2064 vs CTL 2035-2064)'),
]


def build_arrays(direction):
    diff_list, pval_list = [], []
    for sc, ctl, _ in sc_ctl_pairs:
        d = np.zeros((len(region_names), len(season_names)))
        p = np.ones((len(region_names), len(season_names)))
        for ri, rname in enumerate(region_names):
            for si, season in enumerate(season_names):
                sai_vals = np.array(data[sc][direction][season][rname])
                ctl_vals = np.array(data[ctl][direction][season][rname])
                diff = sai_vals - ctl_vals
                d[ri, si] = diff.mean()
                _, pv = stats.ttest_1samp(diff, popmean=0)
                p[ri, si] = pv
        diff_list.append(d)
        pval_list.append(p)
    return diff_list, pval_list


dw_diff, dw_pval = build_arrays('dw')
wd_diff, wd_pval = build_arrays('wd')

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
vmax = 3.0
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                         gridspec_kw={'hspace': 0.18, 'wspace': 0.35})

row_labels = ['D->W', 'W->D']
row_data   = [(dw_diff, dw_pval), (wd_diff, wd_pval)]

for row, (row_label, (diff_list, pval_list)) in enumerate(
        zip(row_labels, row_data)):

    for col, (sc, ctl, sc_label) in enumerate(sc_ctl_pairs):
        ax  = axes[row, col]
        d   = diff_list[col]
        p   = pval_list[col]

        im = ax.imshow(d, cmap=cmap, norm=norm, aspect='auto')

        for ri in range(len(region_names)):
            for si in range(len(season_names)):
                val = d[ri, si]
                pv  = p[ri, si]
                star = '***' if pv < 0.001 else ('**' if pv < 0.01 else
                       ('*' if pv < 0.05 else ''))
                text_color = 'white' if abs(val) > 1.2 else 'black'
                ax.text(si, ri - 0.18, f'{val:+.2f}%',
                        ha='center', va='center', fontsize=9,
                        color=text_color, alpha=0.9)
                if star:
                    ax.text(si, ri + 0.22, star,
                            ha='center', va='center', fontsize=10,
                            color=text_color, fontweight='bold')

        for x in np.arange(-0.5, len(season_names), 1):
            ax.axvline(x, color='white', linewidth=0.8)
        for y in np.arange(-0.5, len(region_names), 1):
            ax.axhline(y, color='white', linewidth=0.8)

        ax.set_xticks(range(len(season_names)))
        ax.set_xticklabels(season_names, fontsize=12)
        ax.set_yticks(range(len(region_names)))
        ax.set_yticklabels(region_names if col == 0 else [], fontsize=12)

        if row == 0:
            ax.set_title(sc_label, fontsize=12, pad=8)

        if col == 0:
            ax.set_ylabel(row_label, fontsize=13, fontweight='bold', labelpad=8)

cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                    shrink=0.6, pad=0.02, aspect=30)
cbar.set_label('SAI - CTL whiplash frequency (%)', fontsize=12)
cbar.ax.tick_params(labelsize=12)

fig.suptitle('Seasonal directional whiplash frequency change: SAI - CTL\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  '
             '(* p<0.05  ** p<0.01  *** p<0.001)',
             fontsize=14, y=0.98, x=0.45)

out_path = fig_dir + 'fig5_seasonal_heatmap_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
