#!/usr/bin/env python
# fig4_seasonal_heatmap_directional_all_scenarios.py
#
# 2 rows x 3 cols heatmap:
#   Row 1: D->W  (SAI-1.5 | SAI-1.0 | Delayed-2045)
#   Row 2: W->D  (SAI-1.5 | SAI-1.0 | Delayed-2045)
# Each cell: regional seasonal diff (SAI - CTL), significance stars
#
# Usage:
#   conda activate arise_spei
#   python fig4_seasonal_heatmap_directional_all_scenarios.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/figures/"
members  = [f"{i:03d}" for i in range(1, 11)]

seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}

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
    'SAI-1.5': ('SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064')),
    'Delayed': ('SPEI3_delayed2045_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045','2064')),
    'CTL_35':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'CTL_45':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2045','2064')),
}

# ------------------------------------------------------------------ #
# Compute seasonal D->W and W->D per member per region
# ------------------------------------------------------------------ #
print("Computing directional seasonal whiplash...")

# data[sc][direction][season][region] = list of 10 member values
data = {sc: {'dw': {s: {r: [] for r in regions} for s in seasons},
             'wd': {s: {r: [] for r in regions} for s in seasons}}
        for sc in scenarios}

for sc_name, (fname_tmpl, varname, tslice) in scenarios.items():
    for mem in members:
        ds   = xr.open_dataset(save_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = spei.diff(dim='time')
        dw = (delta >=  2.0)   # D->W events
        wd = (delta <= -2.0)   # W->D events

        for season, months in seasons.items():
            dw_s = dw.sel(time=dw.time.dt.month.isin(months)).mean(dim='time')
            wd_s = wd.sel(time=wd.time.dt.month.isin(months)).mean(dim='time')

            for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
                mask = region_mask(dw_s, lat_s, lat_n, lon_w, lon_e)
                data[sc_name]['dw'][season][rname].append(
                    float(dw_s.where(mask).mean()) * 100)
                data[sc_name]['wd'][season][rname].append(
                    float(wd_s.where(mask).mean()) * 100)
        ds.close()
    print(f"  {sc_name} done.")

# ------------------------------------------------------------------ #
# Build (6 regions x 4 seasons) diff arrays for each scenario+direction
# ------------------------------------------------------------------ #
region_names = list(regions.keys())
season_names = list(seasons.keys())

sc_ctl_pairs = [
    ('SAI-1.5', 'CTL_35', 'SAI-1.5  (2035–2064)'),
    ('SAI-1.0', 'CTL_35', 'SAI-1.0  (2035–2064)'),
    ('Delayed', 'CTL_45', 'Delayed-2045  (2045–2064)'),
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
# Plot: 2 rows (D->W, W->D) x 3 cols (scenarios)
# ------------------------------------------------------------------ #
vmax = 3.0
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                         gridspec_kw={'hspace': 0.35, 'wspace': 0.35})

row_labels = ['D→W', 'W→D']
row_data   = [(dw_diff, dw_pval), (wd_diff, wd_pval)]

for row, (row_label, (diff_list, pval_list)) in enumerate(
        zip(row_labels, row_data)):

    for col, (sc, ctl, sc_label) in enumerate(sc_ctl_pairs):
        ax  = axes[row, col]
        d   = diff_list[col]
        p   = pval_list[col]

        im = ax.imshow(d, cmap=cmap, norm=norm, aspect='auto')

        # Stars and value annotations
        for ri in range(len(region_names)):
            for si in range(len(season_names)):
                val = d[ri, si]
                pv  = p[ri, si]
                star = '***' if pv < 0.001 else ('**' if pv < 0.01 else
                       ('*' if pv < 0.05 else ''))
                text_color = 'white' if abs(val) > 1.2 else 'black'
                # Value
                ax.text(si, ri - 0.18, f'{val:+.2f}%',
                        ha='center', va='center', fontsize=7,
                        color=text_color, alpha=0.9)
                # Stars
                if star:
                    ax.text(si, ri + 0.22, star,
                            ha='center', va='center', fontsize=8,
                            color=text_color, fontweight='bold')

        # Grid lines
        for x in np.arange(-0.5, len(season_names), 1):
            ax.axvline(x, color='white', linewidth=0.8)
        for y in np.arange(-0.5, len(region_names), 1):
            ax.axhline(y, color='white', linewidth=0.8)

        # Axis labels
        ax.set_xticks(range(len(season_names)))
        ax.set_xticklabels(season_names, fontsize=11)
        ax.set_yticks(range(len(region_names)))
        ax.set_yticklabels(region_names if col == 0 else [], fontsize=10)

        # Title: scenario label on top row only
        if row == 0:
            ax.set_title(sc_label, fontsize=11, pad=8)

        # Row label on leftmost col
        if col == 0:
            ax.set_ylabel(row_label, fontsize=13, fontweight='bold', labelpad=8)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                    shrink=0.6, pad=0.02, aspect=30)
cbar.set_label('SAI − CTL whiplash frequency (%)', fontsize=10)

fig.suptitle('Seasonal directional whiplash frequency change: SAI − CTL\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  (* p<0.05  ** p<0.01  *** p<0.001)',
             fontsize=11, y=1.01)

out_path = fig_dir + 'fig4_seasonal_heatmap_directional_all_scenarios.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
