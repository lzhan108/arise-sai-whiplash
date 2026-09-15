#!/usr/bin/env python
# fig2_regional_boxplot_v2.py
#
# Regional whiplash frequency boxplot (upper) + offset ratio bar chart
# (lower), v2. CTL, SAI-1.5, SAI-1.0, Delayed-2045.
#
# v2 changes vs fig2_regional_boxplot_with_offset_combine_v4.py:
#   - Land mask applied within each region box (region_mask & land_mask)
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#
# Usage:
#   conda activate arise_spei
#   python step8_fig2_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/v2/figures_output/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

# ------------------------------------------------------------------ #
# Load ensemble files
# ------------------------------------------------------------------ #
ens_ctl     = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_v2_allMembers.nc')
ens_sai15   = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI15_v2_allMembers.nc')
ens_sai10   = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI1p0_v2_allMembers.nc')
ens_delayed = xr.open_dataarray(spei_dir + 'whiplash_freq_delayed2045_v2_allMembers.nc')

# ------------------------------------------------------------------ #
# CTL baseline (2015-2034), per region, per member
# ------------------------------------------------------------------ #
base_list = {rname: [] for rname in REGIONS}
for mem in members:
    ctl = xr.open_dataarray(spei_dir + f'SPEI3_CTL_v2_member{mem}_CTLbaseline.nc')
    d   = ctl.diff(dim='time')
    wh  = (np.abs(d.sel(time=slice('2015', '2034'))) >= 2.0).mean(dim='time')
    for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
        rmask = region_mask(wh, lat_s, lat_n, lon_w, lon_e) & land_mask
        base_list[rname].append(float(area_weighted_mean(wh, rmask)) * 100)
    ctl.close()

# ------------------------------------------------------------------ #
# Collect regional data per member
# ------------------------------------------------------------------ #
region_names = list(REGIONS.keys())
ctl_data, s15_data, s10_data, del_data, base_vals = [], [], [], [], []

for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
    rmask = region_mask(ens_ctl, lat_s, lat_n, lon_w, lon_e) & land_mask
    ctl_data.append(area_weighted_mean(ens_ctl, rmask).values * 100)
    s15_data.append(area_weighted_mean(ens_sai15, rmask).values * 100)
    s10_data.append(area_weighted_mean(ens_sai10, rmask).values * 100)
    del_data.append(area_weighted_mean(ens_delayed, rmask).values * 100)
    base_vals.append(np.mean(base_list[rname]))

# ------------------------------------------------------------------ #
# Compute offset ratios
# ------------------------------------------------------------------ #
offset_s15, offset_s10, offset_del = [], [], []
for ri in range(len(region_names)):
    ctl_mean  = np.mean(ctl_data[ri])
    base_mean = base_vals[ri]
    denom     = ctl_mean - base_mean
    if abs(denom) < 1e-6:
        denom = 1e-6
    offset_s15.append((ctl_mean - np.mean(s15_data[ri])) / denom * 100)
    offset_s10.append((ctl_mean - np.mean(s10_data[ri])) / denom * 100)
    offset_del.append((ctl_mean - np.mean(del_data[ri])) / denom * 100)


def get_sig(sai, ctl):
    _, p = stats.ttest_1samp(np.array(sai) - np.array(ctl), popmean=0)
    if p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'


colors = {
    'CTL':      '#4C72B0',
    'SAI-1.5':  '#DD8452',
    'SAI-1.0':  '#55A868',
    'Delayed':  '#C44E52',
}

# ------------------------------------------------------------------ #
# Figure: two subplots, shared x-axis, height ratio 3:1
# ------------------------------------------------------------------ #
fig, (ax_box, ax_off) = plt.subplots(
    2, 1, figsize=(16, 12),
    sharex=True,
    gridspec_kw={'height_ratios': [2, 1.6], 'hspace': 0.2}
)

n_regions  = len(region_names)
x          = np.arange(n_regions) * 1.2
offsets    = [-0.38, -0.13, 0.13, 0.38]
box_w      = 0.22

scenario_data   = [ctl_data,  s15_data,  s10_data,  del_data]
scenario_labels = ['CTL',     'SAI-1.5', 'SAI-1.0', 'Delayed-2045']
scenario_colors = [colors['CTL'], colors['SAI-1.5'], colors['SAI-1.0'], colors['Delayed']]

for si, (data, label, color, offset) in enumerate(
        zip(scenario_data, scenario_labels, scenario_colors, offsets)):
    positions = x + offset
    ax_box.boxplot(
        data,
        positions=positions,
        widths=box_w,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.80),
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        flierprops=dict(marker='o', markersize=3,
                        markerfacecolor=color, markeredgecolor=color, alpha=0.5),
        zorder=3,
    )

sig_scenarios = [
    (s15_data, offsets[1], colors['SAI-1.5']),
    (s10_data, offsets[2], colors['SAI-1.0']),
    (del_data, offsets[3], colors['Delayed']),
]
for ri in range(n_regions):
    ctl_vals = ctl_data[ri]
    all_vals = np.concatenate([d[ri] for d in scenario_data])
    max_y    = np.percentile(all_vals, 100) + 0.4
    step     = 0.7
    bracket_n = 0
    for si, (sc_data, sc_offset, sc_color) in enumerate(sig_scenarios):
        sig_str = get_sig(sc_data[ri], ctl_vals)
        if sig_str == 'ns':
            continue
        y_br = max_y + bracket_n * step
        bracket_n += 1
        x0 = x[ri] + offsets[0]
        x1 = x[ri] + sc_offset
        ax_box.plot([x0, x0, x1, x1], [y_br, y_br+0.15, y_br+0.15, y_br],
                    color='#444444', linewidth=0.8, zorder=4)
        ax_box.text((x0+x1)/2, y_br+0.18, sig_str,
                    ha='center', va='bottom', fontsize=12,
                    color='#444444', fontweight='bold')

for ri, base in enumerate(base_vals):
    ax_box.plot([x[ri]-0.48, x[ri]+0.48], [base, base],
                color='#888888', linewidth=1.2, linestyle='--', zorder=2, alpha=0.7)

for ri in range(n_regions):
    for sc_idx, (sc_data, sc_offset) in enumerate(zip(
            [s15_data, s10_data, del_data], offsets[1:])):
        vals  = np.array(sc_data[ri]) - np.array(ctl_data[ri])
        count = np.sum(vals < 0)
        ypos  = np.max(sc_data[ri]) + 0.25
        color = scenario_colors[sc_idx + 1]
        ax_box.text(x[ri] + sc_offset, ypos, f'{count}/10',
                    ha='center', va='bottom', fontsize=10,
                    color=color,
                    fontweight='bold' if count >= 8 or count <= 2 else 'normal')

ax_box.set_ylabel('Whiplash frequency (%)', fontsize=12)
ax_box.set_title('Regional hydroclimate whiplash frequency\n'
                 'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble',
                 fontsize=14, pad=20)
ax_box.set_ylim(0, 18)
ax_box.grid(axis='y', alpha=0.3, linewidth=0.7)
ax_box.tick_params(axis='y', labelsize=12)
ax_box.spines['top'].set_visible(False)
ax_box.spines['right'].set_visible(False)

ax_box.text(-0.07, 1.02, '(a)', transform=ax_box.transAxes,
            fontsize=14, fontweight='bold', va='top')

handles = [mpatches.Patch(facecolor=colors[k], label=l, alpha=0.85)
           for k, l in zip(['CTL','SAI-1.5','SAI-1.0','Delayed'],
                           ['CTL (2035-2064)','SAI-1.5','SAI-1.0','Delayed-2045'])]
handles.append(plt.Line2D([0],[0], color='#888888', linestyle='--',
                           linewidth=1.2, label='CTL baseline (2015-2034)'))
ax_box.legend(handles=handles, fontsize=12, loc='upper right', framealpha=0.9)

bar_w   = 0.22
bar_off = [-0.22, 0, 0.22]
off_data    = [offset_s15, offset_s10, offset_del]
off_colors  = [colors['SAI-1.5'], colors['SAI-1.0'], colors['Delayed']]
off_labels  = ['SAI-1.5', 'SAI-1.0', 'Delayed-2045']

for si, (odata, ocolor, olabel) in enumerate(zip(off_data, off_colors, off_labels)):
    ax_off.bar(x + bar_off[si], odata, width=bar_w,
               color=ocolor, alpha=0.80, label=olabel, zorder=3)

ax_off.axhline(100, color='#444444', linewidth=1.0, linestyle='--',
               zorder=4, label='Full offset (100%)')
ax_off.axhline(0,   color='#888888', linewidth=0.7, zorder=2)

ax_off.set_ylabel('Offset ratio (%)', fontsize=12)
ax_off.set_ylim(-340, 260)
ax_off.set_xticks(x)
ax_off.set_xticklabels(region_names, fontsize=12, rotation=15, ha='right')
ax_off.tick_params(axis='y', labelsize=12)
ax_off.grid(axis='y', alpha=0.3, linewidth=0.7)
ax_off.spines['top'].set_visible(False)
ax_off.spines['right'].set_visible(False)

ax_off.text(-0.07, 0.98, '(b)', transform=ax_off.transAxes,
            fontsize=14, fontweight='bold', va='top')

for ri in range(len(region_names)):
    vals = [offset_s15[ri], offset_s10[ri], offset_del[ri]]
    base_offsets = [8, 8, 8]
    for i in range(1, 3):
        if vals[i] * vals[i-1] >= 0 and abs(vals[i] - vals[i-1]) < 20:
            base_offsets[i] = base_offsets[i-1] + 35
    for si, val in enumerate(vals):
        va  = 'bottom' if val >= 0 else 'top'
        yoff = base_offsets[si] if val >= 0 else -base_offsets[si]
        ax_off.text(x[ri] + bar_off[si], val + yoff,
                    f'{val:.0f}%', ha='center', va=va,
                    fontsize=10, color='#333333')

out_path = fig_dir + 'fig2_regional_boxplot_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
