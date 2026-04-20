#!/usr/bin/env python
# fig1_regional_boxplot_all_scenarios.py
#
# Regional whiplash frequency boxplot: CTL, SAI-1.5, SAI-1.0, Delayed-2045
# Consistent with ensemble_analysis.ipynb Cell 5 style
#
# Usage:
#   conda activate arise_spei
#   python fig1_regional_boxplot_all_scenarios.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/figures/"

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

# ------------------------------------------------------------------ #
# Load ensemble files
# ------------------------------------------------------------------ #
ens_ctl     = xr.open_dataarray(save_dir + 'whiplash_freq_CTL2035_allMembers_SPEI3.nc')
ens_sai15   = xr.open_dataarray(save_dir + 'whiplash_freq_SAI_allMembers_SPEI3.nc')
ens_sai10   = xr.open_dataarray(save_dir + 'whiplash_freq_SAI1p0_allMembers_SPEI3_CTL2035.nc')
ens_delayed = xr.open_dataarray(save_dir + 'whiplash_freq_delayed2045_allMembers_SPEI3_CTL2035.nc')

# CTL baseline (2015-2034) global reference line
# Compute from SPEI files
members = [f"{i:03d}" for i in range(1, 11)]
base_list = {rname: [] for rname in regions}
for mem in members:
    ctl = xr.open_dataarray(save_dir + f'SPEI3_CTL_member{mem}_CTLbaseline.nc')
    d   = ctl.diff(dim='time')
    wh  = (np.abs(d.sel(time=slice('2015','2034'))) >= 2.0).mean(dim='time')
    for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
        mask = region_mask(wh, lat_s, lat_n, lon_w, lon_e)
        base_list[rname].append(float(wh.where(mask).mean()) * 100)
    ctl.close()

# ------------------------------------------------------------------ #
# Collect regional data per member
# ------------------------------------------------------------------ #
region_names = list(regions.keys())
ctl_data, s15_data, s10_data, del_data, base_vals = [], [], [], [], []

for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
    mask = region_mask(ens_ctl, lat_s, lat_n, lon_w, lon_e)

    ctl_data.append(ens_ctl.where(mask).mean(dim=['lat','lon']).values * 100)
    s15_data.append(ens_sai15.where(mask).mean(dim=['lat','lon']).values * 100)
    s10_data.append(ens_sai10.where(mask).mean(dim=['lat','lon']).values * 100)
    del_data.append(ens_delayed.where(mask).mean(dim=['lat','lon']).values * 100)
    base_vals.append(np.mean(base_list[rname]))

# ------------------------------------------------------------------ #
# Significance: paired t-test each scenario vs CTL
# ------------------------------------------------------------------ #
def get_sig(sai, ctl):
    _, p = stats.ttest_1samp(np.array(sai) - np.array(ctl), popmean=0)
    if p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
colors = {
    'CTL':      '#4C72B0',   # steel blue
    'SAI-1.5':  '#DD8452',   # orange
    'SAI-1.0':  '#55A868',   # green
    'Delayed':  '#C44E52',   # red
}

fig, ax = plt.subplots(figsize=(13, 6))

n_regions  = len(region_names)
x          = np.arange(n_regions)
group_w    = 1.0
offsets    = [-0.38, -0.13, 0.13, 0.38]
box_w      = 0.22

scenario_data   = [ctl_data,  s15_data,  s10_data,  del_data]
scenario_labels = ['CTL',     'SAI-1.5', 'SAI-1.0', 'Delayed-2045']
scenario_colors = [colors['CTL'], colors['SAI-1.5'], colors['SAI-1.0'], colors['Delayed']]

for si, (data, label, color, offset) in enumerate(
        zip(scenario_data, scenario_labels, scenario_colors, offsets)):

    positions = x + offset
    bp = ax.boxplot(
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

# Significance brackets (SAI scenarios vs CTL)
sig_scenarios = [
    (s15_data, offsets[1], colors['SAI-1.5']),
    (s10_data, offsets[2], colors['SAI-1.0']),
    (del_data, offsets[3], colors['Delayed']),
]

for ri in range(n_regions):
    ctl_vals  = ctl_data[ri]
    # Use 95th percentile of all data for this region to set bracket height
    all_vals  = np.concatenate([d[ri] for d in scenario_data])
    max_y     = np.percentile(all_vals, 100) + 0.4
    step      = 0.7
    bracket_n = 0  # count how many brackets drawn

    for si, (sc_data, sc_offset, sc_color) in enumerate(sig_scenarios):
        sig_str = get_sig(sc_data[ri], ctl_vals)
        if sig_str == 'ns':
            continue
        y_br = max_y + bracket_n * step
        bracket_n += 1
        x0   = x[ri] + offsets[0]
        x1   = x[ri] + sc_offset
        ax.plot([x0, x0, x1, x1], [y_br, y_br+0.15, y_br+0.15, y_br],
                color='#444444', linewidth=0.8, zorder=4)
        ax.text((x0+x1)/2, y_br+0.18, sig_str,
                ha='center', va='bottom', fontsize=8, color='#444444', fontweight='bold')

# CTL baseline reference line per region
for ri, base in enumerate(base_vals):
    ax.plot([x[ri]-0.48, x[ri]+0.48], [base, base],
            color='#888888', linewidth=1.2, linestyle='--', zorder=2, alpha=0.7)

# Member agreement annotation (X/10 members show reduction)
for ri in range(n_regions):
    for sc_idx, (sc_data, sc_offset) in enumerate(zip(
            [s15_data, s10_data, del_data], offsets[1:])):
        vals = np.array(sc_data[ri]) - np.array(ctl_data[ri])
        count = np.sum(vals < 0)
        ypos  = np.max(sc_data[ri]) + 0.25
        color = scenario_colors[sc_idx + 1]
        ax.text(x[ri] + sc_offset, ypos, f'{count}/10',
                ha='center', va='bottom', fontsize=7.5,
                color=color, fontweight='bold' if count >= 8 or count <= 2 else 'normal')
ax.set_xticks(x)
ax.set_xticklabels(region_names, fontsize=11, rotation=15, ha='right')
ax.set_ylabel('Whiplash frequency (%)', fontsize=12)
ax.set_title('Regional hydroclimate whiplash frequency\nSPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble',
             fontsize=12)
ax.set_ylim(0, 18)
ax.grid(axis='y', alpha=0.3, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
handles = [mpatches.Patch(facecolor=colors[k], label=l, alpha=0.85)
           for k, l in zip(['CTL','SAI-1.5','SAI-1.0','Delayed'],
                           ['CTL (2035–2064)','SAI-1.5','SAI-1.0','Delayed-2045'])]
handles.append(plt.Line2D([0],[0], color='#888888', linestyle='--',
                           linewidth=1.2, label='CTL baseline (2015–2034)'))
ax.legend(handles=handles, fontsize=10, loc='upper right', framealpha=0.9)

plt.tight_layout()
out_path = fig_dir + 'fig1_regional_whiplash_boxplot_member_agreement.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
