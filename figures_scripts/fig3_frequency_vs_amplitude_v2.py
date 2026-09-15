#!/usr/bin/env python
# fig3_frequency_vs_amplitude_v2.py
#
# Scatter plot: regional whiplash frequency vs amplitude, v2.
# Color = region, Shape = scenario.
#
# v2 changes vs fig3_frequency_vs_amplitude_color_v3.py:
#   - Land mask applied within each region box (region_mask & land_mask)
#   - Area-weighted (cos(lat)) spatial mean instead of a simple
#     arithmetic grid-cell mean
#   - All four scenarios read from unified _v2 SPEI3 files
#     (FAO-56 PET, per-member CTL baseline)
#
# Usage:
#   conda activate arise_spei
#   python step9_fig3_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/v2/figures_output/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

SCENARIOS = {
    'CTL':     ('SPEI3_CTL_v2_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035', '2064')),
    'SAI-1.5': ('SPEI3_SAI15_v2_member{}_CTLbaseline.nc',      'SPEI-3', slice('2035', '2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035', '2064')),
    'Delayed': ('SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3', slice('2045', '2064')),
}

region_colors = {
    'W N America':   '#E07B54',
    'NE Brazil':     '#4C9BE8',
    'Mediterranean': '#6BBF72',
    'W C Africa':    '#B07CC6',
    'W Amazon':      '#E8C84C',
    'N Australia':   '#E85C6B',
}

region_labels = {
    'W N America':   'WNA',
    'NE Brazil':     'NEB',
    'Mediterranean': 'MED',
    'W C Africa':    'WCA',
    'W Amazon':      'WAM',
    'N Australia':   'NAU',
}

scenario_markers = {'CTL': 'o', 'SAI-1.5': 's', 'SAI-1.0': '^', 'Delayed': 'D'}
scenario_sizes   = {'CTL': 100, 'SAI-1.5': 90, 'SAI-1.0': 90, 'Delayed': 80}

# ------------------------------------------------------------------ #
# Compute regional frequency and amplitude per member
# ------------------------------------------------------------------ #
print("Computing regional frequency and amplitude...")

results = {sc: {r: {'freq': [], 'amp': []} for r in REGIONS} for sc in SCENARIOS}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    for mem in members:
        ds   = xr.open_dataset(spei_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = np.abs(spei.diff(dim='time'))
        wh_mask  = delta >= 2.0
        amp_cond = delta.where(wh_mask)

        for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
            rmask = region_mask(spei, lat_s, lat_n, lon_w, lon_e) & land_mask
            freq = float(area_weighted_mean(wh_mask.astype(float).mean(dim='time'), rmask)) * 100
            amp  = float(area_weighted_mean(amp_cond.mean(dim='time'), rmask))
            results[sc_name][rname]['freq'].append(freq)
            results[sc_name][rname]['amp'].append(amp)
        ds.close()
    print(f"  {sc_name} done.")

freq_mean = {sc: {} for sc in SCENARIOS}
amp_mean  = {sc: {} for sc in SCENARIOS}
for sc in SCENARIOS:
    for rname in REGIONS:
        freq_mean[sc][rname] = np.mean(results[sc][rname]['freq'])
        amp_mean[sc][rname]  = np.mean(results[sc][rname]['amp'])

print("\nRegion            CTL_f  S15_f  S10_f  Del_f  CTL_a  S15_a")
for rname in REGIONS:
    print(f"{rname:<18} "
          f"{freq_mean['CTL'][rname]:.2f}  "
          f"{freq_mean['SAI-1.5'][rname]:.2f}  "
          f"{freq_mean['SAI-1.0'][rname]:.2f}  "
          f"{freq_mean['Delayed'][rname]:.2f}  "
          f"{amp_mean['CTL'][rname]:.3f}  "
          f"{amp_mean['SAI-1.5'][rname]:.3f}")

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(11, 7))

region_names = list(REGIONS.keys())
sai_scenarios = ['SAI-1.5', 'SAI-1.0', 'Delayed']

for rname in region_names:
    rcolor = region_colors[rname]
    ctl_f  = freq_mean['CTL'][rname]
    ctl_a  = amp_mean['CTL'][rname]
    for sc in sai_scenarios:
        ax.plot(
            [ctl_f, freq_mean[sc][rname]],
            [ctl_a, amp_mean[sc][rname]],
            color=rcolor, linewidth=1.2,
            linestyle='--', alpha=0.55, zorder=2
        )

for sc in SCENARIOS:
    for rname in region_names:
        f = freq_mean[sc][rname]
        a = amp_mean[sc][rname]
        ax.scatter(f, a,
                   color=region_colors[rname],
                   marker=scenario_markers[sc],
                   s=scenario_sizes[sc],
                   zorder=4, edgecolors='white', linewidths=0.6)

label_offsets = {
    'W N America':   (7,   3),
    'NE Brazil':     (7,   3),
    'Mediterranean': (7,   3),
    'W C Africa':    (7,   8),
    'W Amazon':      (7,   3),
    'N Australia':   (-35, 0),
}

for rname in region_names:
    f = freq_mean['CTL'][rname]
    a = amp_mean['CTL'][rname]
    ax.annotate(region_labels[rname], (f, a),
                textcoords='offset points',
                xytext=label_offsets[rname],
                fontsize=12, color=region_colors[rname], fontweight='bold')

region_handles = [
    mpatches.Patch(color=region_colors[r], label=region_labels[r] + f' ({r})')
    for r in region_names
]
scenario_handles = [
    mlines.Line2D([], [], color='#555555',
                  marker=scenario_markers[sc], linestyle='None',
                  markersize=8, label=sc)
    for sc in SCENARIOS
]

leg1 = ax.legend(handles=region_handles, title='Region',
                 fontsize=12, title_fontsize=12,
                 loc='lower right', framealpha=0.9, edgecolor='#CCCCCC')
ax.add_artist(leg1)
ax.legend(handles=scenario_handles, title='Scenario',
          fontsize=12, title_fontsize=12,
          loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')

ax.set_xlabel('Whiplash frequency (%)', fontsize=12)
ax.set_ylabel('Mean whiplash amplitude (|ΔSPEI|)', fontsize=12)
ax.set_title('Regional whiplash: frequency vs amplitude\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble mean',
             fontsize=14)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = fig_dir + 'fig3_frequency_vs_amplitude_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
