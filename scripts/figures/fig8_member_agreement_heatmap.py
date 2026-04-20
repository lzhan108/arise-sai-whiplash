#!/usr/bin/env python
# fig8_member_agreement_heatmap.py
#
# For each region x scenario: count how many of 10 members
# show negative diff (SAI reduces whiplash)
# Display as heatmap with count/10 and color coding
#
# Usage:
#   conda activate arise_spei
#   python fig8_member_agreement_heatmap.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

sc_configs = [
    ('SAI-1.5', 'SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064'),
                'SPEI3_CTL_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    ('SAI-1.0', 'SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064'),
                'SPEI3_CTL_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    ('Delayed-2045','SPEI3_delayed2045_member{}_CTLbaseline.nc','SPEI3',slice('2045','2064'),
                'SPEI3_CTL_member{}_CTLbaseline.nc',        'SPEI-3', slice('2045','2064')),
]

# ------------------------------------------------------------------ #
# Compute per-member regional frequency diff
# ------------------------------------------------------------------ #
print("Computing per-member regional frequency...")

# agreement[sc][region] = list of 10 diff values
per_member = {sc[0]: {r: [] for r in regions} for sc in sc_configs}

for (sc_name, sai_tmpl, sai_var, sai_tslice,
     ctl_tmpl, ctl_var, ctl_tslice) in sc_configs:
    for mem in members:
        ds_sai = xr.open_dataset(save_dir + sai_tmpl.format(mem))
        ds_ctl = xr.open_dataset(save_dir + ctl_tmpl.format(mem))
        spei_sai = ds_sai[sai_var].sel(time=sai_tslice)
        spei_ctl = ds_ctl[ctl_var].sel(time=ctl_tslice)

        wh_sai = (np.abs(spei_sai.diff(dim='time')) >= 2.0).mean(dim='time')
        wh_ctl = (np.abs(spei_ctl.diff(dim='time')) >= 2.0).mean(dim='time')

        for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
            mask = region_mask(wh_sai, lat_s, lat_n, lon_w, lon_e)
            sai_val = float(wh_sai.where(mask).mean()) * 100
            ctl_val = float(wh_ctl.where(mask).mean()) * 100
            per_member[sc_name][rname].append(sai_val - ctl_val)

        ds_sai.close(); ds_ctl.close()
    print(f"  {sc_name} done.")

# ------------------------------------------------------------------ #
# Build agreement matrix: (6 regions x 3 scenarios)
# count = number of members with negative diff (SAI reduces whiplash)
# ------------------------------------------------------------------ #
region_names = list(regions.keys())
sc_names     = [sc[0] for sc in sc_configs]

agree_count = np.zeros((len(region_names), len(sc_names)), dtype=int)
mean_diff   = np.zeros((len(region_names), len(sc_names)))

for ri, rname in enumerate(region_names):
    for si, sc_name in enumerate(sc_names):
        vals = np.array(per_member[sc_name][rname])
        agree_count[ri, si] = np.sum(vals < 0)   # members showing reduction
        mean_diff[ri, si]   = vals.mean()

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))

# Color: 0/10 = dark red, 5/10 = white, 10/10 = dark blue
cmap = plt.cm.RdBu
norm = mcolors.Normalize(vmin=0, vmax=10)

im = ax.imshow(agree_count, cmap=cmap, norm=norm, aspect='auto')

# Annotations
for ri in range(len(region_names)):
    for si in range(len(sc_names)):
        count   = agree_count[ri, si]
        diff    = mean_diff[ri, si]
        # Bold if 8+/10 or 2-/10 (strong agreement)
        bold    = count >= 8 or count <= 2
        color   = 'white' if (count >= 8 or count <= 2) else 'black'
        ax.text(si, ri - 0.15, f'{count}/10',
                ha='center', va='center', fontsize=12,
                fontweight='bold' if bold else 'normal',
                color=color)
        ax.text(si, ri + 0.22, f'({diff:+.2f}%)',
                ha='center', va='center', fontsize=8.5,
                color=color, alpha=0.85)

# Grid lines
for x in np.arange(-0.5, len(sc_names), 1):
    ax.axvline(x, color='white', linewidth=1.0)
for y in np.arange(-0.5, len(region_names), 1):
    ax.axhline(y, color='white', linewidth=1.0)

ax.set_xticks(range(len(sc_names)))
ax.set_xticklabels(sc_names, fontsize=12)
ax.set_yticks(range(len(region_names)))
ax.set_yticklabels(region_names, fontsize=11)

cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                    shrink=0.85, pad=0.02)
cbar.set_label('Members showing SAI reduction (out of 10)', fontsize=10)
cbar.set_ticks([0, 2, 4, 5, 6, 8, 10])

ax.set_title('Ensemble member agreement: SAI reduces whiplash\n'
             'Number of members (out of 10) showing negative diff (SAI − CTL)\n'
             'Bold = strong agreement (≥8/10 or ≤2/10)',
             fontsize=11)

plt.tight_layout()
out_path = fig_dir + 'fig8_member_agreement_heatmap.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
