#!/usr/bin/env python
# fig2_global_maps_all_scenarios.py
#
# Global maps of whiplash frequency difference (SAI - CTL)
# 3 rows x 2 cols: each scenario has ensemble mean + significance dots
# Consistent with ensemble_analysis.ipynb Cell 0 style
#
# Usage:
#   conda activate arise_spei
#   python fig2_global_maps_all_scenarios.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/figures/"

# Region boxes for reference (lon_w, lon_e, lat_s, lat_n)
region_boxes = {
    'W N America':   (230, 260,  30,  60),
    'NE Brazil':     (315, 345, -15,   5),
    'Mediterranean': (350,  40,  30,  45),
    'W C Africa':    (  5,  30, -10,  15),
    'W Amazon':      (280, 310, -15,   5),
    'N Australia':   (120, 145, -25, -10),
}

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
ens_ctl   = xr.open_dataarray(save_dir + 'whiplash_freq_CTL2035_allMembers_SPEI3.nc')
ens_sai15 = xr.open_dataarray(save_dir + 'whiplash_freq_SAI_allMembers_SPEI3.nc')
ens_sai10 = xr.open_dataarray(save_dir + 'whiplash_freq_SAI1p0_allMembers_SPEI3_CTL2035.nc')
ens_del   = xr.open_dataarray(save_dir + 'whiplash_freq_delayed2045_allMembers_SPEI3_CTL2035.nc')

scenarios = [
    ('SAI-1.5',      ens_sai15, ens_ctl,  '2035–2064'),
    ('SAI-1.0',      ens_sai10, ens_ctl,  '2035–2064'),
    ('Delayed-2045', ens_del,   ens_ctl,  '2045–2064 vs CTL 2035–2064'),
]

lons = ens_ctl.lon.values
lats = ens_ctl.lat.values
lons2d, lats2d = np.meshgrid(lons, lats)

# ------------------------------------------------------------------ #
# Plot: 3 rows x 2 cols
# Left col: ensemble mean diff
# Right col: diff + significance dots
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(3, 2, figsize=(18, 14),
                         subplot_kw={'projection': ccrs.Robinson()},
                         gridspec_kw={'hspace': 0.08, 'wspace': 0.05})

for row, (label, ens_sai, ens_c, period) in enumerate(scenarios):
    diff     = (ens_sai - ens_c)            # (10, lat, lon)
    diff_mean = diff.mean(dim='member') * 100  # %
    _, p_val = stats.ttest_1samp(diff.values, popmean=0, axis=0)
    sig_mask = p_val < 0.05

    global_mean = float(diff_mean.mean())

    for col in range(2):
        ax = axes[row, col]
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')
        ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#F5F5F0')

        im = ax.pcolormesh(lons, lats, diff_mean,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r', vmin=-4, vmax=4, zorder=2)

        # Stippling: non-significant grid points
        nonsig_mask = ~sig_mask
        ax.scatter(lons2d[nonsig_mask], lats2d[nonsig_mask],
                   transform=ccrs.PlateCarree(),
                   s=0.08, c='black', alpha=0.25, zorder=3)

        # Region boxes
        for rname, (lon_w, lon_e, lat_s, lat_n) in region_boxes.items():
            if lon_w > lon_e:  # Mediterranean crosses 0
                ax.plot([lon_w, 360, 360, lon_w, lon_w],
                        [lat_s, lat_s, lat_n, lat_n, lat_s],
                        transform=ccrs.PlateCarree(),
                        color='black', linewidth=0.8, zorder=4)
                ax.plot([0, lon_e, lon_e, 0, 0],
                        [lat_s, lat_s, lat_n, lat_n, lat_s],
                        transform=ccrs.PlateCarree(),
                        color='black', linewidth=0.8, zorder=4)
            else:
                ax.plot([lon_w, lon_e, lon_e, lon_w, lon_w],
                        [lat_s, lat_s, lat_n, lat_n, lat_s],
                        transform=ccrs.PlateCarree(),
                        color='black', linewidth=0.8, zorder=4)

        # Column title
        if col == 0:
            ax.set_title(f'{label}  |  Ensemble mean\nGlobal mean diff: {global_mean:+.2f}%',
                         fontsize=10, loc='left')
        else:
            sig_pct = sig_mask.mean() * 100
            ax.set_title(f'{label}  |  Stippling = not significant ({100-sig_pct:.0f}% of land)\n(p ≥ 0.05, one-sample t-test)',
                         fontsize=10, loc='left')

    # store last im for shared colorbar
    last_im = im

plt.suptitle('SPEI-3 whiplash frequency: SAI − CTL\n10-member ensemble, |ΔSPEI| ≥ 2.0',
             fontsize=13, y=1.01)
plt.subplots_adjust(top=0.95)

# Single shared colorbar at the bottom
cax = fig.add_axes([0.25, 0.02, 0.5, 0.012])
cbar = fig.colorbar(last_im, cax=cax, orientation='horizontal')
cbar.set_label('Whiplash frequency change (%)', fontsize=10)

out_path = fig_dir + 'fig2_global_maps_all_scenarios_stipple_nonsig.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
