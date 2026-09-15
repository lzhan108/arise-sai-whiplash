#!/usr/bin/env python
# fig1_global_maps_v2.py
#
# Global maps of whiplash frequency difference (SAI - CTL), v2.
# 3 rows x 1 col: each scenario has ensemble mean + significance hatching.
#
# v2 changes vs fig1_global_maps_all_scenarios_v3.py:
#   - Excluded grid cells (ocean, hyperarid desert, glaciers/Antarctica)
#     are masked out visually (set to NaN before plotting) so the map
#     matches the land mask used for all summary statistics
#   - Global mean in the title is now the area-weighted, land-masked
#     mean, matching Table 2
#
# Usage:
#   conda activate arise_spei
#   python step7_fig1_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from utils_v2 import load_land_mask, REGIONS, area_weighted_mean

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/v2/figures_output/"

land_mask = load_land_mask()

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
ens_ctl   = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_v2_allMembers.nc')
ens_sai15 = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI15_v2_allMembers.nc')
ens_sai10 = xr.open_dataarray(spei_dir + 'whiplash_freq_SAI1p0_v2_allMembers.nc')
ens_del   = xr.open_dataarray(spei_dir + 'whiplash_freq_delayed2045_v2_allMembers.nc')

scenarios = [
    ('SAI-1.5',      ens_sai15, ens_ctl, '2035-2064'),
    ('SAI-1.0',      ens_sai10, ens_ctl, '2035-2064'),
    ('Delayed-2045', ens_del,   ens_ctl, '2045-2064 vs CTL 2035-2064'),
]

lons = ens_ctl.lon.values
lats = ens_ctl.lat.values
lons2d, lats2d = np.meshgrid(lons, lats)

# ------------------------------------------------------------------ #
# Plot: 3 rows x 1 col
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(3, 1, figsize=(14, 16),
                         subplot_kw={'projection': ccrs.Robinson()},
                         gridspec_kw={'hspace': 0.15})

for row, (label, ens_sai, ens_c, period) in enumerate(scenarios):
    diff      = (ens_sai - ens_c)
    diff_mean = diff.mean(dim='member') * 100
    _, p_val  = stats.ttest_1samp(diff.values, popmean=0, axis=0)
    sig_mask  = p_val < 0.05
    nonsig_mask = ~sig_mask

    # v2: visually mask out excluded grid cells (ocean, desert, glacier)
    diff_mean_masked = diff_mean.where(land_mask)

    # v2: area-weighted, land-masked global mean, matching Table 2
    global_mean = float(area_weighted_mean(diff_mean, land_mask))
    sig_mask_da = xr.DataArray(sig_mask.astype(float), dims=["lat", "lon"],
                                coords={"lat": diff_mean.lat, "lon": diff_mean.lon})
    sig_pct = float(area_weighted_mean(sig_mask_da, land_mask)) * 100

    ax = axes[row]
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')
    ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#FFFFFF')

    im = ax.pcolormesh(lons, lats, diff_mean_masked,
                       transform=ccrs.PlateCarree(),
                       cmap='RdBu_r', vmin=-4, vmax=4, zorder=2)

    # Hatching: non-significant grid points (within the land mask only)
    nonsig_float = (nonsig_mask & land_mask.values).astype(float)
    ax.contourf(lons, lats, nonsig_float,
                levels=[0.5, 1.5],
                hatches=['////'],
                colors='none',
                transform=ccrs.PlateCarree(),
                zorder=3)

    # Region boxes
    for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
        if lon_w > lon_e:
            ax.plot([lon_w, 360, 360, lon_w, lon_w],
                    [lat_s, lat_s, lat_n, lat_n, lat_s],
                    transform=ccrs.PlateCarree(),
                    color='black', linewidth=0.8, zorder=5)
            ax.plot([0, lon_e, lon_e, 0, 0],
                    [lat_s, lat_s, lat_n, lat_n, lat_s],
                    transform=ccrs.PlateCarree(),
                    color='black', linewidth=0.8, zorder=5)
        else:
            ax.plot([lon_w, lon_e, lon_e, lon_w, lon_w],
                    [lat_s, lat_s, lat_n, lat_n, lat_s],
                    transform=ccrs.PlateCarree(),
                    color='black', linewidth=0.8, zorder=5)

    ax.set_title(
        f'{label}  ({period})  |  Global mean: {global_mean:+.2f}%\n'
        f'Hatching = not significant ({100-sig_pct:.0f}% of land, p ≥ 0.05)',
        fontsize=12, loc='left'
    )

    last_im = im

plt.suptitle('SPEI-3 whiplash frequency: SAI - CTL\n'
             '10-member ensemble, |ΔSPEI| ≥ 2.0',
             fontsize=14, y=0.95)

cax = fig.add_axes([0.25, 0.07, 0.52, 0.012])
cbar = fig.colorbar(last_im, cax=cax, orientation='horizontal')
cbar.set_label('Whiplash frequency change (%)', fontsize=12)
cbar.ax.tick_params(labelsize=12)

plt.savefig(fig_dir + 'fig1_global_maps_v2.png',
            dpi=300,
            bbox_inches='tight',
            format='png')
print("Saved: fig1_global_maps_v2.png")
