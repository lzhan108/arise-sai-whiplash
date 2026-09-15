#!/usr/bin/env python
# figS1_CTL_warming_signal_v2.py
#
# Warming-driven change in whiplash frequency: CTL future (2035-2064)
# minus CTL baseline (2015-2034), v2.
#
# v2 changes vs figS_CTL_warming_signal.py:
#   - Excluded grid cells (ocean, hyperarid desert, glaciers/Antarctica)
#     are masked out visually (set to NaN) so the map matches the land
#     mask used for all summary statistics
#   - Global mean in the title is now the area-weighted, land-masked
#     mean, matching Table 2
#   - Uses the v2 CTL baseline ensemble file produced by
#     step4_ensemble_whiplash_v2.py
#     (whiplash_freq_CTL_baseline2015_v2_allMembers.nc)
#
# Usage:
#   conda activate arise_spei
#   python figS_CTL_warming_signal_v2.py

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

ens_future   = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_v2_allMembers.nc')
ens_baseline = xr.open_dataarray(spei_dir + 'whiplash_freq_CTL_baseline2015_v2_allMembers.nc')

diff      = ens_future - ens_baseline
diff_mean = diff.mean(dim='member') * 100
_, p_val  = stats.ttest_1samp(diff.values, popmean=0, axis=0)
nonsig_mask = p_val >= 0.05

lons = ens_future.lon.values
lats = ens_future.lat.values

# v2: visually mask out excluded grid cells
diff_mean_masked = diff_mean.where(land_mask)

# v2: area-weighted, land-masked global mean
global_mean = float(area_weighted_mean(diff_mean, land_mask))

fig, ax = plt.subplots(1, 1, figsize=(14, 6),
                       subplot_kw={'projection': ccrs.Robinson()})

ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')
ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#FFFFFF')

im = ax.pcolormesh(lons, lats, diff_mean_masked,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', vmin=-4, vmax=4, zorder=2)

nonsig_float = (nonsig_mask & land_mask.values).astype(float)
ax.contourf(lons, lats, nonsig_float,
            levels=[0.5, 1.5],
            hatches=['////'],
            colors='none',
            transform=ccrs.PlateCarree(),
            zorder=3)

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
    f'CTL future (2035-2064) minus CTL baseline (2015-2034)  |  Global mean: {global_mean:+.2f}%\n'
    f'Hatching = not significant (p ≥ 0.05)',
    fontsize=12, loc='center'
)

fig.suptitle('Warming-driven change in SPEI-3 whiplash frequency under SSP2-4.5',
             fontsize=14, y=1.01)

cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                    pad=0.05, shrink=0.6, aspect=30)
cbar.set_label('Whiplash frequency change (%)', fontsize=12)
cbar.ax.tick_params(labelsize=12)

out_path = fig_dir + 'figS1_CTL_warming_signal_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
print(f"Saved: {out_path}")
