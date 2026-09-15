#!/usr/bin/env python
# fig4_directional_maps_v2.py
#
# Directional whiplash maps (D->W, W->D), v2.
# 2 rows x 3 cols: SAI-1.5 | SAI-1.0 | Delayed-2045, each SAI - CTL.
#
# v2 changes vs fig4_directional_maps_all_scenarios_v3.py:
#   - Excluded grid cells (ocean, hyperarid desert, glaciers/Antarctica)
#     are masked out visually (set to NaN) so the map matches the land
#     mask used for all summary statistics
#   - Global mean in each panel title is now the area-weighted,
#     land-masked mean
#
# Usage:
#   conda activate arise_spei
#   python step10_fig4_v2.py

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
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()

SCENARIOS = {
    'SAI-1.5': ('SPEI3_SAI15_v2_member{}_CTLbaseline.nc',       'SPEI-3', slice('2035', '2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035', '2064')),
    'Delayed': ('SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045', '2064')),
    'CTL_35':  ('SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
}

# ------------------------------------------------------------------ #
# Compute per-member D->W and W->D frequency
# ------------------------------------------------------------------ #
print("Computing directional whiplash...")
ens_dw, ens_wd = {}, {}

for sc_name, (fname_tmpl, varname, tslice) in SCENARIOS.items():
    dw_list, wd_list = [], []
    for mem in members:
        ds   = xr.open_dataset(spei_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = spei.diff(dim='time')
        dw_list.append((delta >=  2.0).mean(dim='time').values)
        wd_list.append((delta <= -2.0).mean(dim='time').values)
        ds.close()
    ens_dw[sc_name] = np.stack(dw_list, axis=0)
    ens_wd[sc_name] = np.stack(wd_list, axis=0)
    print(f"  {sc_name} done.")

ds_ref = xr.open_dataset(spei_dir + 'SPEI3_CTL_v2_member001_CTLbaseline.nc')
lats = ds_ref.lat.values
lons = ds_ref.lon.values
ds_ref.close()

land_mask_np = land_mask.values

# ------------------------------------------------------------------ #
# Plot: 2 rows x 3 cols
# ------------------------------------------------------------------ #
sc_pairs = [
    ('SAI-1.5', 'CTL_35', '2035-2064'),
    ('SAI-1.0', 'CTL_35', '2035-2064'),
    ('Delayed', 'CTL_35', '2045-2064 vs CTL 2035-2064'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 7),
                         subplot_kw={'projection': ccrs.Robinson()},
                         gridspec_kw={'hspace': 0.05, 'wspace': 0.04})

for col, (sc, ctl, period) in enumerate(sc_pairs):
    for row, (direction, ens_dir) in enumerate([('D->W', ens_dw), ('W->D', ens_wd)]):

        diff      = (ens_dir[sc] - ens_dir[ctl]) * 100
        diff_mean = diff.mean(axis=0)
        _, p_val  = stats.ttest_1samp(diff, popmean=0, axis=0)
        sig_mask  = p_val < 0.05
        nonsig_mask = ~sig_mask

        # v2: visually mask out excluded grid cells
        diff_mean_masked = np.where(land_mask_np, diff_mean, np.nan)

        # v2: area-weighted, land-masked global mean
        diff_mean_da = xr.DataArray(diff_mean, dims=['lat', 'lon'],
                                     coords={'lat': lats, 'lon': lons})
        global_mean = float(area_weighted_mean(diff_mean_da, land_mask))

        ax = axes[row, col]
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#FFFFFF')
        ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')

        im = ax.pcolormesh(lons, lats, diff_mean_masked,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r', vmin=-3, vmax=3, zorder=2)

        nonsig_float = (nonsig_mask & land_mask_np).astype(float)
        ax.contourf(lons, lats, nonsig_float,
                    levels=[0.5, 1.5],
                    hatches=['////'],
                    colors='none',
                    transform=ccrs.PlateCarree(),
                    zorder=3)

        for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
            if lon_w > lon_e:
                ax.plot([lon_w,360,360,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                        transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)
                ax.plot([0,lon_e,lon_e,0,0],[lat_s,lat_s,lat_n,lat_n,lat_s],
                        transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)
            else:
                ax.plot([lon_w,lon_e,lon_e,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                        transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)

        if row == 0:
            ax.set_title(f'{sc}  ({period})\nD->W global mean: {global_mean:+.2f}%',
                         fontsize=12, loc='left')
        else:
            ax.set_title(f'W->D global mean: {global_mean:+.2f}%',
                         fontsize=12, loc='left')

        if col == 0:
            ax.text(-0.04, 0.5, direction, transform=ax.transAxes,
                    fontsize=13, fontweight='bold', rotation=90,
                    va='center', ha='right')

cax = fig.add_axes([0.25, 0.04, 0.5, 0.018])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Whiplash frequency change (%)', fontsize=12)
cbar.ax.tick_params(labelsize=12)

fig.suptitle('SPEI-3 directional whiplash frequency: SAI - CTL\n'
             '10-member ensemble, |ΔSPEI| ≥ 2.0  (hatching = p ≥ 0.05, not significant)',
             fontsize=14, y=0.99)

plt.subplots_adjust(top=0.85, bottom=0.1)
out_path = fig_dir + 'fig4_directional_maps_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
