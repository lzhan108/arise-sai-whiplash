#!/usr/bin/env python
# figS2_seasonal_maps_v2.py
#
# Seasonal directional whiplash maps, v2. One figure per scenario
# (SAI-1.5, SAI-1.0, Delayed-2045): 4 rows (DJF/MAM/JJA/SON) x 2 cols
# (D->W, W->D).
#
# v2 changes vs figS1_seasonal_maps_directional_planA_all_scenarios_v3.py:
#   - Excluded grid cells (ocean, hyperarid desert, glaciers/Antarctica)
#     are masked out visually (set to NaN) so each map matches the
#     land mask used for all summary statistics
#
# Usage:
#   conda activate arise_spei
#   python figS1_seasonal_maps_v2.py

import sys
sys.path.insert(0, '/home/staff/lzhang/arise_whiplash/v2/analysis')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from utils_v2 import load_land_mask, REGIONS

spei_dir = "/disk/dtouma/lzhang/SPEI_v2/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/v2/figures_output/"
members  = [f"{i:03d}" for i in range(1, 11)]

land_mask = load_land_mask()
land_mask_np = land_mask.values

seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}

sc_configs = [
    ('SAI-1.5',
     'SPEI3_SAI15_v2_member{}_CTLbaseline.nc',       'SPEI-3', slice('2035', '2064'),
     'SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
    ('SAI-1.0',
     'SPEI3_SAI1p0_v2_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035', '2064'),
     'SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
    ('Delayed-2045',
     'SPEI3_delayed2045_v2_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045', '2064'),
     'SPEI3_CTL_v2_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035', '2064')),
]


def draw_region_boxes(ax):
    for rname, (lat_s, lat_n, lon_w, lon_e) in REGIONS.items():
        if lon_w > lon_e:
            ax.plot([lon_w,360,360,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                    transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)
            ax.plot([0,lon_e,lon_e,0,0],[lat_s,lat_s,lat_n,lat_n,lat_s],
                    transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)
        else:
            ax.plot([lon_w,lon_e,lon_e,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                    transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)


for (sc_name, sai_tmpl, sai_var, sai_tslice,
     ctl_tmpl, ctl_var, ctl_tslice) in sc_configs:

    print(f"\nProcessing {sc_name}...")

    dw_sai = {s: [] for s in seasons}
    wd_sai = {s: [] for s in seasons}
    dw_ctl = {s: [] for s in seasons}
    wd_ctl = {s: [] for s in seasons}

    for mem in members:
        ds_sai = xr.open_dataset(spei_dir + sai_tmpl.format(mem))
        ds_ctl = xr.open_dataset(spei_dir + ctl_tmpl.format(mem))
        d_sai  = ds_sai[sai_var].sel(time=sai_tslice).diff(dim='time')
        d_ctl  = ds_ctl[ctl_var].sel(time=ctl_tslice).diff(dim='time')

        for season, months in seasons.items():
            dw_sai[season].append((d_sai >=  2.0).sel(time=d_sai.time.dt.month.isin(months)).mean(dim='time').values)
            wd_sai[season].append((d_sai <= -2.0).sel(time=d_sai.time.dt.month.isin(months)).mean(dim='time').values)
            dw_ctl[season].append((d_ctl >=  2.0).sel(time=d_ctl.time.dt.month.isin(months)).mean(dim='time').values)
            wd_ctl[season].append((d_ctl <= -2.0).sel(time=d_ctl.time.dt.month.isin(months)).mean(dim='time').values)
        ds_sai.close(); ds_ctl.close()
        print(f"  member {mem} done")

    for s in seasons:
        dw_sai[s] = np.stack(dw_sai[s], axis=0)
        wd_sai[s] = np.stack(wd_sai[s], axis=0)
        dw_ctl[s] = np.stack(dw_ctl[s], axis=0)
        wd_ctl[s] = np.stack(wd_ctl[s], axis=0)

    ds_ref = xr.open_dataset(spei_dir + 'SPEI3_CTL_v2_member001_CTLbaseline.nc')
    lats   = ds_ref.lat.values
    lons   = ds_ref.lon.values
    ds_ref.close()

    fig, axes = plt.subplots(4, 2, figsize=(10, 13),
                             subplot_kw={'projection': ccrs.Robinson()},
                             constrained_layout=True)

    for si, season in enumerate(seasons.keys()):
        for col, (row_label, sai_d, ctl_d) in enumerate([
            ('D->W', dw_sai, dw_ctl),
            ('W->D', wd_sai, wd_ctl),
        ]):
            diff      = (sai_d[season] - ctl_d[season]) * 100
            diff_mean = diff.mean(axis=0)
            _, p_val  = stats.ttest_1samp(diff, popmean=0, axis=0)
            nonsig    = p_val >= 0.05

            # v2: visually mask out excluded grid cells
            diff_mean_masked = np.where(land_mask_np, diff_mean, np.nan)

            ax = axes[si, col]
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#FFFFFF')
            ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')

            im = ax.pcolormesh(lons, lats, diff_mean_masked,
                               transform=ccrs.PlateCarree(),
                               cmap='RdBu_r', vmin=-3, vmax=3, zorder=2)

            nonsig_float = (nonsig & land_mask_np).astype(float)
            ax.contourf(lons, lats, nonsig_float,
                        levels=[0.5, 1.5],
                        hatches=['////'],
                        colors='none',
                        transform=ccrs.PlateCarree(),
                        zorder=3)

            draw_region_boxes(ax)

            if si == 0:
                ax.set_title(row_label, fontsize=13, fontweight='bold', pad=6)

            if col == 0:
                ax.text(-0.04, 0.5, season,
                        transform=ax.transAxes,
                        fontsize=12, fontweight='bold',
                        rotation=90, va='center', ha='right')

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        shrink=0.5, pad=0.02, aspect=40)
    cbar.set_label('Whiplash frequency change (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f'{sc_name}  |  Seasonal directional whiplash frequency: SAI - CTL\n'
                 f'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  '
                 f'(hatching = p ≥ 0.05, not significant)',
                 fontsize=14)

    sc_str   = sc_name.lower().replace('-','_').replace('.','p')
    out_path = fig_dir + f'figS2_seasonal_maps_{sc_str}_v2.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()

print("\nAll done.")
