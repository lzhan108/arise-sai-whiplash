#!/usr/bin/env python
# fig7_seasonal_maps_directional_planA_all_scenarios.py
#
# For each scenario (SAI-1.5, SAI-1.0, Delayed-2045):
#   One figure with 2 rows x 4 cols:
#     Row 1: D->W  (DJF | MAM | JJA | SON)
#     Row 2: W->D  (DJF | MAM | JJA | SON)
#   Stippling = not significant (p >= 0.05)
#   Region boxes on each panel
#
# Output: 3 files, one per scenario
#
# Usage:
#   conda activate arise_spei
#   python fig7_seasonal_maps_directional_planA_all_scenarios.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/figures/"
members  = [f"{i:03d}" for i in range(1, 11)]

seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}

region_boxes = {
    'W N America':   (230, 260,  30,  60),
    'NE Brazil':     (315, 345, -15,   5),
    'Mediterranean': (350,  40,  30,  45),
    'W C Africa':    (  5,  30, -10,  15),
    'W Amazon':      (280, 310, -15,   5),
    'N Australia':   (120, 145, -25, -10),
}

sc_configs = [
    ('SAI-1.5',
     'SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064'),
     'SPEI3_CTL_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    ('SAI-1.0',
     'SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064'),
     'SPEI3_CTL_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    ('Delayed-2045',
     'SPEI3_delayed2045_member{}_CTLbaseline.nc','SPEI3',  slice('2045','2064'),
     'SPEI3_CTL_member{}_CTLbaseline.nc',        'SPEI-3', slice('2045','2064')),
]

def draw_region_boxes(ax):
    for rname, (lon_w, lon_e, lat_s, lat_n) in region_boxes.items():
        if lon_w > lon_e:
            ax.plot([lon_w,360,360,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                    transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)
            ax.plot([0,lon_e,lon_e,0,0],[lat_s,lat_s,lat_n,lat_n,lat_s],
                    transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)
        else:
            ax.plot([lon_w,lon_e,lon_e,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                    transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=5)

# ------------------------------------------------------------------ #
for (sc_name, sai_tmpl, sai_var, sai_tslice,
     ctl_tmpl, ctl_var, ctl_tslice) in sc_configs:

    print(f"\nProcessing {sc_name}...")

    dw_sai = {s: [] for s in seasons}
    wd_sai = {s: [] for s in seasons}
    dw_ctl = {s: [] for s in seasons}
    wd_ctl = {s: [] for s in seasons}

    for mem in members:
        ds_sai = xr.open_dataset(save_dir + sai_tmpl.format(mem))
        ds_ctl = xr.open_dataset(save_dir + ctl_tmpl.format(mem))
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

    ds_ref = xr.open_dataset(save_dir + 'SPEI3_CTL_member001_CTLbaseline.nc')
    lats   = ds_ref.lat.values
    lons   = ds_ref.lon.values
    ds_ref.close()
    lons2d, lats2d = np.meshgrid(lons, lats)

    # ── 4 rows x 2 cols ──────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(10, 13),
                             subplot_kw={'projection': ccrs.Robinson()},
                             constrained_layout=True)

    for si, season in enumerate(seasons.keys()):
        for col, (row_label, sai_d, ctl_d) in enumerate([
            ('D→W', dw_sai, dw_ctl),
            ('W→D', wd_sai, wd_ctl),
        ]):
            diff      = (sai_d[season] - ctl_d[season]) * 100
            diff_mean = diff.mean(axis=0)
            _, p_val  = stats.ttest_1samp(diff, popmean=0, axis=0)
            nonsig    = p_val >= 0.05

            ax = axes[si, col]
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#F5F5F0')
            ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')

            im = ax.pcolormesh(lons, lats, diff_mean,
                               transform=ccrs.PlateCarree(),
                               cmap='RdBu_r', vmin=-3, vmax=3, zorder=2)

            ax.scatter(lons2d[nonsig], lats2d[nonsig],
                       transform=ccrs.PlateCarree(),
                       s=0.08, c='black', alpha=0.25, zorder=3)

            draw_region_boxes(ax)

            # Column header (top row only)
            if si == 0:
                ax.set_title(row_label, fontsize=13, fontweight='bold', pad=6)

            # Season label (left col only)
            if col == 0:
                ax.text(-0.04, 0.5, season,
                        transform=ax.transAxes,
                        fontsize=12, fontweight='bold',
                        rotation=90, va='center', ha='right')

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        shrink=0.5, pad=0.02, aspect=40)
    cbar.set_label('Whiplash frequency change (%)', fontsize=10)

    fig.suptitle(f'{sc_name}  |  Seasonal directional whiplash frequency: SAI − CTL\n'
                 f'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  '
                 f'(stippling = p ≥ 0.05, not significant)',
                 fontsize=12)

    sc_str   = sc_name.lower().replace('-','_').replace('.','p')
    out_path = fig_dir + f'fig7_seasonal_maps_directional_planA_vertical_{sc_str}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()

print("\nAll done.")
