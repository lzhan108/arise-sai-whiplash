#!/usr/bin/env python
# fig3_directional_maps_all_scenarios.py
#
# 2 rows x 3 cols:
#   Row 1: D->W  (SAI-1.5 | SAI-1.0 | Delayed-2045)  SAI - CTL diff
#   Row 2: W->D  (SAI-1.5 | SAI-1.0 | Delayed-2045)  SAI - CTL diff
#
# Usage:
#   conda activate arise_spei
#   python fig3_directional_maps_all_scenarios.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

save_dir = "/disk/dtouma/lzhang/SPEI/"
fig_dir  = "/home/staff/lzhang/arise_whiplash/figures/"
members  = [f"{i:03d}" for i in range(1, 11)]

region_boxes = {
    'W N America':   (230, 260,  30,  60),
    'NE Brazil':     (315, 345, -15,   5),
    'Mediterranean': (350,  40,  30,  45),
    'W C Africa':    (  5,  30, -10,  15),
    'W Amazon':      (280, 310, -15,   5),
    'N Australia':   (120, 145, -25, -10),
}

scenarios = {
    'SAI-1.5': ('SPEI3_SAI_member{}_CTLbaseline.nc',        'SPEI-3', slice('2035','2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',     'SPEI3',  slice('2035','2064')),
    'Delayed': ('SPEI3_delayed2045_member{}_CTLbaseline.nc', 'SPEI3',  slice('2045','2064')),
    'CTL_35':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'CTL_45':  ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2045','2064')),
}

# ------------------------------------------------------------------ #
# Compute per-member D->W and W->D frequency
# ------------------------------------------------------------------ #
print("Computing directional whiplash...")
ens_dw, ens_wd = {}, {}

for sc_name, (fname_tmpl, varname, tslice) in scenarios.items():
    dw_list, wd_list = [], []
    for mem in members:
        ds   = xr.open_dataset(save_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        delta = spei.diff(dim='time')
        dw_list.append((delta >=  2.0).mean(dim='time').values)
        wd_list.append((delta <= -2.0).mean(dim='time').values)
        ds.close()
    ens_dw[sc_name] = np.stack(dw_list, axis=0)  # (10, lat, lon)
    ens_wd[sc_name] = np.stack(wd_list, axis=0)
    print(f"  {sc_name} done.")

ds_ref = xr.open_dataset(save_dir + 'SPEI3_CTL_member001_CTLbaseline.nc')
lats = ds_ref.lat.values
lons = ds_ref.lon.values
ds_ref.close()
lons2d, lats2d = np.meshgrid(lons, lats)

# ------------------------------------------------------------------ #
# Plot: 2 rows (D->W, W->D) x 3 cols (SAI-1.5, SAI-1.0, Delayed)
# ------------------------------------------------------------------ #
sc_pairs = [
    ('SAI-1.5', 'CTL_35', '2035–2064'),
    ('SAI-1.0', 'CTL_35', '2035–2064'),
    ('Delayed', 'CTL_45', '2045–2064'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9),
                         subplot_kw={'projection': ccrs.Robinson()},
                         gridspec_kw={'hspace': 0.12, 'wspace': 0.04})

for col, (sc, ctl, period) in enumerate(sc_pairs):
    for row, (direction, ens_dir) in enumerate([('D→W', ens_dw), ('W→D', ens_wd)]):

        diff      = (ens_dir[sc] - ens_dir[ctl]) * 100   # %
        diff_mean = diff.mean(axis=0)
        _, p_val  = stats.ttest_1samp(diff, popmean=0, axis=0)
        sig_mask  = p_val < 0.05

        ax = axes[row, col]
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND,  zorder=0, facecolor='#F5F5F0')
        ax.add_feature(cfeature.OCEAN, zorder=4, facecolor='#E8E8E8')

        im = ax.pcolormesh(lons, lats, diff_mean,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r', vmin=-3, vmax=3, zorder=2)

        # Stippling: non-significant grid points
        nonsig_mask = ~sig_mask
        ax.scatter(lons2d[nonsig_mask], lats2d[nonsig_mask],
                   transform=ccrs.PlateCarree(),
                   s=0.08, c='black', alpha=0.25, zorder=3)

        # Region boxes
        for rname, (lon_w, lon_e, lat_s, lat_n) in region_boxes.items():
            if lon_w > lon_e:
                ax.plot([lon_w,360,360,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                        transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=4)
                ax.plot([0,lon_e,lon_e,0,0],[lat_s,lat_s,lat_n,lat_n,lat_s],
                        transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=4)
            else:
                ax.plot([lon_w,lon_e,lon_e,lon_w,lon_w],[lat_s,lat_s,lat_n,lat_n,lat_s],
                        transform=ccrs.PlateCarree(), color='k', linewidth=0.9, zorder=4)

        # Column title (top row only)
        if row == 0:
            ax.set_title(f'{sc}  ({period})\nGlobal mean: {diff_mean.mean():+.2f}%',
                         fontsize=10, loc='left')

        # Row label (left col only)
        if col == 0:
            ax.text(-0.04, 0.5, direction, transform=ax.transAxes,
                    fontsize=13, fontweight='bold', rotation=90,
                    va='center', ha='right')

# Shared colorbar at bottom
cax = fig.add_axes([0.25, 0.04, 0.5, 0.018])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Whiplash frequency change (%)', fontsize=10)

fig.suptitle('SPEI-3 directional whiplash frequency: SAI − CTL\n'
             '10-member ensemble, |ΔSPEI| ≥ 2.0  (stippling = p ≥ 0.05, not significant)',
             fontsize=12, y=1.01)

plt.subplots_adjust(bottom=0.1)
out_path = fig_dir + 'fig3_directional_maps_all_scenarios_stipple_nonsig.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
