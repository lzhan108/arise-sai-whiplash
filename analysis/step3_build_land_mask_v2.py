#!/usr/bin/env python
# step3_build_land_mask_v2.py
#
# Build a reusable land mask combining:
#   1. Land/ocean mask (LANDFRAC > 0.5)
#   2. Desert exclusion (CLM surfdata bare soil PFT >= 80%)
#   3. Glacier/ice exclusion (PCT_GLACIER >= 80%), which also removes
#      essentially all of Antarctica at this grid resolution (verified:
#      100% of 60S-and-south land grid cells fall under the glacier mask)
#
# Literature basis:
#   - SPEIbase (official global SPEI database, spei.csic.es): "Desert
#     and ice areas are masked."
#   - ERA5-Drought (Scientific Data, 2025): recommends masking desert
#     and polar regions for both SPI and SPEI due to poor gamma-fit
#     reliability in these regions.
#   - GRACE-DSI (J. Hydrometeorology, 2017): excludes Antarctica,
#     Greenland, and barren/sparsely vegetated land cover from global
#     drought index comparisons.
#
# Output: land_mask_v2.nc, a boolean DataArray (lat, lon), True = valid
# land grid cell to include in area/regional statistics.
#
# Usage:
#   conda activate arise_spei
#   python step3_build_land_mask_v2.py

import xarray as xr
import numpy as np

proc = '/disk/dtouma/lzhang/arise_FWI/processed/'
out_dir = '/disk/dtouma/lzhang/SPEI_v2/'

# --- Land fraction ---
lf = xr.open_dataset(
    proc + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h0.LANDFRAC.203501-206912.nc'
)['LANDFRAC'].isel(time=0)

# --- Bare soil and glacier fraction from CLM surfdata ---
surf = xr.open_dataset(
    proc + 'surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_c130702.nc'
)
yr = int(np.where(surf['time'].values == 2050)[0][0])
bare = surf['PCT_PFT'].isel(time=yr, lsmpft=0)
bare = bare.rename({'lsmlat': 'lat', 'lsmlon': 'lon'})
bare = bare.assign_coords(lat=lf.lat.values, lon=lf.lon.values)
glac = surf['PCT_GLACIER']
glac = glac.rename({'lsmlat': 'lat', 'lsmlon': 'lon'})
glac = glac.assign_coords(lat=lf.lat.values, lon=lf.lon.values)

# --- Combine into single boolean mask ---
land_mask = (lf > 0.5) & (bare < 80) & (glac < 80)
land_mask.name = 'land_mask'
land_mask.attrs['description'] = (
    'True = valid land grid cell (LANDFRAC>0.5, bare soil PFT<80%, '
    'glacier fraction<80%). Excludes ocean, hyperarid desert, glaciers, '
    'and Antarctica (fully captured by the glacier exclusion).'
)

n_total = land_mask.size
n_valid = int(land_mask.sum())
print(f"Total grid cells: {n_total}")
print(f"Valid land grid cells: {n_valid} ({n_valid/n_total*100:.1f}%)")

out_file = out_dir + 'land_mask_v2.nc'
land_mask.to_netcdf(out_file)
print(f"Saved: {out_file}")
