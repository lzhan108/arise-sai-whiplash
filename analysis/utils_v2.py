#!/usr/bin/env python
# utils_v2.py
#
# Shared utility functions for the v2 pipeline:
#   - land mask loading
#   - region box definitions (Touma et al. 2023 coordinates)
#   - region masking (handles boxes crossing the 0-meridian)
#   - area-weighted spatial mean (cos(lat) weighting), matching the
#     manuscript's Methods text: "area-weighted spatial mean"
#
# Import this module from any downstream v2 script:
#   from utils_v2 import load_land_mask, REGIONS, region_mask, area_weighted_mean

import xarray as xr
import numpy as np

LAND_MASK_FILE = '/disk/dtouma/lzhang/SPEI_v2/land_mask_v2.nc'

# Region boxes (Touma et al. 2023 corrected coordinates, 0-360E)
REGIONS = {
    'W N America':   (30,    45,   235,   244.5),
    'NE Brazil':     (-17.5,  2.5, 300,   327),
    'Mediterranean': (36,    47,   349.5,  48),
    'W C Africa':    (-8,     1,     8,    28),
    'W Amazon':      (-14.5, -0.5, 278,   300),
    'N Australia':   (-20,   -5,   120,   150),
}


def load_land_mask():
    """Load the v2 land mask (LANDFRAC>0.5, desert<80%, glacier<80%)."""
    return xr.open_dataarray(LAND_MASK_FILE)


def region_mask(da, lat_s, lat_n, lon_w, lon_e):
    """Boolean mask for a lat/lon box, handling boxes that cross 0E
    (e.g. Mediterranean: 349.5E to 48E)."""
    if lon_w > lon_e:
        return ((da.lat >= lat_s) & (da.lat <= lat_n) &
                ((da.lon >= lon_w) | (da.lon <= lon_e)))
    else:
        return ((da.lat >= lat_s) & (da.lat <= lat_n) &
                (da.lon >= lon_w) & (da.lon <= lon_e))


def area_weighted_mean(da, mask=None):
    """Area-weighted (cos(lat)) spatial mean over lat/lon dims.

    da   : DataArray with 'lat' and 'lon' dims (may also have other
           dims such as 'member', preserved in the output)
    mask : optional boolean DataArray (lat, lon) restricting the mean
           to specific grid cells (e.g. land mask, region mask, or
           their combination via `land_mask & region_mask(...)`)

    xarray's .weighted() accepts 1D lat-only weights (broadcasting them
    over lon automatically) and automatically excludes NaN data points
    from both the weighted sum and the weight normalization, so masking
    is applied simply via da.where(mask) beforehand.
    """
    weights = np.cos(np.deg2rad(da.lat))
    if mask is not None:
        da = da.where(mask)
    return da.weighted(weights).mean(dim=['lat', 'lon'])
