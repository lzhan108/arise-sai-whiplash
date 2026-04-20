#!/usr/bin/env python
# fig6_rolling_timeseries_all_scenarios.py
#
# Rolling 10-year whiplash frequency time series per region
# Thick line = 10-year moving ensemble median
# Shading = ensemble IQR (25th-75th percentile)
# 6 subplots (one per region), 3 scenarios + CTL
#
# Usage:
#   conda activate arise_spei
#   python fig6_rolling_timeseries_all_scenarios.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

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

# Scenarios: name, file template, varname, time slice, CTL to use
scenarios = {
    'CTL':     ('SPEI3_CTL_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'SAI-1.5': ('SPEI3_SAI_member{}_CTLbaseline.nc',         'SPEI-3', slice('2035','2064')),
    'SAI-1.0': ('SPEI3_SAI1p0_member{}_CTLbaseline.nc',      'SPEI3',  slice('2035','2064')),
    'Delayed': ('SPEI3_delayed2045_member{}_CTLbaseline.nc',  'SPEI3',  slice('2045','2064')),
}

colors = {
    'CTL':     '#4C72B0',
    'SAI-1.5': '#DD8452',
    'SAI-1.0': '#55A868',
    'Delayed': '#C44E52',
}

# ------------------------------------------------------------------ #
# Compute annual whiplash frequency per member per region
# ------------------------------------------------------------------ #
print("Computing annual whiplash frequency per region...")

# annual_data[sc][region] = (n_members, n_years) array
annual_data = {sc: {r: [] for r in regions} for sc in scenarios}

for sc_name, (fname_tmpl, varname, tslice) in scenarios.items():
    for mem in members:
        ds   = xr.open_dataset(save_dir + fname_tmpl.format(mem))
        spei = ds[varname].sel(time=tslice)
        wh   = np.abs(spei.diff(dim='time')) >= 2.0  # (time, lat, lon)

        # Annual mean per region
        years = np.unique(wh.time.dt.year.values)
        for rname, (lat_s, lat_n, lon_w, lon_e) in regions.items():
            mask = region_mask(wh, lat_s, lat_n, lon_w, lon_e)
            annual_vals = []
            for yr in years:
                wh_yr = wh.sel(time=wh.time.dt.year == yr)
                val   = float(wh_yr.where(mask).mean()) * 100
                annual_vals.append(val)
            annual_data[sc_name][rname].append(annual_vals)
        ds.close()
    print(f"  {sc_name} done.")

# Convert to numpy arrays
for sc in scenarios:
    for rname in regions:
        annual_data[sc][rname] = np.array(annual_data[sc][rname])  # (10, n_years)

# Year arrays per scenario
year_arrays = {
    'CTL':     np.arange(2035, 2065),
    'SAI-1.5': np.arange(2035, 2065),
    'SAI-1.0': np.arange(2035, 2065),
    'Delayed': np.arange(2045, 2065),
}

# ------------------------------------------------------------------ #
# Rolling 10-year window
# ------------------------------------------------------------------ #
def rolling_stats(data_2d, window=10):
    """
    data_2d: (n_members, n_years)
    Returns: median, q25, q75 each of shape (n_years,)
    """
    n_members, n_years = data_2d.shape
    # Rolling mean per member first
    rolled = np.array([
        uniform_filter1d(data_2d[m], size=window, mode='reflect')
        for m in range(n_members)
    ])
    median = np.median(rolled, axis=0)
    q25    = np.percentile(rolled, 25, axis=0)
    q75    = np.percentile(rolled, 75, axis=0)
    return median, q25, q75

# ------------------------------------------------------------------ #
# Plot: 2 rows x 3 cols, one subplot per region
# ------------------------------------------------------------------ #
region_names = list(regions.keys())
fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.3})
axes_flat = axes.flatten()

for ri, rname in enumerate(region_names):
    ax = axes_flat[ri]

    for sc_name in ['CTL', 'SAI-1.5', 'SAI-1.0', 'Delayed']:
        years  = year_arrays[sc_name]
        data2d = annual_data[sc_name][rname]
        med, q25, q75 = rolling_stats(data2d, window=10)

        color = colors[sc_name]
        lw    = 2.0 if sc_name != 'CTL' else 1.8
        ls    = '--' if sc_name == 'CTL' else '-'

        ax.plot(years, med, color=color, linewidth=lw,
                linestyle=ls, label=sc_name, zorder=3)
        ax.fill_between(years, q25, q75, color=color,
                        alpha=0.18, zorder=2)

    # Near future / Far future shading
    ax.axvspan(2035, 2049, alpha=0.07, color='steelblue', zorder=1)
    ax.axvspan(2050, 2064, alpha=0.07, color='firebrick', zorder=1)

    ax.set_title(rname, fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Whiplash frequency (%)', fontsize=9)
    ax.set_xlim(2035, 2064)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add Near/Far labels after ylim is set
for ri, rname in enumerate(region_names):
    ax = axes_flat[ri]
    ymax = ax.get_ylim()[1]
    for txt, x, c in [('Near future', 2042, 'steelblue'),
                       ('Far future',  2057, 'firebrick')]:
        ax.text(x, ymax * 0.97, txt, ha='center', va='top',
                fontsize=7.5, color=c, alpha=0.8)

# Legend
handles = [
    plt.Line2D([0],[0], color=colors['CTL'],     lw=1.8, ls='--', label='CTL (2035–2064)'),
    plt.Line2D([0],[0], color=colors['SAI-1.5'], lw=2.0, label='SAI-1.5'),
    plt.Line2D([0],[0], color=colors['SAI-1.0'], lw=2.0, label='SAI-1.0'),
    plt.Line2D([0],[0], color=colors['Delayed'],  lw=2.0, label='Delayed-2045'),
    mpatches.Patch(color='gray', alpha=0.2, label='Ensemble IQR'),
]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.04),
           framealpha=0.9, edgecolor='#CCCCCC')

fig.suptitle('Regional whiplash frequency: 10-year rolling median\n'
             'SPEI-3, |ΔSPEI| ≥ 2.0, 10-member ensemble  '
             '(shading = IQR)',
             fontsize=12)

out_path = fig_dir + 'fig6_rolling_timeseries_all_scenarios.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
