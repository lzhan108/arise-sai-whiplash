"""
preprocess_monthly.py
=====================
Preprocess downloaded ARISE-SAI-1.0 and ARISE-SAI-1.5-Delayed monthly data:

  Step 1: Load first file separately (Jan of start year) + remaining files
  Step 2: Concatenate along time
  Step 3: Remove duplicate time entries (known issue in both scenarios)
  Step 4: Trim to analysis period (SAI-1.0: 2035-01~2069-12; Delayed: 2045-01~2069-12)
  Step 5: Save one clean file per member per variable

Output files:
  /disk/dtouma/lzhang/ARISE_1p0/monthly_concat/
      PRECT_SAI1p0_member001.nc   (2035-01 ~ 2069-12, 420 months)
      TREFHT_SAI1p0_member001.nc
      ...
  /disk/dtouma/lzhang/ARISE_delayed2045/monthly_concat/
      PRECT_delayed2045_member001.nc  (2045-01 ~ 2069-12, 300 months)
      TREFHT_delayed2045_member001.nc
      ...

Usage:
  conda activate arise
  python preprocess_monthly.py
"""

import xarray as xr
import numpy as np
import glob
import os
import cftime

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
SCENARIOS = {
    "SAI1p0": {
        "raw_dir"  : "/disk/dtouma/lzhang/ARISE_1p0/raw_monthly",
        "out_dir"  : "/disk/dtouma/lzhang/ARISE_1p0/monthly_concat",
        "prefix"   : "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5",
        "time_start": (2035, 2),   # data starts Feb 2035 (Jan missing in source)
        "time_end"  : (2069, 12),  # trim end to Dec 2069
    },
    "delayed2045": {
        "raw_dir"  : "/disk/dtouma/lzhang/ARISE_delayed2045/raw_monthly",
        "out_dir"  : "/disk/dtouma/lzhang/ARISE_delayed2045/monthly_concat",
        "prefix"   : "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045",
        "time_start": (2045, 2),   # data starts Feb 2045 (Jan missing in source)
        "time_end"  : (2069, 12),  # trim end to Dec 2069
    },
}

MEMBERS   = [f"{i:03d}" for i in range(1, 11)]
VARIABLES = ["PRECT", "TREFHT", "RHREFHT", "FSDS", "U10"]


# ------------------------------------------------------------------ #
# Helper: remove duplicate time steps (keep first occurrence)
# ------------------------------------------------------------------ #
def drop_duplicate_times(ds):
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    n_dupes = len(ds.time) - len(unique_idx)
    if n_dupes > 0:
        print(f"    Dropping {n_dupes} duplicate time step(s).")
    return ds.isel(time=unique_idx)


# ------------------------------------------------------------------ #
# Helper: trim to target time range
# ------------------------------------------------------------------ #
def trim_time(ds, start_ym, end_ym):
    """Keep only months within [start_ym, end_ym] inclusive."""
    t = ds.time.values
    # Build mask: year/month within range
    def in_range(t_val):
        y = t_val.year
        m = t_val.month
        return (y > start_ym[0] or (y == start_ym[0] and m >= start_ym[1])) and \
               (y < end_ym[0]   or (y == end_ym[0]   and m <= end_ym[1]))
    mask = np.array([in_range(t_val) for t_val in t])
    n_trimmed = np.sum(~mask)
    if n_trimmed > 0:
        print(f"    Trimming {n_trimmed} time step(s) outside target range.")
    return ds.isel(time=mask)


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #
for scenario_name, cfg in SCENARIOS.items():
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Processing: {scenario_name}")
    print(f"Expected: {cfg['time_start'][0]}-{cfg['time_start'][1]:02d} "
          f"→ {cfg['time_end'][0]}-{cfg['time_end'][1]:02d}")
    print(f"{'='*60}")

    for mem in MEMBERS:
        for var in VARIABLES:
            print(f"\n  Member {mem} | {var}")

            pattern = os.path.join(
                cfg["raw_dir"], mem,
                f"{cfg['prefix']}.{mem}.cam.h0.{var}.*.nc"
            )
            files = sorted(glob.glob(pattern))

            if not files:
                print(f"    WARNING: No files found. Pattern: {pattern}")
                continue

            print(f"    Found {len(files)} file(s).")

            # Load first file separately to capture Jan of start year
            # (the first file only has 1 month but gets dropped by
            #  duplicate removal when loaded together with the overlap file)
            ds_first = xr.open_dataset(files[0], use_cftime=True)[[var]]

            # Load remaining files
            ds_rest = xr.open_mfdataset(
                files[1:],
                combine="nested",
                concat_dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                use_cftime=True,
            )[[var]]

            # Concatenate first month + rest
            ds = xr.concat([ds_first, ds_rest], dim="time")
            ds_first.close()
            ds_rest.close()

            # Remove duplicates
            ds = drop_duplicate_times(ds)

            # Trim to target period
            ds = trim_time(ds, cfg["time_start"], cfg["time_end"])

            # Expected month count
            sy, sm = cfg["time_start"]
            ey, em = cfg["time_end"]
            expected_months = (ey - sy) * 12 + (em - sm + 1)

            print(f"    Time range: {str(ds.time.values[0])[:7]} "
                  f"→ {str(ds.time.values[-1])[:7]}  "
                  f"({len(ds.time)} months, expected {expected_months})")

            if len(ds.time) != expected_months:
                print(f"    WARNING: month count mismatch! "
                      f"Got {len(ds.time)}, expected {expected_months}")

            # Save
            out_file = os.path.join(
                cfg["out_dir"],
                f"{var}_{scenario_name}_member{mem}.nc"
            )
            ds.to_netcdf(out_file)
            print(f"    Saved: {out_file}")
            ds.close()

print("\nAll done.")
