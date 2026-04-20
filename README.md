# ARISE-SAI Hydroclimate Whiplash Analysis

Analysis code for:

> **Stratospheric Aerosol Injection Reduces Hydroclimate Whiplash: Sensitivity to Forcing Intensity and Deployment Timing**
> Leyuan Zhang, Danielle Touma (2026)
> *Earth's Future* (under review)

---

## Overview

This repository contains Python scripts for analyzing the effects of stratospheric aerosol injection (SAI) on hydroclimate whiplash — rapid month-to-month transitions between wet and dry conditions — using the ARISE-SAI ensemble of CESM2-WACCM6 simulations.

Whiplash events are defined as months where the absolute change in SPEI-3 exceeds 2.0 standard deviations (|ΔSPEI-3| ≥ 2.0). Three SAI scenarios are analyzed: ARISE-SAI-1.5, ARISE-SAI-1.0, and ARISE-SAI-2045 (Delayed), compared against an SSP2-4.5 no-intervention control.

---

## Requirements

```bash
conda create -n arise_spei python=3.9
conda activate arise_spei
pip install xarray numpy matplotlib cartopy scipy netCDF4
```

---

## Data Access

Model output is publicly available from the NCAR AWS S3 archive:

```
s3://ncar-cesm2-arise/raw/
```

Processed SPEI-3 files used as inputs to these scripts are stored on the ICICLE2 HPC cluster at UT Austin (`/disk/dtouma/lzhang/SPEI/`) and are available upon reasonable request.

---

## Repository Structure

```
scripts/
├── analysis/                          # Core whiplash frequency calculations
│   ├── preprocess_monthly.py          # Monthly data preprocessing
│   ├── ensemble_whiplash_new_scenarios_CTL2035.py   # Global whiplash (unified CTL2035 baseline, Method B)
│   ├── regional_whiplash_all_scenarios_CTL2035.py   # Regional whiplash (Method B, main analysis)
│   ├── regional_whiplash_all_scenarios_CTL2045_planA.py  # Regional whiplash (Method A, sensitivity)
│   ├── directional_whiplash_all_scenarios.py        # D→W and W→D decomposition
│   ├── seasonal_whiplash_all_scenarios.py           # Seasonal analysis (DJF/MAM/JJA/SON)
│   └── amplitude_whiplash_all_scenarios.py          # Event amplitude analysis
│
└── figures/                           # Figure generation scripts
    ├── fig1_regional_boxplot_member_agreement.py    # Fig 1: Regional boxplot + member agreement
    ├── fig2_global_maps_all_scenarios_stipple_nonsig.py  # Fig 2: Global maps (stippling = not sig.)
    ├── fig3_directional_maps_all_scenarios_stipple_nonsig.py  # Fig 3: Directional maps
    ├── fig4_seasonal_heatmap_directional_all_scenarios.py    # Fig 4: Seasonal heatmap
    ├── fig6_rolling_timeseries_all_scenarios.py     # Fig 6: 10-year rolling time series
    ├── fig7_seasonal_maps_directional_planA_all_scenarios.py # Fig 7: Seasonal maps (all scenarios)
    ├── fig8_member_agreement_heatmap.py             # Fig 8: Member agreement heatmap
    └── fig9_bidirectional_timeseries_planB.py       # Fig 9: D→W and W→D time series
```

---

## Key Parameters

| Parameter | Value |
|---|---|
| SPEI window | 3 months (SPEI-3) |
| Whiplash threshold | \|ΔSPEI-3\| ≥ 2.0 |
| Baseline period | 2015–2034 (CTL) |
| Analysis period | 2035–2064 (SAI-1.5, SAI-1.0); 2045–2064 (Delayed) |
| Ensemble members | 10 |
| Significance test | One-sample t-test (p < 0.05) |
| Cross-scenario baseline | CTL 2035–2064 (Method B) |

---

## Six Target Regions

| Region | Lat | Lon |
|---|---|---|
| W N America | 30–60°N | 230–260°E |
| NE Brazil | 15°S–5°N | 315–345°E |
| Mediterranean | 30–45°N | 350°E–40°E |
| W C Africa | 10°S–15°N | 5–30°E |
| W Amazon | 15°S–5°N | 280–310°E |
| N Australia | 25–10°S | 120–145°E |

---

## Citation

If you use this code, please cite:

Zhang, L., & Touma, D. (2026). Stratospheric Aerosol Injection Reduces Hydroclimate Whiplash: Sensitivity to Forcing Intensity and Deployment Timing. *Earth's Future*. [DOI to be added upon publication]

---

## Contact

Leyuan Zhang — leyuan.zhang@austin.utexas.edu  
Jackson School of Geosciences, Institute for Geophysics  
University of Texas at Austin
