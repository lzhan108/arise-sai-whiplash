# ARISE-SAI Hydroclimate Whiplash Analysis

Analysis code for a study examining whether stratospheric aerosol injection (SAI) can offset warming-driven increases in hydroclimate whiplash, defined as rapid month-to-month transitions between wet and dry conditions.

**Authors:** Leyuan Zhang, Danielle Touma
**Affiliation:** Jackson School of Geosciences, Institute for Geophysics, University of Texas at Austin

---

## Requirements

```bash
conda create -n arise_spei python=3.9
conda activate arise_spei
pip install xarray numpy matplotlib cartopy scipy netCDF4
```

---

## Repository Structure

```
analysis/ # SPEI-3 computation, whiplash frequency, tables
figures_scripts/ # Figure generation scripts
```

---

## Data Access

Model output (ARISE-SAI-1.5, ARISE-SAI-1.0, ARISE-SAI-2045) is publicly available at:
- https://doi.org/10.5065/9kcn-9y79
- https://doi.org/10.26024/0cs0-ev98

Processed SPEI-3 data and this analysis/figure code are archived via Zenodo: https://doi.org/10.5281/zenodo.22774373


---

## Citation

Software citation:

Zhang, L., & Touma, D. (2026). Code for "Winners and Losers: How Stratospheric Aerosol Injection Reshapes Hydroclimate Whiplash Risk" [Software]. Zenodo. https://doi.org/10.5281/zenodo.22774373

---

## Contact

Leyuan Zhang: leyuan.zhang@austin.utexas.edu
