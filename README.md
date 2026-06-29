# Analysis of Tropospheric Noise at Small Spatial Scales Based on GNSS Zenith Total Delay and Numerical Weather Prediction Models

## Overview

This repository contains all scripts, input datasets, and outputs required to reproduce the analyses and figures presented in:

**Olaseinde, A., & Maurer, J.**
*Analysis of Tropospheric Noise at Small Spatial Scales Based on GNSS Zenith Total Delay and Numerical Weather Prediction Models.*

The study evaluates tropospheric residual variability at small spatial scales by comparing Global Navigation Satellite System (GNSS) Zenith Total Delay (ZTD) observations against two Numerical Weather Prediction (NWP) products:

* ERA5 reanalysis
* High-Resolution Rapid Refresh (HRRR)

The repository reproduces all figures and statistical analyses used in the manuscript.

---

## Repository Structure

```text
tropospheric-noise-ztd-nwp/
│
├── data/
│   └── Supplement_Original_Data/
│       ├── ERA5_*.csv
│       └── HRRR_*.csv
│
├── outputs/
│   ├── Figure outputs
│   ├── Summary tables
│   └── Intermediate analysis files
│
├── scripts/
│   ├── figure1_era5_1to1.py
│   ├── figure2_era5_timeseries.py
│   ├── figure3_hrrr_1to1.py
│   ├── figure4_hrrr_timeseries.py
│   ├── figure5_sigma_sp.py
│   ├── figure6_sigma_t.py
│   └── run_all.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Requirements

The repository was developed and tested using:

* Python 3.11+
* NumPy
* Pandas
* Matplotlib
* SciPy
* Scikit-learn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Reproducing All Figures

To reproduce the complete analysis workflow:

```bash
python scripts/run_all.py
```

All generated figures and summary tables will be written automatically to the `outputs/` directory.

---

## Figures

### Figure 1

**ERA5 GNSS–ERA5 ZTD comparison (1:1 analysis)**

### Figure 2

**ERA5 residual time-series analysis**

### Figure 3

**HRRR GNSS–HRRR ZTD comparison (1:1 analysis)**

### Figure 4

**HRRR residual time-series analysis**

### Figure 5

**Spatial variability analysis (σᵖ or σsp)**

### Figure 6

**Temporal variability analysis (σᵗ or σt)**

---

## Data Sources

The repository includes processed datasets required to reproduce the analyses:

* ERA5 residual datasets
* HRRR residual datasets
* GNSS station observations

---

## Citation

If you use this repository or associated methods, please cite:

Olaseinde, A., & Maurer, J.
*Analysis of Tropospheric Noise at Small Spatial Scales Based on GNSS Zenith Total Delay and Numerical Weather Prediction Models.*

---

## Author

**Adeyinka Olaseinde**
Ph.D. Student
Department of Geological Sciences
The University of Alabama

---

## License

This repository is distributed under the MIT License.

