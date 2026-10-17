# Analysis of Tropospheric Noise at Small Spatial Scales Based on GNSS Zenith Total Delay and Numerical Weather Prediction Models

## Overview

This repository contains the Python scripts and processed input datasets needed to reproduce the manuscript analyses of GNSS Zenith Total Delay (ZTD) residual variability using ERA5 and HRRR numerical weather prediction products.

Authors: Adeyinka Olaseinde and James Maurer.

The complete workflow generates 26 manuscript outputs: seven main figures, one supplementary figure, nine supplementary table files, and nine analysis CSV files.

## Repository Contents

- scripts/ — Python scripts for manuscript figures, statistics, tables, and workflow execution.
- data/Supplement_Original_Data/ — Processed ERA5 and HRRR datasets and verified GNSS station metadata.
- requirements.txt — Python dependencies.
- README.md — Reproduction instructions.
- .gitignore — Excludes generated results and temporary files.

Generated manuscript outputs are intentionally excluded from GitHub.

## Requirements

Python 3.11 or newer is recommended.

Install the dependencies:

    python3 -m pip install -r requirements.txt

Dependencies include NumPy, Pandas, Matplotlib, SciPy, Scikit-learn, Cartopy, PyProj, python-docx, and openpyxl.

## Reproduce the Manuscript

Clone the repository:

    git clone https://github.com/AdeyinkaNew12/tropospheric-noise-ztd-nwp.git
    cd tropospheric-noise-ztd-nwp

Install the dependencies:

    python3 -m pip install -r requirements.txt

Run the complete workflow:

    python3 scripts/run_all.py

## Generated Outputs

After execution, manuscript results are organized into:

- manuscript_outputs/Main_Figures/ — 7 files
- manuscript_outputs/Supplementary_Figures/ — 1 file
- manuscript_outputs/Supplementary_Tables/ — 9 files
- manuscript_outputs/Analysis_CSVs/ — 9 files
- manuscript_outputs/OUTPUT_VERIFICATION_REPORT.csv — verification report

The verification report is additional to the 26 manuscript output files.

## Main Figures

1. Figure 1 — Global GNSS station location map.
2. Figure 2 — ERA5 GNSS–NWP ZTD comparison.
3. Figure 3 — ERA5 residual time series.
4. Figure 4 — HRRR GNSS–NWP ZTD comparison.
5. Figure 5 — HRRR residual time series.
6. Figure 6 — Spatial residual variability.
7. Figure 7 — Temporal residual variability.

## Supplementary Figure

Figure S8 — Topographic maps of the GNSS station study regions.

## Supplementary Tables and Analysis CSVs

The workflow generates supporting station metadata, station-spacing statistics, observation summaries, station-level metrics, and model-comparison results.

## Input Data

Processed ERA5 and HRRR residual datasets and GNSS station metadata are included under data/Supplement_Original_Data/.

The workflow uses repository-relative paths and does not require files from the original authors Desktop.

## Reproducibility Notes

- Generated figures, tables, and analysis outputs remain on the computer running the workflow and are excluded from GitHub.
- The supplementary topographic map may require internet access to retrieve terrain imagery.
- The workflow passed a clean Git-snapshot execution test using an existing Python environment.
- Installation of all dependencies on a fresh computer has not yet been independently verified.

## Citation

Olaseinde, A., & Maurer, J. Analysis of Tropospheric Noise at Small Spatial Scales Based on GNSS Zenith Total Delay and Numerical Weather Prediction Models.

## Author

Adeyinka Olaseinde
Department of Geological Sciences
The University of Alabama

## License

See the LICENSE file for applicable licensing terms.
