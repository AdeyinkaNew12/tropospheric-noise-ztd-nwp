
import subprocess
import sys

scripts = [
    "figure1_era5_1to1.py",
    "figure2_era5_timeseries.py",
    "figure3_hrrr_1to1.py",
    "figure4_hrrr_timeseries.py",
    "figure6_sigma_sp.py",
    "figure7_sigma_t.py"
]

for s in scripts:
    print("="*60)
    print("Running:", s)
    print("="*60)

    subprocess.run(
        [sys.executable, s],
        check=True
    )

print("\nAll analyses completed.")
