import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = BASE_DIR / "scripts"

SCRIPTS = [
    "figure1_era5_1to1.py",
    "figure2_era5_timeseries.py",
    "figure3_hrrr_1to1.py",
    "figure4_hrrr_timeseries.py",
    "figure5_sigma_sp.py",
    "figure6_sigma_t.py",
]

def main():
    for script_name in SCRIPTS:
        script_path = SCRIPT_DIR / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        print("=" * 70)
        print(f"Running: {script_name}")
        print("=" * 70)

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            check=True,
        )

    print("\n✅ All analyses completed successfully.")

if __name__ == "__main__":
    main()
