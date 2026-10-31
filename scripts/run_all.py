#!/usr/bin/env python3
"""
Reproduce GNSS–NWP manuscript Figures 1–7,
Supplementary Figure S8, and supplementary tables.

Run from the repository root:

    python3 scripts/run_all.py
"""

from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = BASE_DIR / "scripts"

FIGURE_SCRIPTS = [
    "figure1_world_map.py",
    "figure2_era5_1to1.py",
    "figure3_era5_timeseries.py",
    "figure4_hrrr_1to1.py",
    "figure5_hrrr_timeseries.py",
    "figure6_sigma_sp.py",
    "figure7_sigma_t.py",
]

SUPPLEMENTARY_SCRIPTS = [
    "generate_bias_mae_sigma_summary.py",
    "generate_table_s1.py",
    "generate_table_s6_station_spacing.py",
    "generate_table_s7_station_observations.py",
    "generate_supplementary_word_tables.py",
    "format_table_s6.py",
]

STATION_CSV = (
    BASE_DIR
    / "data"
    / "Supplement_Original_Data"
    / "StationMapfile_VERIFIED.csv"
)

FIGURE_S8 = SCRIPT_DIR / "supplementary_station_map.py"

FIGURE_S8_OUTPUT = (
    BASE_DIR
    / "figures"
    / "Figure S8.png"
)

def execute(script_name, extra_args=None):
    script = SCRIPT_DIR / script_name

    if not script.is_file():
        raise FileNotFoundError(script)

    command = [sys.executable, str(script)]

    if extra_args:
        command.extend(extra_args)

    print("\n" + "=" * 72, flush=True)
    print("RUNNING:", script_name, flush=True)
    print("=" * 72, flush=True)

    start = time.time()

    subprocess.run(
        command,
        cwd=str(BASE_DIR),
        check=True
    )

    elapsed = time.time() - start

    print(
        f"COMPLETED: {script_name} ({elapsed:.1f} seconds)",
        flush=True
    )

def main():
    print("=" * 72, flush=True)
    print("GNSS–NWP MANUSCRIPT REPRODUCIBILITY WORKFLOW", flush=True)
    print("=" * 72, flush=True)

    print("\nRepository:", BASE_DIR, flush=True)

    if not STATION_CSV.is_file():
        raise FileNotFoundError(
            f"Figure S8 station CSV missing: {STATION_CSV}"
        )

    for script_name in FIGURE_SCRIPTS:
        execute(script_name)

    for script_name in SUPPLEMENTARY_SCRIPTS:
        execute(script_name)

    FIGURE_S8_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    execute(
        FIGURE_S8.name,
        [
            "--stations", str(STATION_CSV),
            "--output", str(FIGURE_S8_OUTPUT),
        ]
    )

    print("\n" + "=" * 72, flush=True)
    print("ALL MANUSCRIPT SCRIPTS COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 72, flush=True)

    print("\nFigure 1 PDF:")
    print(BASE_DIR / "figures" / "Figure1_GNSS_World_Map.pdf")

    print("\nFigure S8:")
    print(FIGURE_S8_OUTPUT)

    print("\nSupplementary tables and other figures:")
    print(BASE_DIR / "outputs")

if __name__ == "__main__":
    main()


# AUTO_MANUSCRIPT_OUTPUT_ORGANIZATION
if __name__ == "__main__":
    import subprocess as _organizer_subprocess
    import sys as _organizer_sys

    _organizer_subprocess.run(
        [
            _organizer_sys.executable,
            str(__import__("pathlib").Path(__file__).resolve().parent / "organize_manuscript_outputs.py"),
        ],
        check=True,
    )
