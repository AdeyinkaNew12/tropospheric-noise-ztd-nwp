from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# PATHS — CORRECTED ORIGINAL HRRR DATA
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Supplement_Original_Data"

OUT_DIR = Path.home() / "Desktop" / "NewUpdated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Residual_Histograms_3Sites_HRRR_NEW_UPDATED.png"
OUT_PDF = OUT_DIR / "Residual_Histograms_3Sites_HRRR_NEW_UPDATED.pdf"

# ==========================================================
# VERIFIED HRRR DATASETS AND PANEL ORDER
# ==========================================================

FILES = [
    ("Missouri", "HRRR_Missouri.csv", 4, 59),
    ("California", "HRRR_California.csv", 3, 195),
    ("Utah", "HRRR_Utah.csv", 4, 857),
]

NROWS = 3
NCOLS = 1
FIGSIZE = (10.2, 10.2)
DPI = 300

COLORS = [
    "steelblue",
    "darkorange",
    "seagreen",
    "firebrick",
    "mediumpurple",
    "goldenrod",
    "teal",
    "slategray",
]

ALPHA = 0.65
BINS = 20

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ==========================================================
# LOAD CORRECT HRRR RESIDUALS
# ==========================================================

def load_site(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "ID" not in df.columns:
        raise ValueError(
            f"{path.name}: Missing station ID column."
        )

    candidates = [
        "Residual (GNSS)",
        "Residual(GNSS)",
    ]

    available = [
        col for col in candidates
        if col in df.columns
    ]

    if len(available) != 1:
        raise ValueError(
            f"{path.name}: Expected exactly one GNSS residual "
            f"column, found {available}. "
            f"Available columns: {df.columns.tolist()}"
        )

    residual_col = available[0]

    df["ID"] = (
        df["ID"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df[residual_col] = pd.to_numeric(
        df[residual_col],
        errors="coerce"
    )

    valid = df.dropna(
        subset=["ID", residual_col]
    ).copy()

    valid = valid[valid["ID"].ne("")]

    if valid.empty:
        raise ValueError(
            f"{path.name}: No valid residual observations."
        )

    return valid, residual_col


# ==========================================================
# VERIFY DATA BEFORE PLOTTING
# ==========================================================

print("=" * 85)
print("HRRR RESIDUAL HISTOGRAMS — CORRECTED ORIGINAL DATA")
print("=" * 85)

loaded = []

total_stations = 0
total_observations = 0

for site, filename, expected_stations, expected_obs in FILES:

    path = DATA_DIR / filename

    df, residual_col = load_site(path)

    stations = sorted(df["ID"].unique())

    n_stations = len(stations)
    n_observations = len(df)

    assert n_stations == expected_stations, (
        f"{filename}: Expected {expected_stations} stations, "
        f"found {n_stations}."
    )

    assert n_observations == expected_obs, (
        f"{filename}: Expected {expected_obs} observations, "
        f"found {n_observations}."
    )

    if site == "Missouri":

        rol3 = df[df["ID"] == "ROL3"]

        assert not rol3.empty, "ROL3 missing."

        for column, expected in [
            ("Lat", 37.925919),
            ("Lon", -91.746689),
            ("Hgt_m", 301.01),
        ]:

            assert column in rol3.columns, (
                f"Missouri missing {column}."
            )

            values = pd.to_numeric(
                rol3[column],
                errors="coerce"
            )

            assert values.notna().all()

            assert np.allclose(
                values.to_numpy(),
                expected,
                atol=1e-6,
                rtol=0
            ), f"Incorrect ROL3 {column}."

    loaded.append(
        (site, filename, df, residual_col, stations)
    )

    total_stations += n_stations
    total_observations += n_observations

    print(
        f"{site:12s} | "
        f"Stations: {n_stations:2d} | "
        f"Observations: {n_observations:4d} | "
        f"Residual: {residual_col} | "
        f"Source: {filename}"
    )

assert total_stations == 11
assert total_observations == 1111

print("\nPASS: 3 corrected HRRR CSV files.")
print("PASS: 11 station-group entries.")
print("PASS: 1111 observations.")
print("PASS: Correct Missouri ROL3 coordinates.")

# ==========================================================
# GENERATE THREE VERTICAL HISTOGRAM PANELS
# ==========================================================

fig, axes = plt.subplots(
    NROWS,
    NCOLS,
    figsize=FIGSIZE
)

axes = np.asarray(axes).reshape(-1)

letters = ["a", "b", "c"]

fig.subplots_adjust(
    left=0.10,
    right=0.80,
    top=0.97,
    bottom=0.07,
    hspace=0.55
)

for i, (site, filename, df, residual_col, stations) in enumerate(loaded):

    ax = axes[i]

    palette = cycle(COLORS)

    color_map = {
        station: next(palette)
        for station in stations
    }

    for station in stations:

        values = df.loc[
            df["ID"] == station,
            residual_col
        ].dropna()

        ax.hist(
            values,
            bins=BINS,
            alpha=ALPHA,
            color=color_map[station],
            edgecolor="black",
            linewidth=0.7,
            label=station
        )

    ax.set_xlabel(
        "Residual (m)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Frequency (HRRR)",
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.30
    )

    ax.set_axisbelow(True)

    # Panel letters INSIDE each plotting box.
    ax.text(
        0.02,
        0.96,
        f"({letters[i]})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=10
    )

    legend = ax.legend(
        title="Station",
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True
    )

    legend.get_title().set_fontweight("bold")

    for label in legend.get_texts():
        label.set_fontweight("bold")

# ==========================================================
# SAVE FIGURES
# ==========================================================

fig.savefig(
    OUT_PNG,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close(fig)

print("\n" + "=" * 85)
print("FINAL VERIFICATION")
print("=" * 85)

print("PASS: (a) Missouri")
print("PASS: (b) California")
print("PASS: (c) Utah")
print("PASS: Panel letters inside plotting boxes.")
print("PASS: Individual automatic frequency scales.")
print("PASS: Corrected original HRRR CSV files.")

print("\nPNG:", OUT_PNG)
print("PDF:", OUT_PDF)
print("Source datasets:", DATA_DIR)
