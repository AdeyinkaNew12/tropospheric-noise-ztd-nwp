from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import cycle

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Supplement_Original_Data"

OUT_DIR = Path.home() / "Desktop" / "GNSS Final Results" / "Supplementary_Figures_Updated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    ("Australia", "ERA5_Australia.csv"),
    ("Japan-1", "ERA5_Japan1.csv"),
    ("Germany", "ERA5_Germany.csv"),
    ("Alaska", "ERA5_Alaska.csv"),
    ("Japan-2", "ERA5_Japan2.csv"),
    ("Italy", "ERA5_Italy.csv"),
    ("Chile", "ERA5_Chile.csv"),
    ("Guadeloupe", "ERA5_Guadeloupe.csv"),
    ("Rolla", "ERA5_Rolla.csv"),
    ("California", "ERA5_California.csv"),
]

NROWS, NCOLS = 5, 2
FIGSIZE = (14, 18)
DPI = 600

OUT_PNG = OUT_DIR / "Residual_Histograms_ALL_SITES_ERA5_NEW_UPDATED.png"
OUT_PDF = OUT_DIR / "Residual_Histograms_ALL_SITES_ERA5_NEW_UPDATED.pdf"

# ============================================================
# STYLING
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

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

# ============================================================
# HELPERS
# ============================================================

def norm(s):
    return re.sub(r"[()\s_\-]", "", str(s).lower())


def find_col(df, candidates):
    norm_map = {norm(c): c for c in df.columns}

    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]

    return None


def load_site(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing original dataset: {path}"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    id_col = find_col(
        df,
        ["ID", "Station", "StationID", "Name"]
    )

    resid_col = "Residual(ERA5)" if "Residual(ERA5)" in df.columns else "Residual"

    if resid_col not in df.columns:
        raise ValueError(
            f"Missing required residual column "
            f"'{resid_col}' in {path.name}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if id_col is None:
        raise ValueError(
            f"Missing station ID column in {path.name}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[id_col] = (
        df[id_col]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df[resid_col] = pd.to_numeric(
        df[resid_col],
        errors="coerce"
    )

    dfc = df.dropna(
        subset=[id_col, resid_col]
    ).copy()

    dfc = dfc[
        dfc[id_col].ne("")
    ].copy()

    return dfc, id_col, resid_col


# ============================================================
# VALIDATE ALL DATASETS BEFORE PLOTTING
# ============================================================

print("=" * 75)
print("ERA5 RESIDUAL HISTOGRAMS — CORRECTED ORIGINAL DATA")
print("=" * 75)

loaded = []
total_observations = 0
total_stations = 0

for site, filename in FILES:
    path = DATA_DIR / filename

    dfc, id_col, resid_col = load_site(path)

    stations = sorted(
        dfc[id_col].dropna().unique()
    )

    if len(stations) == 0 or len(dfc) == 0:
        raise ValueError(
            f"No valid observations for {site}"
        )

    loaded.append(
        (site, filename, dfc, id_col, resid_col, stations)
    )

    total_observations += len(dfc)
    total_stations += len(stations)

    print(
        f"{site:12s} | "
        f"Stations: {len(stations):2d} | "
        f"Observations: {len(dfc):5d} | "
        f"Source: {filename}"
    )

if len(loaded) != 10:
    raise ValueError(
        f"Expected 10 datasets, found {len(loaded)}"
    )

if total_stations != 41:
    raise ValueError(
        f"Expected 41 station-group entries, "
        f"found {total_stations}"
    )

if total_observations != 4989:
    raise ValueError(
        f"Expected 4989 observations, "
        f"found {total_observations}"
    )

print("\nPASS: 10 ERA5 datasets verified.")
print("PASS: 41 station-group entries verified.")
print("PASS: 4989 observations verified.")

# ============================================================
# BUILD COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(
    NROWS,
    NCOLS,
    figsize=FIGSIZE
)

axes = axes.flatten()
letters = list("abcdefghijklmnopqrstuvwxyz")

for i, (
    site, filename, dfc, id_col, resid_col, stations
) in enumerate(loaded):

    ax = axes[i]

    color_cycle = cycle(COLORS)

    color_map = {
        st: next(color_cycle)
        for st in stations
    }

    # Use common bin edges within each panel so that
    # station histograms are directly comparable.
    all_residuals = dfc[resid_col].to_numpy()

    bin_edges = np.histogram_bin_edges(
        all_residuals,
        bins=BINS
    )

    # Plot one histogram per station.
    for st in stations:
        data = dfc.loc[
            dfc[id_col] == st,
            resid_col
        ].dropna()

        if data.empty:
            continue

        ax.hist(
            data,
            bins=bin_edges,
            alpha=ALPHA,
            color=color_map[st],
            edgecolor="black",
            linewidth=0.6,
            label=st
        )

    ax.set_xlabel(
        "Residual (m)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Frequency (ERA5)",
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
        zorder=0
    )

    ax.set_axisbelow(True)

    # Panel letter INSIDE the upper-left corner of each subplot
    ax.text(
        0.02, 0.96,
        f"({letters[i]})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=10
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    for tick in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        tick.set_fontweight("bold")

    # Legend outside the panel.
    leg = ax.legend(
        title="Station",
        fontsize=9,
        title_fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True
    )

    leg.get_title().set_fontweight("bold")

    for txt in leg.get_texts():
        txt.set_fontweight("bold")

# ============================================================
# SPACING AND OUTPUT
# ============================================================

plt.subplots_adjust(
    left=0.07,
    right=0.93,
    top=0.97,
    bottom=0.05,
    wspace=0.75,
    hspace=0.42
)

fig.savefig(
    OUT_PNG,
    dpi=DPI,
    bbox_inches="tight"
)


plt.close(fig)

# ============================================================
# FINAL VERIFICATION
# ============================================================

assert OUT_PNG.is_file()

print("\n" + "=" * 75)
print("ERA5 HISTOGRAM FIGURE GENERATED SUCCESSFULLY")
print("=" * 75)

print("Datasets:", len(loaded))
print("Station-group entries:", total_stations)
print("Valid observations:", total_observations)

print("\nPNG:", OUT_PNG)
print("Original data directory:", DATA_DIR)
