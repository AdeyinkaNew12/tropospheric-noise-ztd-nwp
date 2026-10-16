from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "Supplement_Original_Data"

OUTPUT_DIR = Path.home() / "Desktop" / "NewUpdated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "Residual_vs_Height_ALL_SITES_NEW_UPDATED.png"
OUT_PDF = OUTPUT_DIR / "Residual_vs_Height_ALL_SITES_NEW_UPDATED.pdf"

FILES = [
    ("Japan-1", "ERA5_Japan1.csv"),
    ("Missouri", "ERA5_Missouri.csv"),
    ("Alaska", "ERA5_Alaska.csv"),
    ("Australia", "ERA5_Australia.csv"),
    ("Germany", "ERA5_Germany.csv"),
    ("Japan-2", "ERA5_Japan2.csv"),
    ("Italy", "ERA5_Italy.csv"),
    ("Chile", "ERA5_Chile.csv"),
    ("Guadeloupe", "ERA5_Guadeloupe.csv"),
    ("California", "ERA5_California.csv"),
]

NROWS, NCOLS = 5, 2
FIGSIZE = (13.5, 15)

FONT_FAMILY = "DejaVu Sans"
AX_LABEL_SIZE = 11
TICK_SIZE = 10
PANEL_LABEL_SIZE = 13

MARKER_SIZE = 20
EDGE_LW = 0.6
ALPHA = 0.85
GRID_ALPHA = 0.28

LEGEND_SIZE = 8.5
LEGEND_TITLE_SIZE = 9.5
LEGEND_MARKERSCALE = 0.9
LEGEND_FRAMEALPHA = 0.95

MARKERS = ["o", "s", "D", "^", "P", "X", "v", "<", ">", "*"]

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


def load_height_residual(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    id_col = find_col(df, [
        "ID", "Station", "StationID", "Name"
    ])

    resid_col = find_col(df, [
        "Residual(ERA5)",
        "Residual (ERA5)",
        "Residual(ERA-5)"
    ])

    hgt_col = find_col(df, [
        "Hgt_m", "Height", "Elevation", "Elev"
    ])

    if id_col is None:
        raise KeyError(f"Station column missing: {path.name}")

    if resid_col is None:
        raise KeyError(f"ERA5 residual column missing: {path.name}")

    if hgt_col is None:
        raise KeyError(f"Elevation column missing: {path.name}")

    df[id_col] = (
        df[id_col]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df[resid_col] = pd.to_numeric(
        df[resid_col], errors="coerce"
    )

    df[hgt_col] = pd.to_numeric(
        df[hgt_col], errors="coerce"
    )

    df = df.dropna(
        subset=[id_col, resid_col, hgt_col]
    ).copy()

    df = df[df[id_col].ne("")]

    return df, id_col, hgt_col, resid_col


# ============================================================
# VALIDATE DATASETS
# ============================================================

for site, filename in FILES:
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing source dataset for {site}: {path}"
        )

print("=" * 75)
print("RESIDUAL VS HEIGHT — UPDATED ORIGINAL ERA5 DATA")
print("=" * 75)

# ============================================================
# BUILD FIGURE
# ============================================================

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.weight": "bold"
})

fig, axes = plt.subplots(
    NROWS, NCOLS, figsize=FIGSIZE
)

axes = axes.flatten()

panel_letters = list("abcdefghijklmnopqrstuvwxyz")

total_observations = 0

for i, (ax, (site, filename)) in enumerate(
    zip(axes, FILES)
):
    path = DATA_DIR / filename

    df, id_col, hgt_col, resid_col = load_height_residual(path)

    stations = sorted(df[id_col].unique())

    marker_map = {
        st: MARKERS[j % len(MARKERS)]
        for j, st in enumerate(stations)
    }

    palette = cycle(plt.cm.tab10.colors)

    colors = {
        st: next(palette)
        for st in stations
    }

    # Plot every valid observation.
    for st in stations:
        sub = df[df[id_col] == st]

        ax.scatter(
            sub[hgt_col],
            sub[resid_col],
            color=colors[st],
            marker=marker_map[st],
            edgecolors="black",
            linewidth=EDGE_LW,
            s=MARKER_SIZE,
            alpha=ALPHA,
            zorder=3
        )

    ax.set_title(
        f"({panel_letters[i]}) {site}",
        loc="left",
        pad=2,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Height (m)",
        fontsize=AX_LABEL_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Residual (ERA5) (m)",
        fontsize=AX_LABEL_SIZE,
        fontweight="bold"
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=GRID_ALPHA,
        zorder=0
    )

    handles = [
        Line2D(
            [0], [0],
            marker=marker_map[st],
            color="w",
            markerfacecolor=colors[st],
            markeredgecolor="black",
            markersize=6,
            label=st
        )
        for st in stations
    ]

    legend = ax.legend(
        handles=handles,
        title="Station",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        frameon=True,
        framealpha=LEGEND_FRAMEALPHA,
        markerscale=LEGEND_MARKERSCALE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0
    )

    legend.get_title().set_fontweight("bold")

    for text in legend.get_texts():
        text.set_fontweight("bold")

    total_observations += len(df)

    print(
        f"{site:12s} | "
        f"Stations: {len(stations):2d} | "
        f"Observations: {len(df):5d} | "
        f"Source: {filename}"
    )

# ============================================================
# SAVE FIGURE
# ============================================================

plt.subplots_adjust(
    wspace=0.65,
    hspace=0.40
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close(fig)

print("\n" + "=" * 75)
print("FIGURE GENERATED SUCCESSFULLY")
print("=" * 75)

print("Total observations:", total_observations)
print("PNG:", OUT_PNG)
print("PDF:", OUT_PDF)
print("Source datasets:", DATA_DIR)
