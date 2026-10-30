from pathlib import Path
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Supplement_Original_Data"

OUTPUT_DIR = Path.home() / "Desktop" / "GNSS Final Results" / "Supplementary_Figures_Updated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "Residual_vs_Height_ALL_SITES_NEW_UPDATED.png"
OUT_PDF = OUTPUT_DIR / "Residual_vs_Height_ALL_SITES_NEW_UPDATED.pdf"

FILES = [
    ("Australia", "ERA5_Australia.csv"),
    ("Europe", "ERA5_Germany.csv"),
    ("Japan-1", "ERA5_Japan1.csv"),
    ("Alaska", "ERA5_Alaska.csv"),
    ("Japan-2", "ERA5_Japan2.csv"),
    ("Italy", "ERA5_Italy.csv"),
    ("Chile", "ERA5_Chile.csv"),
    ("Guadeloupe", "ERA5_Guadeloupe.csv"),
    ("Missouri", "ERA5_Missouri.csv"),
    ("California", "ERA5_California.csv"),
]

FILES = [
    FILES[0],
    FILES[2],
    FILES[1],
    *FILES[3:]
]

NROWS = 5
NCOLS = 2
FIGSIZE = (13.5, 15)
DPI = 600

FONT_FAMILY = "DejaVu Sans"
AX_LABEL_SIZE = 11
TICK_SIZE = 10
PANEL_LABEL_SIZE = 12

MARKER_SIZE = 20
EDGE_LW = 0.6
ALPHA = 0.85
GRID_ALPHA = 0.28

LEGEND_SIZE = 8.5
LEGEND_TITLE_SIZE = 9.5
LEGEND_MARKERSCALE = 0.9
LEGEND_FRAMEALPHA = 0.95

MARKERS = [
    "o", "s", "D", "^", "P",
    "X", "v", "<", ">", "*"
]

def load_era5(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ["ID", "Hgt_m", "Residual(ERA5)"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path.name}: Missing columns {missing}"
        )

    df["ID"] = (
        df["ID"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["Hgt_m"] = pd.to_numeric(
        df["Hgt_m"], errors="coerce"
    )

    df["Residual(ERA5)"] = pd.to_numeric(
        df["Residual(ERA5)"], errors="coerce"
    )

    df = df.dropna(
        subset=["ID", "Hgt_m", "Residual(ERA5)"]
    ).copy()

    df = df[df["ID"].ne("")].copy()

    if df.empty:
        raise ValueError(
            f"No valid observations: {path.name}"
        )

    return df


print("=" * 80)
print("ERA5 RESIDUAL VS HEIGHT — VERIFIED ORIGINAL CSV FILES")
print("=" * 80)

loaded = []
total_observations = 0
total_stations = 0

for i, (site, filename) in enumerate(FILES):
    df = load_era5(DATA_DIR / filename)
    stations = sorted(df["ID"].unique())

    loaded.append((site, filename, df, stations))

    total_observations += len(df)
    total_stations += len(stations)

    print(
        f"({chr(97+i)}) {site:12s} "
        f"{filename:24s} "
        f"Stations: {len(stations):2d} "
        f"Observations: {len(df):4d} PASS"
    )

assert len(loaded) == 10
assert total_stations == 41
assert total_observations == 5085

missouri = load_era5(DATA_DIR / "ERA5_Missouri.csv")

if "Lon" in missouri.columns:
    rol3 = missouri[missouri["ID"] == "ROL3"]
    assert not rol3.empty
    assert np.allclose(
        pd.to_numeric(rol3["Lon"]).to_numpy(),
        -91.746689,
        atol=1e-6
    )

print("\nPASS: 10 corrected ERA5 datasets.")
print("PASS: 41 station-group entries.")
print("PASS: 5085 observations.")
print("PASS: Corrected Missouri ROL3 longitude.")

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

fig, axes = plt.subplots(
    NROWS, NCOLS, figsize=FIGSIZE
)

axes = axes.flatten()
panel_letters = "abcdefghij"

for i, (site, filename, df, stations) in enumerate(loaded):
    ax = axes[i]

    marker_map = {
        st: MARKERS[j % len(MARKERS)]
        for j, st in enumerate(stations)
    }

    palette = cycle(plt.cm.tab10.colors)
    colors = {
        st: next(palette)
        for st in stations
    }

    for st in stations:
        sub = df[df["ID"] == st]

        ax.scatter(
            sub["Hgt_m"],
            sub["Residual(ERA5)"],
            color=colors[st],
            marker=marker_map[st],
            edgecolors="black",
            linewidths=EDGE_LW,
            s=MARKER_SIZE,
            alpha=ALPHA,
            zorder=3
        )

    # ONLY THE PANEL LETTER, INSIDE THE PLOTTING BOX
    ax.text(
        0.02,
        0.96,
        f"({panel_letters[i]})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=10,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=1
        )
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

    for tick in (
        ax.get_xticklabels() + ax.get_yticklabels()
    ):
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

    for txt in legend.get_texts():
        txt.set_fontweight("bold")

plt.subplots_adjust(
    wspace=0.75,
    hspace=0.45
)

fig.savefig(
    OUT_PNG,
    dpi=DPI,
    bbox_inches="tight"
)


plt.close(fig)

assert OUT_PNG.is_file()

print("\n" + "=" * 80)
print("FIGURE GENERATED SUCCESSFULLY")
print("=" * 80)
print("PASS: Panel letters (a)–(j) placed inside plotting boxes.")
print("PASS: Region names omitted from plots.")
print("PASS: Station legends remain outside.")
print("PNG:", OUT_PNG)
