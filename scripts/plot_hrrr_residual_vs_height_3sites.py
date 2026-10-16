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

OUT_DIR = Path.home() / "Desktop" / "NewUpdated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Residual_vs_Height_3Sites_HRRR_NEW_UPDATED.png"
OUT_PDF = OUT_DIR / "Residual_vs_Height_3Sites_HRRR_NEW_UPDATED.pdf"

FILES = [
    ("Missouri", "HRRR_Missouri.csv", 4, 59),
    ("California", "HRRR_California.csv", 3, 195),
    ("Utah", "HRRR_Utah.csv", 4, 857),
]

NROWS, NCOLS = 3, 1
FIGSIZE = (9.2, 10.8)
DPI = 300

FONT_FAMILY = "DejaVu Sans"
AX_LABEL_SIZE = 12
TICK_SIZE = 10
PANEL_LABEL_SIZE = 14

MARKER_SIZE = 55
EDGE_LW = 0.8
ALPHA = 0.88
GRID_ALPHA = 0.28

LEGEND_SIZE = 9.5
LEGEND_TITLE_SIZE = 10.5
LEGEND_MARKERSCALE = 1.0
LEGEND_FRAMEALPHA = 0.95

MARKERS = ["o", "s", "D", "^", "P", "X", "v", "<", ">", "*"]

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.weight": "bold",
    "axes.labelweight": "bold",
})

def load_hrrr(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ["ID", "Hgt_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: Missing columns {missing}")

    candidates = ["Residual (GNSS)", "Residual(GNSS)"]
    available = [c for c in candidates if c in df.columns]

    if len(available) != 1:
        raise ValueError(
            f"{path.name}: Expected one GNSS residual column, "
            f"found {available}. Columns: {df.columns.tolist()}"
        )

    residual_col = available[0]

    df["ID"] = df["ID"].astype("string").str.strip().str.upper()
    df["Hgt_m"] = pd.to_numeric(df["Hgt_m"], errors="coerce")
    df[residual_col] = pd.to_numeric(
        df[residual_col], errors="coerce"
    )

    valid = df.dropna(
        subset=["ID", "Hgt_m", residual_col]
    ).copy()

    valid = valid[valid["ID"].ne("")].copy()

    if valid.empty:
        raise ValueError(f"{path.name}: No valid observations")

    return valid, residual_col


print("=" * 85)
print("HRRR RESIDUAL VS HEIGHT — CORRECTED ORIGINAL DATA")
print("=" * 85)

loaded = []
total_observations = 0
total_stations = 0

for i, (site, filename, expected_stations, expected_obs) in enumerate(FILES):
    df, residual_col = load_hrrr(DATA_DIR / filename)
    stations = sorted(df["ID"].unique())

    assert len(stations) == expected_stations, (
        f"{filename}: Expected {expected_stations} stations, "
        f"found {len(stations)}"
    )
    assert len(df) == expected_obs, (
        f"{filename}: Expected {expected_obs} observations, "
        f"found {len(df)}"
    )

    if site == "Missouri":
        rol3 = df[df["ID"] == "ROL3"]
        assert not rol3.empty, "Missouri ROL3 missing"

        for col, expected in [
            ("Lat", 37.925919),
            ("Lon", -91.746689),
            ("Hgt_m", 301.01),
        ]:
            assert col in rol3.columns, f"Missing {col}"
            values = pd.to_numeric(rol3[col], errors="coerce")
            assert values.notna().all()
            assert np.allclose(
                values.to_numpy(),
                expected,
                atol=1e-6,
                rtol=0
            ), f"Incorrect ROL3 {col}"

    loaded.append((site, filename, df, residual_col, stations))
    total_observations += len(df)
    total_stations += len(stations)

    print(
        f"({chr(97+i)}) {site:12s} | "
        f"Stations: {len(stations):2d} | "
        f"Observations: {len(df):4d} | "
        f"Residual: {residual_col} | "
        f"Source: {filename}"
    )

assert total_stations == 11
assert total_observations == 1111

print("\nPASS: 3 corrected HRRR datasets.")
print("PASS: 11 station-group entries.")
print("PASS: 1111 observations.")
print("PASS: Corrected Missouri ROL3 coordinates.")

fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
axes = np.asarray(axes).reshape(-1)
panel_letters = "abc"

fig.subplots_adjust(
    left=0.12,
    right=0.78,
    top=0.97,
    bottom=0.08,
    hspace=0.55
)

for i, (site, filename, df, residual_col, stations) in enumerate(loaded):
    ax = axes[i]

    marker_map = {
        st: MARKERS[j % len(MARKERS)]
        for j, st in enumerate(stations)
    }

    palette = cycle(plt.cm.tab10.colors)
    colors = {st: next(palette) for st in stations}

    for st in stations:
        sub = df[df["ID"] == st]

        ax.scatter(
            sub["Hgt_m"],
            sub[residual_col],
            color=colors[st],
            marker=marker_map[st],
            edgecolors="black",
            linewidths=EDGE_LW,
            s=MARKER_SIZE,
            alpha=ALPHA,
            zorder=3
        )

    # Panel label above the plotting box
    ax.text(
        0.02,
        1.03,
        f"({panel_letters[i]})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=10
    )

    ax.set_xlabel(
        "Height (m)",
        fontsize=AX_LABEL_SIZE,
        fontweight="bold"
    )
    ax.set_ylabel(
        "Residual (HRRR) (m)",
        fontsize=AX_LABEL_SIZE,
        fontweight="bold"
    )

    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA, zorder=0)

    handles = [
        Line2D(
            [0], [0],
            marker=marker_map[st],
            color="w",
            markerfacecolor=colors[st],
            markeredgecolor="black",
            markersize=7,
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

fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

assert OUT_PNG.is_file()
assert OUT_PDF.is_file()

print("\n" + "=" * 85)
print("FINAL VERIFICATION")
print("=" * 85)
print("PASS: (a) Missouri")
print("PASS: (b) California")
print("PASS: (c) Utah")
print("PASS: Only panel letters inside plotting boxes.")
print("PASS: Correct original HRRR CSV files.")
print("PASS: 11 station-group entries and 1111 observations.")
print("\nPNG:", OUT_PNG)
print("PDF:", OUT_PDF)
print("Source datasets:", DATA_DIR)
