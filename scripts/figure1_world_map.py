
from pathlib import Path
import math

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Supplement_Original_Data" / "GNSS_Final_Station_Coordinates.csv"
OUT = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Missing input CSV: {DATA}")

df = pd.read_csv(DATA)

required = {"Region", "Station", "Latitude", "Longitude"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing CSV columns: {sorted(missing)}")

df["Region"] = df["Region"].astype(str).str.strip()
df["Station"] = df["Station"].astype(str).str.strip()

df["Latitude"] = pd.to_numeric(df["Latitude"], errors="raise")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="raise")

if df[["Latitude", "Longitude"]].isna().any().any():
    raise ValueError("Missing station coordinates.")

if not df["Latitude"].between(-90, 90).all():
    raise ValueError("Invalid latitude detected.")

if not df["Longitude"].between(-180, 180).all():
    raise ValueError("Invalid longitude detected.")

# Verify that ERA5/HRRR duplicates have identical coordinates.
coordinate_counts = (
    df.groupby(["Region", "Station"])[["Latitude", "Longitude"]]
    .nunique()
)

if (coordinate_counts > 1).any().any():
    raise ValueError("Conflicting coordinates found for repeated stations.")

# One plotting record per station within each study region.
df = df.drop_duplicates(["Region", "Station"]).copy()

if len(df) != 45 or df["Region"].nunique() != 11:
    raise ValueError(
        f"Expected 45 regional station entries across 11 regions. "
        f"Found {len(df)} entries across {df['Region'].nunique()} regions."
    )

# Colors and label positions follow the original flat-world-map reference.
# Callout positions use axes-fraction coordinates.
CONFIG = {
    "Alaska": {
        "color": "#1677b9",
        "box": (0.055, 0.89)
    },
    "Utah": {
        "color": "#00976d",
        "box": (0.205, 0.83)
    },
    "California": {
        "color": "#e66a00",
        "box": (0.065, 0.57)
    },
    "Missouri": {
        "color": "#8736a4",
        "box": (0.325, 0.75)
    },
    "Germany": {
        "color": "#b18a00",
        "box": (0.485, 0.86)
    },
    "Italy": {
        "color": "#956057",
        "box": (0.605, 0.61)
    },
    "Guadeloupe": {
        "color": "#dc168d",
        "box": (0.435, 0.46)
    },
    "Chile": {
        "color": "#0092a9",
        "box": (0.18, 0.29)
    },
    "Japan1": {
        "color": "#f3a000",
        "box": (0.815, 0.49)
    },
    "Japan2": {
        "color": "#253c98",
        "box": (0.92, 0.76)
    },
    "Australia": {
        "color": "#444444",
        "box": (0.82, 0.25)
    },
}

if set(df["Region"]) != set(CONFIG):
    raise ValueError(
        f"Unexpected study regions: {set(df['Region']) ^ set(CONFIG)}"
    )

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
})

fig = plt.figure(figsize=(17, 10), facecolor="white")

# PlateCarree produces the flat geographic layout of the reference.
ax = fig.add_axes(
    [0.025, 0.16, 0.95, 0.78],
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [-180, 180, -60, 85],
    crs=ccrs.PlateCarree()
)

ax.add_feature(
    cfeature.OCEAN,
    facecolor="white",
    zorder=0
)

ax.add_feature(
    cfeature.LAND,
    facecolor="#f4f4ef",
    edgecolor="none",
    zorder=1
)

ax.add_feature(
    cfeature.BORDERS,
    edgecolor="#b7b7b7",
    linewidth=0.4,
    zorder=2
)

ax.coastlines(
    resolution="110m",
    linewidth=0.6,
    color="#999999",
    zorder=3
)

ax.spines["geo"].set_visible(False)

# Draw the 11 regional markers and their callout boxes.
for region, cfg in CONFIG.items():

    group = df[df["Region"] == region].copy()

    lon = group["Longitude"].mean()
    lat = group["Latitude"].mean()

    color = cfg["color"]
    box_x, box_y = cfg["box"]

    ax.scatter(
        lon,
        lat,
        s=55,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=7
    )

    station_names = sorted(group["Station"].tolist())

    # Two-column station arrangement.
    rows = [
        "  ".join(station_names[i:i+2])
        for i in range(0, len(station_names), 2)
    ]

    display_region = (
        region.replace("Japan1", "Japan-1")
              .replace("Japan2", "Japan-2")
    )

    label = display_region + "\n" + "\n".join(rows)

    ax.annotate(
        label,
        xy=(lon, lat),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        xytext=(box_x, box_y),
        textcoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        linespacing=1.15,
        color="#202020",
        bbox=dict(
            boxstyle="round,pad=0.32,rounding_size=0.12",
            facecolor="white",
            edgecolor=color,
            linewidth=1.6
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            linewidth=1.6,
            shrinkA=4,
            shrinkB=2,
            connectionstyle="arc3,rad=0"
        ),
        annotation_clip=False,
        zorder=8
    )

handles = []

for region, cfg in CONFIG.items():
    display_region = (
        region.replace("Japan1", "Japan-1")
              .replace("Japan2", "Japan-2")
    )

    handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=cfg["color"],
            markeredgecolor="white",
            markersize=8,
            label=display_region
        )
    )

handles.append(
    Patch(
        facecolor="#f4f4ef",
        edgecolor="#777777",
        linewidth=0.9,
        label="World Map"
    )
)

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.035),
    ncol=6,
    frameon=False,
    fontsize=10,
    columnspacing=1.7,
    handletextpad=0.5
)

stem = "Figure1_GNSS_World_Map"

for extension in ["pdf"]:
    output = OUT / f"{stem}.{extension}"

    fig.savefig(
        output,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white"
    )


    print("SAVED:", output)

plt.close(fig)

print("\n" + "=" * 65)
print("FIGURE 1 VERIFICATION")
print("=" * 65)

print("Original CSV records: 52")
print("Regional station entries:", len(df))
print("Unique GNSS station IDs:", df["Station"].nunique())
print("Study regions:", df["Region"].nunique())

print("\nStations per region:")
print(
    df.groupby("Region")["Station"]
    .nunique()
    .to_string()
)

print("\nFIGURE 1 GENERATION COMPLETE")
