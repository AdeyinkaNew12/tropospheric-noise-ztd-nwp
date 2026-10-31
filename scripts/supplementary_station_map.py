#!/usr/bin/env python3

"""
Supplementary GNSS station map.

Features:
- 11 study regions
- Colored elevation maps with hillshading
- Corrected California station coordinates
- Latitude and longitude axes
- Station labels and counts
- Geographic scale bars
- Elevation relief (Delta z)

Terrain source:
AWS Terrain Tiles, Terrarium elevation encoding.

Requires an internet connection.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource

from pyproj import Transformer, Geod
import contextily as ctx


# ==================================================
# SETTINGS
# ==================================================

REGIONS = [
    ("Alaska", "Alaska"),
    ("Australia", "Australia"),
    ("California", "California"),
    ("Chile", "Chile"),
    ("Germany", "Europe"),
    ("Guadeloupe", "Guadeloupe"),
    ("Italy", "Italy"),
    ("Japan-1", "Japan"),
    ("Japan-2", "Japan2"),
    ("Rolla", "Rolla"),
    ("Utah", "Utah"),
]

CALIFORNIA = [
    ("P665", 40.456, -121.526),
    ("P666", 40.465, -121.515),
    ("P668", 40.509, -121.463),
]

TERRARIUM = (
    "https://s3.amazonaws.com/"
    "elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)

TO_M = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:3857",
    always_xy=True,
)

TO_LL = Transformer.from_crs(
    "EPSG:3857",
    "EPSG:4326",
    always_xy=True,
)

GEOD = Geod(ellps="WGS84")

DPI = 300


# ==================================================
# LOAD STATION DATA
# ==================================================

def load_stations(path):
    data_dir = Path(__file__).resolve().parents[1] / "data" / "Supplement_Original_Data"

    mapping = {
        "Alaska": "Alaska",
        "Australia": "Australia",
        "California": "California",
        "Chile": "Chile",
        "Germany": "Europe",
        "Guadeloupe": "Guadeloupe",
        "Italy": "Italy",
        "Japan1": "Japan",
        "Japan2": "Japan2",
        "Rolla": "Rolla",
        "Utah": "Utah",
    }

    records = []

    for region, site in mapping.items():
        available = [
            data_dir / f"ERA5_{region}.csv",
            data_dir / f"HRRR_{region}.csv",
        ]

        selected = [p for p in available if p.is_file()]

        if not selected:
            raise FileNotFoundError(f"No original CSV for {region}")

        for csv_path in selected:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            required = {"ID", "Lat", "Lon", "Hgt_m"}
            missing = required - set(df.columns)

            if missing:
                raise ValueError(f"{csv_path.name}: Missing {missing}")

            df = df.rename(columns={
                "ID": "Station",
                "Lat": "Latitude",
                "Lon": "Longitude",
                "Hgt_m": "Elevation_m",
            })

            df["Station"] = df["Station"].astype("string").str.strip().str.upper()

            for col in ["Latitude", "Longitude", "Elevation_m"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(
                subset=["Station", "Latitude", "Longitude", "Elevation_m"]
            )
            df = df[df["Station"].ne("")]

            for station, group in df.groupby("Station"):
                coords = group[
                    ["Latitude", "Longitude", "Elevation_m"]
                ]

                if (
                    (coords.max() - coords.min()).abs()
                    > 0.000001
                ).any():
                    raise ValueError(
                        f"{csv_path.name}: Inconsistent coordinates "
                        f"or elevations for {station}"
                    )

                row = coords.iloc[0]

                records.append({
                    "Station": station,
                    "Latitude": float(row["Latitude"]),
                    "Longitude": float(row["Longitude"]),
                    "Elevation_m": float(row["Elevation_m"]),
                    "Site": site,
                    "Source": csv_path.name,
                })

    result = pd.DataFrame(records)

    for (site, station), group in result.groupby(["Site", "Station"]):
        coords = group[["Latitude", "Longitude", "Elevation_m"]]

        if ((coords.max() - coords.min()).abs() > 0.000001).any():
            raise ValueError(
                f"ERA5/HRRR mismatch for {site} station {station}"
            )

    result = result.drop_duplicates(
        subset=["Site", "Station"]
    ).reset_index(drop=True)

    rol3 = result[
        (result["Site"] == "Rolla")
        & (result["Station"] == "ROL3")
    ]

    assert len(rol3) == 1
    assert abs(rol3.iloc[0]["Latitude"] - 37.925919) < 1e-6
    assert abs(rol3.iloc[0]["Longitude"] - (-91.746689)) < 1e-6
    assert abs(rol3.iloc[0]["Elevation_m"] - 301.01) < 1e-6

    print("PASS: Corrected original ERA5/HRRR station data.")
    print("PASS: Station coordinates and elevations verified.")
    print("PASS: Rolla ROL3 elevation = 301.01 m.")
    print(f"PASS: {len(result)} unique regional station entries.")

    return result


# ==================================================
# REGIONAL MAP EXTENT
# ==================================================

def get_extent(group):

    x, y = TO_M.transform(
        group["Longitude"].to_numpy(),
        group["Latitude"].to_numpy(),
    )

    width = max(
        np.ptp(x),
        7000.0,
    ) * 1.45

    height = max(
        np.ptp(y),
        7000.0,
    ) * 1.45

    side = max(width, height)

    cx = (min(x) + max(x)) / 2
    cy = (min(y) + max(y)) / 2

    return (
        cx - side / 2,
        cx + side / 2,
        cy - side / 2,
        cy + side / 2,
    )


# ==================================================
# DOWNLOAD AND PROCESS ELEVATION
# ==================================================

def get_terrain(extent):

    west, east, south, north = extent

    span = max(
        east - west,
        north - south,
    )

    if span < 14000:
        zoom = 12
    elif span < 30000:
        zoom = 11
    elif span < 60000:
        zoom = 10
    else:
        zoom = 9

    rgb, bounds = ctx.bounds2img(
        west,
        south,
        east,
        north,
        zoom=zoom,
        source=TERRARIUM,
        ll=False,
        wait=0,
        max_retries=2,
    )

    arr = np.asarray(
        rgb,
        dtype=np.float64,
    )

    # Decode Terrarium RGB elevation.
    elevation = (
        arr[:, :, 0] * 256
        + arr[:, :, 1]
        + arr[:, :, 2] / 256
        - 32768
    )

    dx = (
        bounds[1] - bounds[0]
    ) / elevation.shape[1]

    dy = (
        bounds[3] - bounds[2]
    ) / elevation.shape[0]

    light = LightSource(
        azdeg=315,
        altdeg=45,
    )

    colored_terrain = light.shade(
        elevation,
        cmap=plt.get_cmap("terrain"),
        vert_exag=1.3,
        dx=dx,
        dy=dy,
        blend_mode="overlay",
    )

    return (
        colored_terrain,
        elevation,
        bounds,
    )


# ==================================================
# SCALE BAR
# ==================================================

def add_scale_bar(ax, km=5):

    west, east = ax.get_xlim()
    south, north = ax.get_ylim()

    x0 = west + 0.60 * (east - west)
    y0 = south + 0.09 * (north - south)

    lon, lat = TO_LL.transform(
        x0,
        y0,
    )

    lon2, lat2, _ = GEOD.fwd(
        lon,
        lat,
        90,
        km * 1000,
    )

    x1, _ = TO_M.transform(
        lon2,
        lat2,
    )

    # Use a shorter bar when 5 km
    # cannot fit in the map panel.
    if x1 > west + 0.97 * (east - west):

        km = 3

        lon2, lat2, _ = GEOD.fwd(
            lon,
            lat,
            90,
            km * 1000,
        )

        x1, _ = TO_M.transform(
            lon2,
            lat2,
        )

    if x1 > west + 0.97 * (east - west):

        km = 1

        lon2, lat2, _ = GEOD.fwd(
            lon,
            lat,
            90,
            km * 1000,
        )

        x1, _ = TO_M.transform(
            lon2,
            lat2,
        )

    ax.plot(
        [x0, x1],
        [y0, y0],
        color="black",
        linewidth=2.5,
        zorder=30,
    )

    tick = 0.007 * (north - south)

    for x in [x0, x1]:

        ax.plot(
            [x, x],
            [y0 - tick, y0 + tick],
            color="black",
            linewidth=1.5,
            zorder=30,
        )

    ax.text(
        (x0 + x1) / 2,
        y0 + 0.015 * (north - south),
        f"{km} km",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        zorder=30,
    )


# ==================================================
# DRAW EACH STUDY REGION
# ==================================================

def draw_region(
    ax,
    group,
    title,
    letter,
):

    extent = get_extent(group)

    west, east, south, north = extent

    relief = None

    try:

        (
            colored_terrain,
            elevation,
            bounds,
        ) = get_terrain(extent)

        ax.imshow(
            colored_terrain,
            extent=bounds,
            origin="upper",
            interpolation="bilinear",
            zorder=0,
        )

        # Calculate elevation range
        # within the displayed region.
        h, w = elevation.shape

        x0 = max(
            0,
            int(
                (west - bounds[0])
                / (bounds[1] - bounds[0])
                * w
            ),
        )

        x1 = min(
            w,
            int(
                (east - bounds[0])
                / (bounds[1] - bounds[0])
                * w
            ),
        )

        y0 = max(
            0,
            int(
                (bounds[3] - north)
                / (bounds[3] - bounds[2])
                * h
            ),
        )

        y1 = min(
            h,
            int(
                (bounds[3] - south)
                / (bounds[3] - bounds[2])
                * h
            ),
        )

        crop = elevation[
            y0:y1,
            x0:x1,
        ]

        if crop.size:

            relief = int(
                np.round(
                    np.nanmax(crop)
                    - np.nanmin(crop)
                )
            )

    except Exception as exc:

        print(
            f"WARNING: Terrain unavailable "
            f"for {title}: {exc}"
        )

        ax.set_facecolor("#e4e9dc")

    # Station coordinates
    x, y = TO_M.transform(
        group["Longitude"].to_numpy(),
        group["Latitude"].to_numpy(),
    )

    # GNSS station markers
    ax.scatter(
        x,
        y,
        marker="^",
        s=38,
        facecolor="#f3d333",
        edgecolor="black",
        linewidth=0.7,
        zorder=10,
    )

    # Manual label offsets for closely spaced stations.
    # Offsets are in screen points, NOT geographic coordinates.
    LABEL_OFFSETS = {
        "Guadeloupe": {
            "SOUF": (10, 18),
            "F562": (18, 8),
            "CRA6": (-42, 8),
            "F8O2": (18, -10),
            "BUL9": (-42, -12),
        },
        "Rolla": {
            "ROL1": (8, 13),
            "ROL2": (8, -13),
        },
    }

    region_offsets = LABEL_OFFSETS.get(title, {})

    # Station labels
    for station, xx, yy in zip(
        group["Station"],
        x,
        y,
    ):

        offset = region_offsets.get(station, (3, 3))

        ax.annotate(
            station,
            (xx, yy),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] >= 0 else "right",
            va="center" if station in region_offsets else "bottom",
            arrowprops=(
                dict(
                    arrowstyle="-",
                    color="black",
                    linewidth=0.6,
                    shrinkA=1,
                    shrinkB=3,
                )
                if station in region_offsets
                else None
            ),
            fontsize=8,
            color="black",
            fontweight="bold",
            zorder=11,
            path_effects=[
                pe.withStroke(
                    linewidth=2.2,
                    foreground="white",
                )
            ],
        )

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # Latitude and longitude axes
    xticks = np.linspace(
        west,
        east,
        4,
    )

    yticks = np.linspace(
        south,
        north,
        4,
    )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(
        [
            f"{TO_LL.transform(xx, (south+north)/2)[0]:.2f}"
            for xx in xticks
        ],
        fontsize=7,
    )

    ax.set_yticklabels(
        [
            f"{TO_LL.transform((west+east)/2, yy)[1]:.2f}"
            for yy in yticks
        ],
        fontsize=7,
    )

    ax.set_xlabel(
        "Longitude",
        fontsize=7,
        labelpad=1,
    )

    ax.set_ylabel(
        "Latitude",
        fontsize=7,
        labelpad=1,
    )

    ax.tick_params(
        length=2,
        pad=1,
    )

    # Panel heading
    ax.set_title(
        f"({letter}) {title}",
        fontsize=10,
        fontweight="bold",
        pad=3,
    )

    # Station count and elevation relief
    details = f"n = {len(group)}"

    if relief is not None:

        details += (
            f"\nΔz = {relief:,} m"
        )

    ax.text(
        0.02,
        0.03,
        details,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
            pad=2,
        ),
        zorder=30,
    )

    add_scale_bar(
        ax,
        km=5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stations",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1] / "data/Supplement_Original_Data/StationMapfile_VERIFIED.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "figures/FigureS8_Supplementary_Topographic_Map.png"),
    )

    args = parser.parse_args()

    df = load_stations(args.stations)

    print("\nCalifornia stations:")

    print(
        df[df["Site"] == "California"][
            [
                "Station",
                "Latitude",
                "Longitude",
            ]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(14, 17),
        layout="constrained",
    )

    for i, (title, site) in enumerate(REGIONS):

        group = df[
            df["Site"] == site
        ]

        print(
            f"Drawing {title}: "
            f"n={len(group)}",
            flush=True,
        )

        if group.empty:

            axes.flat[i].axis("off")
            continue

        draw_region(
            axes.flat[i],
            group,
            title,
            chr(97 + i),
        )

    axes.flat[-1].axis("off")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        "\nSaved:",
        args.output,
    )


if __name__ == "__main__":
    main()
