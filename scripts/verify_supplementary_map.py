#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

# Import functions and settings from your map script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supplementary_station_map import (
    REGIONS,
    CALIFORNIA,
    load_stations,
    get_extent,
    get_terrain,
    TO_M,
    TO_LL,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

HOME = Path.home()

STATION_CSV = (
    HOME
    / "Desktop"
    / "All data For Master Thesis "
    / "All_data_For_Master_Thesis"
    / "StationMapfile.csv"
)

OUTPUT_DIR = (
    HOME
    / "Desktop"
    / "GNSS_Final_Results_20261009_202753"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEOD = Geod(ellps="WGS84")

EXPECTED_COUNTS = {
    "Alaska": 5,
    "Australia": 4,
    "California": 3,
    "Chile": 4,
    "Germany": 4,
    "Guadeloupe": 5,
    "Italy": 3,
    "Japan-1": 5,
    "Japan-2": 4,
    "Missouri": 4,
    "Utah": 4,
}

# Values displayed in your latest PNG.
# These are comparison targets, NOT independently
# verified elevation measurements.
DISPLAYED_RELIEF = {
    "Alaska": 1246,
    "Australia": 213,
    "California": 1269,
    "Chile": 404,
    "Germany": 14,
    "Guadeloupe": 1254,
    "Italy": 527,
    "Japan-1": 151,
    "Japan-2": 165,
    "Missouri": 107,
    "Utah": 500,
}

# --------------------------------------------------
# VERIFY CALIFORNIA
# --------------------------------------------------

df = load_stations(STATION_CSV)

cal = df[df["Site"] == "California"].copy()

expected_cal = {
    "P665": (40.456, -121.526),
    "P666": (40.465, -121.515),
    "P668": (40.509, -121.463),
}

print("\n" + "=" * 65)
print("CALIFORNIA STATION VERIFICATION")
print("=" * 65)

cal_ok = True

actual_ids = set(cal["Station"])

if actual_ids != set(expected_cal):
    cal_ok = False
    print("FAIL: California station IDs do not match.")

for station, (lat, lon) in expected_cal.items():

    rows = cal[cal["Station"] == station]

    if len(rows) != 1:
        cal_ok = False
        print(f"FAIL: {station} count = {len(rows)}")
        continue

    row = rows.iloc[0]

    correct = (
        np.isclose(row["Latitude"], lat, atol=1e-7)
        and np.isclose(row["Longitude"], lon, atol=1e-7)
    )

    if not correct:
        cal_ok = False

    print(
        f"{station}: "
        f"({row['Latitude']:.3f}, "
        f"{row['Longitude']:.3f}) "
        f"{'PASS' if correct else 'FAIL'}"
    )

print(
    "P667 excluded:",
    "PASS" if "P667" not in actual_ids else "FAIL"
)

print(
    "Overall California:",
    "PASS" if cal_ok else "FAIL"
)

# --------------------------------------------------
# VERIFY ALL REGIONS
# --------------------------------------------------

results = []

print("\n" + "=" * 65)
print("REGIONAL MAP VERIFICATION")
print("=" * 65)

for title, site in REGIONS:

    group = df[df["Site"] == site].copy()

    n = len(group)

    expected_n = EXPECTED_COUNTS[title]

    count_ok = n == expected_n

    extent = get_extent(group)

    west, east, south, north = extent

    # Verify all stations fall within map boundaries.
    x, y = TO_M.transform(
        group["Longitude"].to_numpy(),
        group["Latitude"].to_numpy(),
    )

    inside = bool(
        np.all(x >= west)
        and np.all(x <= east)
        and np.all(y >= south)
        and np.all(y <= north)
    )

    # Geographic extent in kilometers.
    center_y = (south + north) / 2

    lon_w, lat_w = TO_LL.transform(west, center_y)
    lon_e, lat_e = TO_LL.transform(east, center_y)

    _, _, width_m = GEOD.inv(
        lon_w, lat_w,
        lon_e, lat_e,
    )

    width_km = width_m / 1000

    # Reproduce the current script's scale-bar
    # selection logic.
    x0 = west + 0.60 * (east - west)

    scale_km = 5

    for candidate in [5, 3, 1]:

        lon0, lat0 = TO_LL.transform(x0, south)

        lon1, lat1, _ = GEOD.fwd(
            lon0,
            lat0,
            90,
            candidate * 1000,
        )

        x1, _ = TO_M.transform(lon1, lat1)

        scale_km = candidate

        if x1 <= west + 0.97 * (east - west):
            break

    scale_fits = (
        x1 <= west + 0.97 * (east - west)
    )

    _, _, actual_bar_m = GEOD.inv(
        lon0, lat0,
        lon1, lat1,
    )

    scale_error_m = abs(
        actual_bar_m - scale_km * 1000
    )

    # --------------------------------------------------
    # ELEVATION VERIFICATION
    # --------------------------------------------------

    relief_m = np.nan
    terrain_ok = False

    try:

        terrain_rgb, elevation, bounds = (
            get_terrain(extent)
        )

        # Contextily bounds:
        # left, right, bottom, top
        left, right, bottom, top = bounds

        h, w = elevation.shape

        col0 = max(
            0,
            int(np.floor(
                (west - left) / (right - left) * w
            )),
        )

        col1 = min(
            w,
            int(np.ceil(
                (east - left) / (right - left) * w
            )),
        )

        row0 = max(
            0,
            int(np.floor(
                (top - north) / (top - bottom) * h
            )),
        )

        row1 = min(
            h,
            int(np.ceil(
                (top - south) / (top - bottom) * h
            )),
        )

        crop = elevation[row0:row1, col0:col1]

        if crop.size and np.isfinite(crop).any():

            relief_m = float(
                np.nanmax(crop) - np.nanmin(crop)
            )

            terrain_ok = True

    except Exception as exc:

        print(
            f"Terrain verification failed for "
            f"{title}: {exc}"
        )

    displayed = DISPLAYED_RELIEF[title]

    relief_difference = (
        abs(relief_m - displayed)
        if np.isfinite(relief_m)
        else np.nan
    )

    relief_match = bool(
        np.isfinite(relief_difference)
        and relief_difference <= 5
    )

    result = {
        "Region": title,
        "Station_count": n,
        "Expected_count": expected_n,
        "Count_PASS": count_ok,
        "Stations_inside_extent_PASS": inside,
        "Map_width_km": round(width_km, 3),
        "Scale_bar_km": scale_km,
        "Scale_fits_PASS": scale_fits,
        "Scale_error_m": round(scale_error_m, 6),
        "Displayed_DeltaZ_m": displayed,
        "Recomputed_DeltaZ_m": (
            round(relief_m, 1)
            if np.isfinite(relief_m)
            else np.nan
        ),
        "DeltaZ_difference_m": (
            round(relief_difference, 1)
            if np.isfinite(relief_difference)
            else np.nan
        ),
        "DeltaZ_matches_PNG_PASS": relief_match,
        "Terrain_download_PASS": terrain_ok,
    }

    results.append(result)

    print(f"\n{title}")
    print(
        f"  Stations: {n}/{expected_n} "
        f"{'PASS' if count_ok else 'FAIL'}"
    )
    print(
        f"  Stations inside extent: "
        f"{'PASS' if inside else 'FAIL'}"
    )
    print(
        f"  Scale: {scale_km} km; "
        f"fits={'YES' if scale_fits else 'NO'}"
    )
    print(
        f"  Scale distance error: "
        f"{scale_error_m:.6f} m"
    )
    print(
        f"  Displayed Δz: {displayed} m"
    )
    print(
        f"  Recomputed Δz: "
        f"{relief_m:.1f} m"
        if np.isfinite(relief_m)
        else "  Recomputed Δz: unavailable"
    )
    print(
        f"  Δz matches PNG: "
        f"{'PASS' if relief_match else 'CHECK'}"
    )

# --------------------------------------------------
# JAPAN NETWORK COMPARISON
# --------------------------------------------------

j1 = df[df["Site"] == "Japan"][
    ["Station", "Latitude", "Longitude"]
].sort_values("Station").reset_index(drop=True)

j2 = df[df["Site"] == "Japan2"][
    ["Station", "Latitude", "Longitude"]
].sort_values("Station").reset_index(drop=True)

japan_same = j1.equals(j2)

print("\n" + "=" * 65)
print("JAPAN-1 VS JAPAN-2")
print("=" * 65)

print(
    "Same station names and coordinates:",
    "YES" if japan_same else "NO"
)

# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

report = pd.DataFrame(results)

report_path = (
    OUTPUT_DIR
    / "Supplementary_Map_Verification.csv"
)

report.to_csv(report_path, index=False)

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print(
    "Station counts:",
    int(report["Count_PASS"].sum()),
    "/ 11 passed"
)

print(
    "Station extents:",
    int(report["Stations_inside_extent_PASS"].sum()),
    "/ 11 passed"
)

print(
    "Scale-bar fit:",
    int(report["Scale_fits_PASS"].sum()),
    "/ 11 passed"
)

print(
    "Elevation comparison:",
    int(report["DeltaZ_matches_PNG_PASS"].sum()),
    "/ 11 matched"
)

print("\nSaved verification report:")
print(report_path)

print(
    "\nNOTE: This verifies internal consistency "
    "of the map script and source station CSV. "
    "It does not independently establish GNSS "
    "survey accuracy, DEM accuracy, or whether "
    "the published PNG was generated from the "
    "current script."
)
