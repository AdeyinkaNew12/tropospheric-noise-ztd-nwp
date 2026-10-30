#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "Supplement_Original_Data"

OUT_DIR = ROOT / 'manuscript_outputs' / 'Supplementary_Tables'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Corrected California coordinates and elevations.
# These are provided reference values.
CALIFORNIA = {
    "P665": (40.456, -121.526, 2326.799),
    "P666": (40.465, -121.515, 2471.543),
    "P668": (40.509, -121.463, 1982.498),
}

REGION_ORDER = [
    "Alaska",
    "Australia",
    "California",
    "Chile",
    "Europe",
    "Guadeloupe",
    "Italy",
    "Japan",
    "Japan2",
    "Missouri",
    "Utah",
]

def find_column(df, names):
    lookup = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }

    for name in names:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]

    return None


def parse_dates(values):
    s = values.astype(str).str.strip()

    # YYYYMMDD or YYYYMMDD.0
    compact = s.str.replace(r"\.0$", "", regex=True)

    result = pd.to_datetime(
        compact.where(compact.str.fullmatch(r"\d{8}")),
        format="%Y%m%d",
        errors="coerce",
    )

    # ISO and other common date formats.
    missing = result.isna()

    if missing.any():
        result.loc[missing] = pd.to_datetime(
            s.loc[missing],
            errors="coerce",
            format="mixed",
        )

    return result


def get_region_dataset(filename):
    stem = filename.removesuffix(".csv")

    if stem.startswith("ERA5_"):
        return stem[5:], "ERA5"

    if stem.startswith("HRRR_"):
        return stem[5:], "HRRR"

    return None, None


def process_file(path):
    region, dataset = get_region_dataset(path.name)

    if dataset is None:
        return [], []

    df = pd.read_csv(path)

    station_col = find_column(
        df, ["ID", "Station", "StationID", "Station_Name"]
    )

    lat_col = find_column(
        df, ["Lat", "Latitude"]
    )

    lon_col = find_column(
        df, ["Lon", "Longitude", "Long"]
    )

    elev_col = find_column(
        df, ["Hgt_m", "Elev_m", "Elevation", "Height", "Elev"]
    )

    date_col = find_column(
        df, ["Date", "Datetime", "Timestamp", "ObservationDate"]
    )

    problems = []

    if station_col is None:
        problems.append({
            "File": path.name,
            "Issue": "Missing station ID column",
        })
        return [], problems

    if date_col is None:
        problems.append({
            "File": path.name,
            "Issue": "Missing date column",
        })

    df[station_col] = (
        df[station_col].astype(str).str.strip()
    )

    if region == "California":
        df = df[
            df[station_col].isin(CALIFORNIA)
        ].copy()

    if date_col is not None:
        df["_date"] = parse_dates(df[date_col])
    else:
        df["_date"] = pd.NaT

    records = []

    for station, g in df.groupby(station_col, sort=True):
        lat = np.nan
        lon = np.nan
        elev = np.nan

        if lat_col:
            lat = pd.to_numeric(
                g[lat_col], errors="coerce"
            ).median()

        if lon_col:
            lon = pd.to_numeric(
                g[lon_col], errors="coerce"
            ).median()

        if elev_col:
            elev = pd.to_numeric(
                g[elev_col], errors="coerce"
            ).median()

        if region == "California":
            lat, lon, elev = CALIFORNIA[station]

        valid_dates = g["_date"].dropna()

        if valid_dates.empty:
            start = ""
            end = ""
        else:
            start = valid_dates.min().strftime("%Y-%m-%d")
            end = valid_dates.max().strftime("%Y-%m-%d")

        records.append({
            "Site": region,
            "Dataset": dataset,
            "ID": station,
            "Lat": round(float(lat), 6) if pd.notna(lat) else np.nan,
            "Lon": round(float(lon), 6) if pd.notna(lon) else np.nan,
            "Elev (m)": round(float(elev), 2) if pd.notna(elev) else np.nan,
            "Start Date": start,
            "End Date": end,
            "N_Obs": len(g),
            "N_Unique_Dates": valid_dates.nunique(),
            "Source_File": path.name,
        })

        if len(valid_dates) != len(g):
            problems.append({
                "File": path.name,
                "Issue": (
                    f"{station}: {len(g) - len(valid_dates)} "
                    "rows have missing/unparseable dates"
                ),
            })

        if lat_col is None or lon_col is None or elev_col is None:
            problems.append({
                "File": path.name,
                "Issue": (
                    f"{station}: one or more geographic/elevation "
                    "columns missing; values may be blank"
                ),
            })

    return records, problems


def main():
    approved_files = [
        "ERA5_Alaska.csv",
        "ERA5_Australia.csv",
        "ERA5_California.csv",
        "ERA5_Chile.csv",
        "ERA5_Germany.csv",
        "ERA5_Guadeloupe.csv",
        "ERA5_Italy.csv",
        "ERA5_Japan1.csv",
        "ERA5_Japan2.csv",
        "ERA5_Missouri.csv",
        "HRRR_California.csv",
        "HRRR_Missouri.csv",
        "HRRR_Utah.csv",
    ]

    files = [DATA_DIR / name for name in approved_files]

    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required input datasets missing: " + ", ".join(missing)
        )

    print("DATA DIRECTORY:", DATA_DIR)
    print("CSV files found:", len(files))

    all_records = []
    all_problems = []

    for path in files:
        region, dataset = get_region_dataset(path.name)

        if dataset is None:
            continue

        print("Processing:", path.name)

        records, problems = process_file(path)

        all_records.extend(records)
        all_problems.extend(problems)

    if not all_records:
        raise SystemExit(
            "No ERA5_*.csv or HRRR_*.csv datasets found. "
            "Check the filenames in the data directory."
        )

    table = pd.DataFrame(all_records)

    table["_region_order"] = table["Site"].apply(
        lambda x: (
            REGION_ORDER.index(x)
            if x in REGION_ORDER
            else len(REGION_ORDER)
        )
    )

    table = table.sort_values(
        ["_region_order", "Dataset", "ID"]
    ).drop(columns="_region_order")

    table_path = OUT_DIR / "Table_S1_GNSS_Stations_FINAL.csv"

    table.to_csv(table_path, index=False)

    problems_path = OUT_DIR / "Table_S1_Data_Checks.csv"

    pd.DataFrame(
        all_problems,
        columns=["File", "Issue"]
    ).to_csv(problems_path, index=False)

    # Excel workbook with separate sheets.
    excel_path = OUT_DIR / "Table_S1_GNSS_Stations_FINAL.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        table.to_excel(
            writer,
            sheet_name="All Stations",
            index=False,
        )

        for dataset in ["ERA5", "HRRR"]:
            subset = table[table["Dataset"] == dataset]

            if not subset.empty:
                subset.to_excel(
                    writer,
                    sheet_name=dataset,
                    index=False,
                )

        pd.DataFrame(
            all_problems,
            columns=["File", "Issue"]
        ).to_excel(
            writer,
            sheet_name="Data Checks",
            index=False,
        )

    print("\n" + "=" * 60)
    print("TABLE S1 SUMMARY")
    print("=" * 60)

    print("\nRows:", len(table))

    print("\nStations by site and dataset:")
    print(
        table.groupby(["Site", "Dataset"])["ID"]
        .nunique()
        .to_string()
    )

    print("\nCalifornia:")
    print(
        table[table["Site"] == "California"]
        .to_string(index=False)
    )

    print("\nMissing coordinates:",
          table[["Lat", "Lon"]].isna().any(axis=1).sum())

    print("Missing elevations:",
          table["Elev (m)"].isna().sum())

    print("Missing date ranges:",
          (table["Start Date"] == "").sum())

    print("\nFiles saved:")
    print(table_path)
    print(excel_path)
    print(problems_path)


if __name__ == "__main__":
    main()
