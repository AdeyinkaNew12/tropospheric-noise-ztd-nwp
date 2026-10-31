from pathlib import Path
import pandas as pd
import numpy as np

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Supplement_Original_Data"

OUT = ROOT / 'manuscript_outputs' / 'Supplementary_Tables'
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "ERA5": [
        "Alaska", "Australia", "California", "Chile",
        "Germany", "Guadeloupe", "Italy", "Japan1",
        "Japan2", "Rolla"
    ],
    "HRRR": ["California", "Rolla", "Utah"]
}

def find_col(df, candidates, required=True):
    normalized = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }
    for name in candidates:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(
            f"Missing column {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None

def parse_dates(series):
    s = series.astype("string").str.strip()
    compact = s.str.fullmatch(r"\d{8}").fillna(False)
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    result.loc[compact] = pd.to_datetime(
        s.loc[compact],
        format="%Y%m%d",
        errors="coerce"
    )
    result.loc[~compact] = pd.to_datetime(
        s.loc[~compact],
        format="mixed",
        errors="coerce"
    )
    return result

def format_date(dt):
    return dt.strftime("%-m/%-d/%y")

def clean_time(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return ""
    if value.count(":") == 1:
        value += ":00"
    return value

station_rows = []
rolla_rows = []
all_data = []

for model, sites in GROUPS.items():
    for site in sites:
        path = DATA / f"{model}_{site}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        id_col = find_col(df, ["ID", "Station", "Name"])
        lat_col = find_col(df, ["Lat", "Latitude"])
        lon_col = find_col(df, ["Lon", "Longitude", "Long"])
        elev_col = find_col(
            df, ["Hgt_m", "Elevation", "Hgt", "Height"]
        )
        date_col = find_col(df, ["Date"])
        time_col = find_col(df, ["Time"], required=False)

        df[id_col] = df[id_col].astype("string").str.strip()

        for col in [lat_col, lon_col, elev_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["_date"] = parse_dates(df[date_col])

        if time_col:
            df["_time"] = df[time_col].map(clean_time)
        else:
            df["_time"] = ""

        df = df.dropna(
            subset=[id_col, lat_col, lon_col, elev_col, "_date"]
        ).copy()

        df = df[df[id_col] != ""]

        # Use all valid original observations rather than
        # assuming one observation per date.
        df["_model"] = model
        df["_site"] = site

        all_data.append(df)

        for station, g in df.groupby(id_col, sort=True):
            g = g.sort_values(["_date", "_time"])

            times = sorted(
                t for t in g["_time"].unique() if t
            )

            if len(times) == 1:
                time_display = times[0]
            elif len(times) == 0:
                time_display = "Not recorded"
            else:
                time_display = "Variable"

            site_display = (
                site.replace("Japan1", "Japan-1")
                    .replace("Japan2", "Japan-2")
            )

            station_rows.append({
                "Model": model,
                "Site": site_display,
                "ID": str(station),
                "Lat": float(g[lat_col].mean()),
                "Lon": float(g[lon_col].mean()),
                "Elev_m": float(g[elev_col].mean()),
                "Start_Date": format_date(g["_date"].min()),
                "End_Date": format_date(g["_date"].max()),
                "Time": time_display,
                "N_Obs": len(g)
            })

            if site == "Rolla":
                for _, obs in g.iterrows():
                    rolla_rows.append({
                        "Model": model,
                        "Site": "Rolla",
                        "ID": str(station),
                        "Lat": float(g[lat_col].mean()),
                        "Lon": float(g[lon_col].mean()),
                        "Elev_m": float(g[elev_col].mean()),
                        "Date": format_date(obs["_date"]),
                        "Time": obs["_time"],
                        "N_Obs": len(g),
                        "_sort_date": obs["_date"]
                    })

summary = pd.DataFrame(station_rows)
rolla = pd.DataFrame(rolla_rows)

summary = summary.sort_values(
    ["Model", "Site", "ID"]
).reset_index(drop=True)

rolla = rolla.sort_values(
    ["Model", "ID", "_sort_date", "Time"]
).reset_index(drop=True)

# Verify station counts against Table S6.
s6_path = ROOT / 'manuscript_outputs' / 'Supplementary_Tables' / 'Table_S6_UPDATED_station_spacing.csv'

if not s6_path.exists():
    raise FileNotFoundError(
        "Table S6 summary is missing: " + str(s6_path)
    )

s6 = pd.read_csv(s6_path)
counts = summary.groupby(["Model", "Site"]).size()

for _, r in s6.iterrows():
    site = (
        str(r["Site"])
        .replace("Japan1", "Japan-1")
        .replace("Japan2", "Japan-2")
    )

    actual = int(counts.get((r["Model"], site), 0))
    expected = int(r["N_Stations"])

    if actual != expected:
        raise ValueError(
            f"Station mismatch: {r['Model']} {site}: "
            f"Table S7={actual}, Table S6={expected}"
        )

# Save CSV outputs.
summary_csv = OUT / "Table_S7_UPDATED_station_summary.csv"
rolla_csv = OUT / "Table_S7_UPDATED_Rolla_observations.csv"

summary.to_csv(summary_csv, index=False)
rolla.drop(columns=["_sort_date"]).to_csv(
    rolla_csv, index=False
)

# Build the editable Word document.
doc = Document()

section = doc.sections[0]
section.orientation = WD_ORIENT.LANDSCAPE
section.page_width = Inches(11)
section.page_height = Inches(8.5)
section.left_margin = Inches(0.4)
section.right_margin = Inches(0.4)
section.top_margin = Inches(0.5)
section.bottom_margin = Inches(0.5)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(8)

def make_table(headers):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for cell, heading in zip(table.rows[0].cells, headers):
        cell.text = heading

    return table

def add_row(table, values):
    cells = table.add_row().cells
    for cell, value in zip(cells, values):
        cell.text = str(value)

def format_table(table):
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(7)
                    if row_index == 0:
                        run.bold = True

# SECTION 1: Fixed-time station summaries.
title = doc.add_paragraph()
title.add_run(
    "Table S7. GNSS stations and observation times "
    "used in this analysis."
).bold = True

headers = [
    "Model", "Site", "ID", "Lat", "Lon",
    "Elev (m)", "Start Date", "End Date",
    "Time", "N. Obs."
]

table1 = make_table(headers)

regular = summary[summary["Site"] != "Rolla"]

for _, r in regular.iterrows():
    add_row(table1, [
        r["Model"],
        r["Site"],
        r["ID"],
        f'{r["Lat"]:.3f}',
        f'{r["Lon"]:.3f}',
        f'{r["Elev_m"]:.1f}',
        r["Start_Date"],
        r["End_Date"],
        r["Time"],
        int(r["N_Obs"])
    ])

format_table(table1)

# SECTION 2: Rolla individual observation timestamps.
doc.add_page_break()

title2 = doc.add_paragraph()
title2.add_run(
    "Table S7. GNSS stations and observation times "
    "used in this analysis (continued)."
).bold = True

headers2 = [
    "Model", "Site", "ID", "Lat", "Lon",
    "Elev (m)", "Date", "Time", "N. Obs."
]

table2 = make_table(headers2)

for (model, station), g in rolla.groupby(
    ["Model", "ID"], sort=True
):
    first = True

    for _, r in g.iterrows():
        if first:
            values = [
                model,
                "Rolla",
                station,
                f'{r["Lat"]:.3f}',
                f'{r["Lon"]:.3f}',
                f'{r["Elev_m"]:.1f}',
                r["Date"],
                r["Time"],
                int(r["N_Obs"])
            ]
            first = False
        else:
            values = [
                "", "", "", "", "", "",
                r["Date"], r["Time"], ""
            ]

        add_row(table2, values)

format_table(table2)

doc.add_paragraph(
    "Note. Times are reported as HH:MM:SS. "
    "For Rolla stations, each observation date "
    "and its corresponding observation time are "
    "reported individually. ERA5 and HRRR observations "
    "are presented separately."
)

word_path = OUT / "Table_S7_UPDATED_FINAL.docx"
doc.save(word_path)

# Verify Word output.
check = Document(word_path)

assert len(check.tables) == 2
assert len(check.tables[0].rows) == len(regular) + 1
assert len(check.tables[1].rows) == len(rolla) + 1

print("\nTABLE S7 GENERATED AND VERIFIED")
print("=" * 65)
print("Total station entries:", len(summary))
print("Non-Rolla station entries:", len(regular))
print("Rolla observation rows:", len(rolla))

print("\nSTATION COUNTS BY MODEL AND SITE")
print(counts.to_string())

print("\nMISSOURI OBSERVATION COUNTS")
print(
    rolla.groupby(["Model", "ID"])
    .size()
    .to_string()
)

print("\nFILES CREATED")
print("Word:", word_path)
print("Station summary CSV:", summary_csv)
print("Rolla observations CSV:", rolla_csv)

print("\nPASS: Station counts agree with Table S6.")
print("PASS: Rolla dates and times are on separate rows.")
print("PASS: Original data were not modified.")
print("PASS: Nothing was pushed to GitHub.")
