from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from pyproj import Geod
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

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

geod = Geod(ellps="WGS84")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c

    normalized = {
        str(c).lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }

    for c in candidates:
        key = c.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"Missing {candidates}. Available: {df.columns.tolist()}"
    )


results = []
pair_rows = []

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
        hgt_col = find_col(
            df, ["Hgt_m", "Elevation", "Hgt", "Height"]
        )

        for col in [lat_col, lon_col, hgt_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=[id_col, lat_col, lon_col, hgt_col]
        ).copy()

        df[id_col] = df[id_col].astype(str).str.strip()

        stations = (
            df.groupby(id_col)
            .agg(
                Lat=(lat_col, "mean"),
                Lon=(lon_col, "mean"),
                Elev=(hgt_col, "mean")
            )
            .reset_index()
        )

        n = len(stations)

        if n < 2:
            raise ValueError(
                f"{model} {site}: fewer than two stations."
            )

        horizontal = []
        vertical = []
        distance_3d = []

        for (_, a), (_, b) in combinations(
            stations.iterrows(), 2
        ):
            _, _, horizontal_m = geod.inv(
                a["Lon"], a["Lat"],
                b["Lon"], b["Lat"]
            )

            h_km = horizontal_m / 1000
            v_m = abs(a["Elev"] - b["Elev"])
            d3_km = np.sqrt(h_km**2 + (v_m / 1000)**2)

            horizontal.append(h_km)
            vertical.append(v_m)
            distance_3d.append(d3_km)

            pair_rows.append({
                "Model": model,
                "Site": site,
                "Station_1": a[id_col],
                "Station_2": b[id_col],
                "Horizontal_km": h_km,
                "Vertical_m": v_m,
                "Distance_3D_km": d3_km
            })

        def sd(values):
            return float(np.std(values, ddof=1))

        results.append({
            "Model": model,
            "Site": site,
            "N_Stations": n,
            "N_Pairs": len(horizontal),
            "Mean_Horizontal_km": np.mean(horizontal),
            "SD_Horizontal_km": sd(horizontal),
            "Max_Horizontal_km": np.max(horizontal),
            "Mean_Vertical_m": np.mean(vertical),
            "SD_Vertical_m": sd(vertical),
            "Mean_3D_km": np.mean(distance_3d),
            "SD_3D_km": sd(distance_3d),
            "Elevation_Range_m":
                stations["Elev"].max() - stations["Elev"].min()
        })

summary = pd.DataFrame(results)
pairs = pd.DataFrame(pair_rows)

summary_csv = OUT / "Table_S6_UPDATED_station_spacing.csv"
pairs_csv = OUT / "Table_S6_station_pair_distances.csv"

summary.to_csv(summary_csv, index=False)
pairs.to_csv(pairs_csv, index=False)

doc = Document()
section = doc.sections[0]
section.orientation = WD_ORIENT.LANDSCAPE
section.page_width = Inches(11)
section.page_height = Inches(8.5)
section.left_margin = Inches(0.35)
section.right_margin = Inches(0.35)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(8)

caption = doc.add_paragraph()
caption.add_run(
    "Table S6. Station spacing and elevation "
    "characteristics for each GNSS station group."
).bold = True

headers = [
    "Model", "Site", "N Stations",
    "Mean Horizontal\nDistance (km)",
    "SD Horizontal\nDistance (km)",
    "Max Horizontal\nDistance (km)",
    "Mean Vertical\nDifference (m)",
    "SD Vertical\nDifference (m)",
    "Mean 3-D\nDistance (km)",
    "SD 3-D\nDistance (km)",
    "Elevation\nRange (m)"
]

fields = [
    "Model", "Site", "N_Stations",
    "Mean_Horizontal_km", "SD_Horizontal_km",
    "Max_Horizontal_km", "Mean_Vertical_m",
    "SD_Vertical_m", "Mean_3D_km",
    "SD_3D_km", "Elevation_Range_m"
]

table = doc.add_table(rows=1, cols=len(headers))
table.style = "Table Grid"
table.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, heading in enumerate(headers):
    table.rows[0].cells[i].text = heading

for _, r in summary.iterrows():
    cells = table.add_row().cells

    for i, field in enumerate(fields):
        value = r[field]

        if field == "Site":
            value = str(value).replace("Japan1", "Japan-1")
            value = value.replace("Japan2", "Japan-2")
        elif field == "N_Stations":
            value = str(int(value))
        elif field == "Model":
            value = str(value)
        elif field.endswith("_m"):
            value = f"{value:.1f}"
        else:
            value = f"{value:.2f}"

        cells[i].text = str(value)

for row_index, row in enumerate(table.rows):
    for cell in row.cells:
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(7)
                if row_index == 0:
                    run.bold = True

doc.add_paragraph(
    "Note. Horizontal and 3-D distance statistics are "
    "calculated from all unique station pairs within "
    "each group. Vertical difference is the absolute "
    "elevation difference between station pairs, and "
    "elevation range is the difference between the "
    "maximum and minimum station elevations."
)

word_file = OUT / "Table_S6_UPDATED_station_spacing.docx"
doc.save(word_file)

check_doc = Document(word_file)

assert len(summary) == 13
assert len(check_doc.tables) == 1
assert len(check_doc.tables[0].rows) == 14
assert len(check_doc.tables[0].columns) == 11

print("\nTABLE S6 GENERATED AND VERIFIED")
print(summary.to_string(index=False))
print("\nWord document:", word_file)
print("Summary CSV:", summary_csv)
print("Station-pair CSV:", pairs_csv)
