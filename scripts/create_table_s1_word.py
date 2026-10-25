from pathlib import Path
import pandas as pd

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# ---------------------------------------------------
# INPUT AND OUTPUT
# ---------------------------------------------------

folder = (
    Path.home()
    / "Desktop"
    / "GNSS_Final_Results_20261009_202753"
)

input_file = folder / "Table_S1_GNSS_Stations_FINAL.xlsx"

output_file = folder / "Table_S1_Updated_ERA5_HRRR.docx"

if not input_file.exists():
    raise SystemExit(f"Excel file not found: {input_file}")

# ---------------------------------------------------
# READ FINAL STATION DATA
# ---------------------------------------------------

df = pd.read_excel(input_file, sheet_name="All Stations")

columns = [
    "Site",
    "Dataset",
    "ID",
    "Lat",
    "Lon",
    "Elev (m)",
    "Start Date",
    "End Date",
    "N_Obs",
]

missing = [c for c in columns if c not in df.columns]

if missing:
    raise SystemExit(f"Missing required columns: {missing}")

df = df[columns].copy()

if df[columns].isna().any().any():
    raise SystemExit(
        "Some required values are missing. "
        "Please check the Excel workbook."
    )

df["Dataset"] = df["Dataset"].astype(str).str.upper().str.strip()

if not df["Dataset"].isin(["ERA5", "HRRR"]).all():
    raise SystemExit(
        "Unexpected dataset labels found. "
        "Check the Dataset column."
    )

# Preserve the region ordering from the supplementary map
site_order = [
    "Alaska",
    "Australia",
    "California",
    "Chile",
    "Germany",
    "Guadeloupe",
    "Italy",
    "Japan1",
    "Japan2",
    "Missouri",
    "Utah",
]

df["Site"] = df["Site"].replace({
    "Europe": "Germany",
    "Japan": "Japan1",
})

df["_order"] = df["Site"].apply(
    lambda x: site_order.index(x)
    if x in site_order else len(site_order)
)

df = df.sort_values(
    ["_order", "Site", "Dataset", "ID"]
).drop(columns="_order")

# ---------------------------------------------------
# CREATE WORD DOCUMENT
# ---------------------------------------------------

doc = Document()

section = doc.sections[0]
section.top_margin = Inches(0.65)
section.bottom_margin = Inches(0.65)
section.left_margin = Inches(0.45)
section.right_margin = Inches(0.45)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(9)

title = doc.add_paragraph()
title.add_run("Supplemental Tables").bold = True

caption = doc.add_paragraph()
caption.add_run(
    "Table S1. GNSS stations used in the ERA5 and HRRR analyses."
).bold = True

table = doc.add_table(rows=1, cols=len(columns))
table.style = "Table Grid"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = True

# ---------------------------------------------------
# TABLE HEADER
# ---------------------------------------------------

for i, column in enumerate(columns):
    table.rows[0].cells[i].text = column

# ---------------------------------------------------
# DATE FORMATTING
# ---------------------------------------------------

def format_date(value):
    if pd.isna(value):
        return ""

    dt = pd.to_datetime(value, errors="coerce")

    if pd.isna(dt):
        return str(value)

    return f"{dt.month}/{dt.day}/{dt.year % 100:02d}"

# ---------------------------------------------------
# ADD TABLE ROWS
# ---------------------------------------------------

for _, row in df.iterrows():

    cells = table.add_row().cells

    for i, column in enumerate(columns):

        value = row[column]

        if column in ["Lat", "Lon"]:
            text = f"{float(value):.3f}"

        elif column == "Elev (m)":
            text = f"{float(value):.1f}"

        elif column == "N_Obs":
            text = str(int(value))

        elif column in ["Start Date", "End Date"]:
            text = format_date(value)

        else:
            text = str(value)

        cells[i].text = text

# ---------------------------------------------------
# FORMAT CELLS
# ---------------------------------------------------

for row_index, row in enumerate(table.rows):

    for cell in row.cells:

        cell.vertical_alignment = (
            WD_CELL_VERTICAL_ALIGNMENT.CENTER
        )

        for paragraph in cell.paragraphs:

            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)

            for run in paragraph.runs:

                run.font.name = "Times New Roman"
                run.font.size = Pt(8)

                if row_index == 0:
                    run.bold = True

# Repeat header on each page
header_row = table.rows[0]._tr
trPr = header_row.get_or_add_trPr()
tblHeader = OxmlElement("w:tblHeader")
tblHeader.set(qn("w:val"), "true")
trPr.append(tblHeader)

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

doc.save(output_file)

print("\nTABLE S1 CREATED SUCCESSFULLY")
print("Total records:", len(df))
print("\nRecords by dataset:")
print(df["Dataset"].value_counts().to_string())

print("\nRecords by site and dataset:")
print(
    df.groupby(["Site", "Dataset"])
    .size()
    .to_string()
)

print("\nSaved to:")
print(output_file)
