from pathlib import Path
import pandas as pd

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "manuscript_outputs" / "Supplementary_Tables"
ANALYSIS = ROOT / "manuscript_outputs" / "Analysis_CSVs"

TABLES.mkdir(parents=True, exist_ok=True)


def read_csv(folder, filename, expected_rows):
    path = folder / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != expected_rows:
        raise ValueError(
            f"{filename}: expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    return df


def number(value, decimals=2):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def integer(value):
    return str(int(value))


def site(value):
    return str(value).replace("Japan1", "Japan-1").replace(
        "Japan2", "Japan-2"
    )


def build_document(doc, caption, headers, rows, note=""):
    if not doc.sections[0].orientation == WD_ORIENT.LANDSCAPE:
        section = doc.sections[0]
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11)
        section.page_height = Inches(8.5)
        section.left_margin = Inches(0.45)
        section.right_margin = Inches(0.45)
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(8)

    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(caption)
    run.bold = True
    run.font.size = Pt(10)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for cell, heading in zip(table.rows[0].cells, headers):
        cell.text = heading

    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat_header = OxmlElement("w:tblHeader")
    repeat_header.set(qn("w:val"), "true")
    header_properties.append(repeat_header)

    for values in rows:
        if len(values) != len(headers):
            raise ValueError(
                f"{caption}: incorrect number of row values"
            )
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = str(value)

    font_size = 6.5 if len(headers) >= 10 else 7.5

    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)

                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(font_size)
                    run.bold = row_index == 0

    if note:
        p = doc.add_paragraph(note)
        p.paragraph_format.space_before = Pt(6)

    return table


def save_table(name, caption, headers, rows, note=""):
    doc = Document()
    build_document(doc, caption, headers, rows, note)

    path = TABLES / f"Table_{name}.docx"
    doc.save(path)

    check = Document(path)
    if len(check.tables) != 1:
        raise ValueError(f"Word verification failed: {name}")

    if len(check.tables[0].rows) != len(rows) + 1:
        raise ValueError(f"Row verification failed: {name}")

    print(f"PASS: {name}: {len(rows)} rows")
    return caption, headers, rows, note


def main():
    combined_tables = []

    s1 = read_csv(
        TABLES, "Table_S1_GNSS_Stations_FINAL.csv", 52
    )

    s1a = s1[s1["Site"] != "Missouri"]

    rows = [
        [
            site(r["Site"]), r["Dataset"], r["ID"],
            number(r["Lat"], 4), number(r["Lon"], 4),
            number(r["Elev (m)"], 2),
            integer(r["N_Obs"])
        ]
        for _, r in s1a.iterrows()
    ]

    combined_tables.append(save_table(
        "S1a",
        "Table S1a. GNSS station characteristics for "
        "non-Missouri study regions.",
        ["Site", "Model", "Station", "Latitude (°)",
         "Longitude (°)", "Elevation (m)", "Observations"],
        rows,
        "Note. Missouri field stations are presented "
        "separately in Table S1b."
    ))

    mo = read_csv(
        TABLES, "Table_S7_UPDATED_Missouri_observations.csv", 122
    )

    rows = [
        [
            r["Model"], r["ID"],
            number(r["Lat"], 4), number(r["Lon"], 4),
            number(r["Elev_m"], 2),
            r["Date"], r["Time"]
        ]
        for _, r in mo.iterrows()
    ]

    combined_tables.append(save_table(
        "S1b",
        "Table S1b. Missouri GNSS field station "
        "characteristics and individual acquisition dates and times.",
        ["Model", "Station", "Latitude (°)", "Longitude (°)",
         "Elevation (m)", "Date", "Time"],
        rows,
        "Note. Dates and times are reported for each "
        "model-specific observation record."
    ))

    for model, expected, table_id in [
        ("ERA5", 41, "S2"),
        ("HRRR", 11, "S3")
    ]:
        df = read_csv(
            ANALYSIS,
            f"{model}_station_level_bias_mae_sigma.csv",
            expected
        )

        rows = [
            [
                site(r["Name"]), r["Station"],
                integer(r["N_times"]),
                number(r["Bias_mm"]),
                number(r["MAE_mm"]),
                number(r["Sigma_t_mm"])
            ]
            for _, r in df.iterrows()
        ]

        combined_tables.append(save_table(
            table_id,
            f"Table {table_id}. Station-level bias, "
            f"mean absolute error (MAE), and temporal "
            f"standard deviation for {model}.",
            ["Site", "Station", "N", "Bias (mm)",
             "MAE (mm)", "σt (mm)"],
            rows,
            "Note. Statistics are calculated from valid "
            "station residuals; σt is the sample standard deviation."
        ))

    overall = read_csv(
        ANALYSIS, "overall_bias_mae_sigma_comparison.csv", 2
    )

    rows = [
        [
            r["Dataset"], integer(r["N_stations"]),
            integer(r["N_observations"]),
            number(r["Overall_Bias_mm"]),
            number(r["Overall_MAE_mm"]),
            number(r["Overall_Sigma_t_mm"])
        ]
        for _, r in overall.iterrows()
    ]

    combined_tables.append(save_table(
        "S4",
        "Table S4. Overall ERA5 and HRRR residual "
        "statistics across GNSS stations.",
        ["Model", "Stations", "Observations",
         "Mean bias (mm)", "Mean MAE (mm)", "Mean σt (mm)"],
        rows,
        "Note. Overall statistics are unweighted averages "
        "of station-level statistics."
    ))

    spatial = read_csv(
        ROOT / "outputs", "sigma_sp_site_summary.csv", 13
    )

    rows = [
        [
            r["Dataset"], site(r["SiteName"]),
            integer(r["N_valid_times"]),
            number(r["Mean_sigma_sp"] * 1000),
            number(r["Min_sigma_sp"] * 1000),
            number(r["Max_sigma_sp"] * 1000)
        ]
        for _, r in spatial.iterrows()
    ]

    combined_tables.append(save_table(
        "S5",
        "Table S5. Site-level spatial standard "
        "deviation of GNSS–NWP residuals.",
        ["Model", "Site", "Valid times",
         "Mean σsp (mm)", "Min σsp (mm)", "Max σsp (mm)"],
        rows,
        "Note. Spatial standard deviations were converted "
        "from meters to millimeters."
    ))

    spacing = read_csv(
        TABLES, "Table_S6_UPDATED_station_spacing.csv", 13
    )

    fields = [
        "Model", "Site", "N_Stations", "N_Pairs",
        "Mean_Horizontal_km", "SD_Horizontal_km",
        "Max_Horizontal_km", "Mean_Vertical_m",
        "SD_Vertical_m", "Mean_3D_km",
        "SD_3D_km", "Elevation_Range_m"
    ]

    headers = [
        "Model", "Site", "Stations", "Pairs",
        "Mean horiz. (km)", "SD horiz. (km)",
        "Max horiz. (km)", "Mean vertical (m)",
        "SD vertical (m)", "Mean 3-D (km)",
        "SD 3-D (km)", "Elevation range (m)"
    ]

    rows = []
    for _, r in spacing.iterrows():
        values = []
        for field in fields:
            if field == "Site":
                values.append(site(r[field]))
            elif field == "Model":
                values.append(r[field])
            elif field in ("N_Stations", "N_Pairs"):
                values.append(integer(r[field]))
            else:
                values.append(number(r[field]))
        rows.append(values)

    combined_tables.append(save_table(
        "S6",
        "Table S6. Station spacing and elevation "
        "characteristics for each GNSS station group.",
        headers, rows,
        "Note. Distance statistics are based on unique "
        "station pairs within each group."
    ))

    observations = read_csv(
        TABLES, "Table_S7_UPDATED_station_summary.csv", 52
    )

    rows = [
        [
            r["Model"], site(r["Site"]), r["ID"],
            number(r["Lat"], 4), number(r["Lon"], 4),
            number(r["Elev_m"], 2),
            r["Start_Date"], r["End_Date"],
            r["Time"], integer(r["N_Obs"])
        ]
        for _, r in observations.iterrows()
    ]

    combined_tables.append(save_table(
        "S7",
        "Table S7. GNSS station observation periods "
        "and observation counts.",
        ["Model", "Site", "Station", "Lat (°)", "Lon (°)",
         "Elev. (m)", "Start date", "End date",
         "Time", "N"],
        rows,
        "Note. Missouri's individual acquisition dates "
        "and times are listed in Table S1b."
    ))

    combined = Document()

    for index, (caption, headers, rows, note) in enumerate(
        combined_tables
    ):
        if index:
            combined.add_page_break()

        build_document(
            combined, caption, headers, rows, note
        )

    combined_path = TABLES / "All_Supplementary_Tables.docx"
    combined.save(combined_path)

    check = Document(combined_path)

    if len(check.tables) != 8:
        raise ValueError(
            "Combined Word document must contain eight tables."
        )

    print("\nALL WORD TABLES CONFIGURED")
    print("Individual tables: 8")
    print("Combined document:", combined_path)


if __name__ == "__main__":
    main()
