from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]

candidates = sorted((ROOT / "outputs").rglob("Table_S6.docx"))

if not candidates:
    raise SystemExit("ERROR: Table_S6.docx not found in outputs.")

for path in candidates:
    doc = Document(path)

    if not doc.tables:
        raise SystemExit(f"ERROR: No table found in {path}")

    table = doc.tables[0]

    if len(table.columns) != 12:
        raise SystemExit(
            f"ERROR: Expected 12 columns, found {len(table.columns)}"
        )

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)

    table.autofit = False

    widths = [
        0.55, 0.83, 0.42, 0.42,
        0.64, 0.60, 0.62, 0.68,
        0.62, 0.61, 0.61, 0.80
    ]

    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)

    for row_index, row in enumerate(table.rows):
        for col_index, cell in enumerate(row.cells):
            cell.width = Inches(widths[col_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

            tcPr = cell._tc.get_or_add_tcPr()

            borders = tcPr.first_child_found_in("w:tcBorders")
            if borders is None:
                borders = OxmlElement("w:tcBorders")
                tcPr.append(borders)

            for edge in ("top", "left", "bottom", "right"):
                tag = qn(f"w:{edge}")
                border = borders.find(tag)

                if border is None:
                    border = OxmlElement(f"w:{edge}")
                    borders.append(border)

                border.set(qn("w:val"), "single")
                border.set(qn("w:sz"), "8")
                border.set(qn("w:color"), "000000")

            if row_index == 0:
                direction = tcPr.find(qn("w:textDirection"))
                if direction is None:
                    direction = OxmlElement("w:textDirection")
                    tcPr.append(direction)
                direction.set(qn("w:val"), "btLr")

            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)

                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9)
                    if row_index == 0:
                        run.bold = True

        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(OxmlElement("w:cantSplit"))

    table.rows[0].height = Inches(1.55)

    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith("Table S6"):
            paragraph.paragraph_format.space_after = Pt(10)
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)

    doc.save(path)
    print("FORMATTED:", path)
    print("Rows:", len(table.rows))
    print("Columns:", len(table.columns))
