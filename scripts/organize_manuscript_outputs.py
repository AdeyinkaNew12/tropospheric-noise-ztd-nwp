from pathlib import Path
import csv
import shutil

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop" / "GNSS Final Results"

SOURCES = {
    "Main_Figures": {
        "figures": [
            "Figure1_GNSS_World_Map.pdf",
            "Figure2_ERA5_GNSS_1to1.png",
        ],
        "outputs": [
            "Figure3_ERA5_Residual_TimeSeries.png",
            "Figure4_HRRR_GNSS_1to1.png",
            "Figure5_HRRR_Residual_TimeSeries.png",
            "Figure6_sigma_sp_all_points_OPEN_HRRR_FILLED_ERA5.png",
            "Figure7_sigma_t_station_only_OPEN_HRRR_FILLED_ERA5.png",
        ],
    },
    "Supplementary_Figures": {
        "figures": [
            "FigureS8_Supplementary_Topographic_Map.png",
        ],
    },
    "Supplementary_Tables": {
        "manuscript_outputs/Supplementary_Tables": [
            "Table_S1a.docx",
            "Table_S1b.docx",
            "Table_S2.docx",
            "Table_S3.docx",
            "Table_S4.docx",
            "Table_S5.docx",
            "Table_S6.docx",
            "Table_S7.docx",
            "All_Supplementary_Tables.docx",
        ],
    },
    "Analysis_CSVs": {
        "outputs": [
            "sigma_sp_all_time_steps.csv",
            "sigma_sp_site_summary.csv",
            "Figure7_sigma_t_station_only_outputs.csv",
            "Figure7_stations_used.csv",
        ],
        "manuscript_outputs/Analysis_CSVs": [
            "ERA5_station_level_bias_mae_sigma.csv",
            "ERA5_summary_by_site_bias_mae_sigma.csv",
            "HRRR_station_level_bias_mae_sigma.csv",
            "HRRR_summary_by_region_bias_mae_sigma.csv",
            "overall_bias_mae_sigma_comparison.csv",
        ],
    },
}

def main():
    report = []
    missing = []
    planned = []

    for category, groups in SOURCES.items():
        for source_folder, filenames in groups.items():
            for filename in filenames:
                source = ROOT / source_folder / filename
                destination = DESKTOP / category / filename

                if source.is_file():
                    planned.append((source, destination))
                    status = "PASS"
                else:
                    missing.append(str(source))
                    status = "MISSING"

                report.append({
                    "Category": category,
                    "Filename": filename,
                    "Status": status,
                    "Source": str(source),
                })

    if missing:
        print("\nOUTPUT VERIFICATION FAILED")
        for item in missing:
            print("MISSING:", item)
        raise SystemExit(
            "Missing outputs. Desktop copy was not started."
        )

    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    report_path = DESKTOP / "OUTPUT_VERIFICATION_REPORT.csv"

    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Category", "Filename", "Status", "Source"
            ]
        )
        writer.writeheader()
        writer.writerows(report)

    print("\n" + "=" * 65)
    print("FINAL MANUSCRIPT OUTPUT VERIFICATION")
    print("=" * 65)
    print("Destination:", DESKTOP)
    print("Expected files:", len(report))
    print("Files verified:", len(planned))
    print("Missing files:", len(missing))

    for category in SOURCES:
        folder = DESKTOP / category
        print(category + ":", len(list(folder.iterdir())))

    print("\nPASS: All expected files copied successfully.")
    print("Report:", report_path)

if __name__ == "__main__":
    main()
