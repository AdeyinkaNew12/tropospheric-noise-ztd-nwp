from pathlib import Path
import csv
import shutil

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "manuscript_outputs"

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
            "Table_S1_GNSS_Stations_FINAL.csv",
            "Table_S1_GNSS_Stations_FINAL.xlsx",
            "Table_S1_Data_Checks.csv",
            "Table_S6_UPDATED_station_spacing.docx",
            "Table_S6_UPDATED_station_spacing.csv",
            "Table_S6_station_pair_distances.csv",
            "Table_S7_UPDATED_FINAL.docx",
            "Table_S7_UPDATED_station_summary.csv",
            "Table_S7_UPDATED_Missouri_observations.csv",
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

report = []
missing = []

for category, source_groups in SOURCES.items():
    destination_dir = FINAL / category
    destination_dir.mkdir(parents=True, exist_ok=True)

    for source_folder, filenames in source_groups.items():
        for filename in filenames:
            source = ROOT / source_folder / filename
            destination = destination_dir / filename

            if source.is_file():
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                status = "PASS"
            else:
                status = "MISSING"
                missing.append(str(source))

            report.append({
                "Category": category,
                "Filename": filename,
                "Status": status,
                "Source": str(source),
            })

report_path = FINAL / "OUTPUT_VERIFICATION_REPORT.csv"

with report_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Category", "Filename", "Status", "Source"]
    )
    writer.writeheader()
    writer.writerows(report)

print("\n" + "=" * 65)
print("LOCAL MANUSCRIPT OUTPUT VERIFICATION")
print("=" * 65)
print("Output folder:", FINAL)
print("Expected files:", len(report))
print("Files found:", len(report) - len(missing))
print("Missing files:", len(missing))

for category in SOURCES:
    print(category + ":", len(list((FINAL / category).iterdir())))

if missing:
    for filename in missing:
        print("MISSING:", filename)
    raise SystemExit("Output verification FAILED")

print("PASS: ALL 26 FILES VERIFIED")
print("No Desktop files required by organizer.")
