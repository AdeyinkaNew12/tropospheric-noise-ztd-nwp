from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Supplement_Original_Data"
OUT = ROOT / 'manuscript_outputs'
SUMMARY_OUT = ROOT / 'manuscript_outputs' / 'Analysis_CSVs'
FIGURE6 = Path(__file__).resolve().parents[1] / "outputs" / "Figure7_sigma_t_station_only_outputs.csv"

ERA5_REGIONS = [
    "Australia", "Japan1", "Germany", "Alaska", "Japan2",
    "Italy", "Chile", "Guadeloupe", "Missouri", "California"
]
HRRR_REGIONS = ["Missouri", "California", "Utah"]

def find_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"Missing column from {candidates}. "
        f"Available: {df.columns.tolist()}"
    )

def calculate_stations(dataset, regions):
    rows = []

    for region in regions:
        path = DATA / f"{dataset}_{region}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        station_col = find_col(df, ["ID", "Station", "Name"])
        lat_col = find_col(df, ["Lat", "Latitude"])
        elev_col = find_col(df, ["Hgt_m", "Elevation", "Hgt"])

        if dataset == "ERA5":
            residual_col = find_col(
                df, ["Residual(ERA5)", "Residual (ERA5)"]
            )
        else:
            residual_col = find_col(
                df, [
                    "Residual (GNSS)", "Residual(GNSS)",
                    "Residual(HRRR)"
                ]
            )

        df[station_col] = df[station_col].astype("string").str.strip()

        for col in [residual_col, lat_col, elev_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=[station_col, residual_col, lat_col, elev_col]
        ).copy()

        for station, group in df.groupby(station_col):
            residual = group[residual_col]

            if len(residual) < 2:
                continue

            rows.append({
                "Dataset": dataset,
                "Name": region,
                "Station": str(station),
                "Latitude": group[lat_col].mean(),
                "Elevation": group[elev_col].mean(),
                "N_times": len(residual),
                "Bias_m": residual.mean(),
                "MAE_m": residual.abs().mean(),
                "Sigma_t_m": residual.std(ddof=1),
            })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(f"No valid {dataset} stations")

    for metric in ["Bias", "MAE", "Sigma_t"]:
        result[f"{metric}_mm"] = result[f"{metric}_m"] * 1000

    result["MatchedGroup"] = result["Name"].where(
        result["Name"].isin(["Missouri", "California", "Utah"]),
        "None"
    )

    return result.sort_values(
        ["Name", "Station"]
    ).reset_index(drop=True)

era5 = calculate_stations("ERA5", ERA5_REGIONS)
hrrr = calculate_stations("HRRR", HRRR_REGIONS)

combined = pd.concat([era5, hrrr], ignore_index=True)

# ============================================================
# 3. VERIFY AGAINST FINAL FIGURE 6
# ============================================================
if not FIGURE6.is_file():
    raise FileNotFoundError(FIGURE6)

fig6 = pd.read_csv(FIGURE6)
fig6.columns = fig6.columns.str.strip()

required = {"Dataset", "Name", "Station", "N_times", "Sigma_t"}

if not required.issubset(fig6.columns):
    raise ValueError(
        f"Missing Figure 6 columns: {required - set(fig6.columns)}"
    )

keys = ["Dataset", "Name", "Station"]

for col in keys:
    fig6[col] = fig6[col].astype(str).str.strip()
    combined[col] = combined[col].astype(str).str.strip()

if fig6.duplicated(keys).any() or combined.duplicated(keys).any():
    raise ValueError("Duplicate station keys detected")

check = combined.merge(
    fig6[keys + ["N_times", "Sigma_t"]],
    on=keys,
    how="outer",
    suffixes=("_new", "_fig6"),
    indicator=True
)

if not check["_merge"].eq("both").all():
    print(check.loc[
        check["_merge"] != "both", keys + ["_merge"]
    ].to_string(index=False))
    raise ValueError("Station selection differs from Figure 6")

if not (
    check["N_times_new"].astype(int)
    == check["N_times_fig6"].astype(int)
).all():
    raise ValueError("Observation counts differ from Figure 6")

if not np.allclose(
    check["Sigma_t_m"],
    check["Sigma_t"],
    rtol=1e-7,
    atol=1e-9
):
    raise ValueError("Sigma_t differs from Figure 6")

if len(era5) != 41 or len(hrrr) != 11:
    raise ValueError(
        f"Unexpected station counts: ERA5={len(era5)}, "
        f"HRRR={len(hrrr)}"
    )

california = combined[
    combined["Name"].eq("California")
]

if "P667" in set(california["Station"]):
    raise ValueError("Outdated station P667 detected")

print("\n" + "=" * 75)
print("FIGURE 6 VERIFICATION PASSED")
print("=" * 75)
print(f"ERA5 stations: {len(era5)}")
print(f"HRRR stations: {len(hrrr)}")
print("Station selection: PASSED")
print("Observation counts: PASSED")
print("Temporal standard deviations: PASSED")

# ============================================================
# 4. GENERATE SITE AND REGION SUMMARIES
# ============================================================
def make_summary(df):
    return (
        df.groupby(
            ["Dataset", "Name", "MatchedGroup"],
            as_index=False
        )
        .agg(
            N_stations=("Station", "count"),
            N_observations=("N_times", "sum"),
            Mean_Bias_mm=("Bias_mm", "mean"),
            Mean_MAE_mm=("MAE_mm", "mean"),
            Min_MAE_mm=("MAE_mm", "min"),
            Max_MAE_mm=("MAE_mm", "max"),
            Mean_sigma_t_mm=("Sigma_t_mm", "mean")
        )
        .sort_values("Name")
        .reset_index(drop=True)
    )

era5_summary = make_summary(era5)
hrrr_summary = make_summary(hrrr)

# ============================================================
# 5. OVERALL COMPARISON — ORIGINAL ANACONDA METHODOLOGY
# ============================================================
overall = (
    combined.groupby("Dataset", as_index=False)
    .agg(
        N_stations=("Station", "count"),
        N_observations=("N_times", "sum"),
        Overall_Bias_mm=("Bias_mm", "mean"),
        Overall_MAE_mm=("MAE_mm", "mean"),
        Overall_Sigma_t_mm=("Sigma_t_mm", "mean")
    )
)

overall["Dataset"] = pd.Categorical(
    overall["Dataset"],
    categories=["ERA5", "HRRR"],
    ordered=True
)

overall = overall.sort_values("Dataset").reset_index(drop=True)
overall["Dataset"] = overall["Dataset"].astype(str)

# ============================================================
# 6. SAVE FIVE CSV OUTPUTS
# ============================================================
SUMMARY_OUT.mkdir(parents=True, exist_ok=True)

outputs = {
    "ERA5_station_level_bias_mae_sigma.csv": era5,
    "ERA5_summary_by_site_bias_mae_sigma.csv": era5_summary,
    "HRRR_station_level_bias_mae_sigma.csv": hrrr,
    "HRRR_summary_by_region_bias_mae_sigma.csv": hrrr_summary,
    "overall_bias_mae_sigma_comparison.csv": overall
}

for filename, dataframe in outputs.items():
    path = SUMMARY_OUT / filename
    dataframe.to_csv(path, index=False, float_format="%.6f")
    print("Saved:", path)

# ============================================================
# 7. DISPLAY TABLE S4
# ============================================================
print("\n" + "=" * 75)
print("TABLE S4 — OVERALL ERA5 AND HRRR COMPARISON")
print("=" * 75)

print(
    overall.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

era5_mae = overall.loc[
    overall["Dataset"].eq("ERA5"), "Overall_MAE_mm"
].iloc[0]

hrrr_mae = overall.loc[
    overall["Dataset"].eq("HRRR"), "Overall_MAE_mm"
].iloc[0]

difference = (hrrr_mae - era5_mae) / era5_mae * 100

print(f"\nHRRR MAE difference relative to ERA5: {difference:+.2f}%")
print("\nALL FIVE SUMMARY CSV FILES GENERATED SUCCESSFULLY")
