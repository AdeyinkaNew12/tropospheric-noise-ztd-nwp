import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# ============================================================
# REPOSITORY PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "Supplement_Original_Data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA5_FILES = [
    "ERA5_Australia.csv",
    "ERA5_Japan1.csv",
    "ERA5_Germany.csv",
    "ERA5_Alaska.csv",
    "ERA5_Japan2.csv",
    "ERA5_Italy.csv",
    "ERA5_Chile.csv",
    "ERA5_Guadeloupe.csv",
    "ERA5_Missouri.csv",
    "ERA5_California.csv",
]

HRRR_FILES = [
    "HRRR_Missouri.csv",
    "HRRR_California.csv",
    "HRRR_Utah.csv",
]

OUT_PNG = OUTPUT_DIR / "Figure5_sigma_sp_all_points_OPEN_HRRR_FILLED_ERA5.png"
OUT_CSV_ALL_TIMES = OUTPUT_DIR / "sigma_sp_all_time_steps.csv"
OUT_CSV_SUMMARY = OUTPUT_DIR / "sigma_sp_site_summary.csv"

# ============================================================
# SETTINGS
# ============================================================
MIN_STATIONS = 3
DPI = 300

LAT_OFFSET = 0.03
ELEV_OFFSET = 25.0

era5_to_hrrr_group = {
    "ERA5_Missouri": "Missouri",
    "ERA5_California": "California",
}

hrrr_file_to_group = {
    "HRRR_Missouri.csv": "Missouri",
    "HRRR_California.csv": "California",
    "HRRR_Utah.csv": "Utah",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 1.2,
})


# ============================================================
# HELPERS
# ============================================================
def _norm(s: str) -> str:
    return re.sub(r"[()\s_\-]", "", str(s)).lower()


def find_col(df, candidates):
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    nm = {_norm(c): c for c in cols}

    for c in candidates:
        k = _norm(c)
        if k in nm:
            return nm[k]

    for c in candidates:
        k = _norm(c)
        for nk, real in nm.items():
            if k in nk:
                return real

    return None


def nice_limits(values, pad_ratio=0.08):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return (0, 1)

    vmin, vmax = np.min(v), np.max(v)

    if vmax == vmin:
        return (vmin - 1, vmax + 1)

    pad = (vmax - vmin) * pad_ratio
    return (vmin - pad, vmax + pad)


def gaussian_mu0(x, a, sigma):
    return a * np.exp(-(x ** 2) / (2 * sigma ** 2))


def try_parse_datetime(date_series, time_series=None):
    if time_series is not None:
        combo = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    else:
        combo = date_series.astype(str).str.strip()

    dt = pd.to_datetime(combo, errors="coerce")

    if dt.isna().all():
        dt = pd.to_datetime(combo, errors="coerce", format="mixed")

    return dt


def combine_datetime_columns(df, date_col, time_col):
    if date_col is None and time_col is None:
        raise ValueError("Both date and time columns are missing.")

    if date_col is not None and time_col is not None:
        return try_parse_datetime(df[date_col], df[time_col])

    if date_col is not None:
        return try_parse_datetime(df[date_col], None)

    return pd.to_datetime(df[time_col], errors="coerce", format="mixed")


def standardize_station_names(series):
    return series.astype(str).str.strip()


def fit_gaussian_mu0(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 3:
        return None, None

    p0 = [np.nanmax(y), max(np.nanstd(x), 1.0)]
    bounds = ([0.0, 1e-6], [np.inf, np.inf])

    try:
        popt, _ = curve_fit(
            gaussian_mu0,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )
        return popt[0], popt[1]
    except Exception as e:
        print("Gaussian fit failed:", e)
        return None, None


marker_map = {
    "California": "D",
    "Missouri": "s",
    "Utah": "P",
    "Alaska": "o",
    "Australia": "^",
    "Chile": "v",
    "Germany": "<",
    "Guadeloupe": ">",
    "Italy": "X",
    "Japan1": "h",
    "Japan2": "*",
}


def get_marker(row):
    if row["Dataset"] == "ERA5":
        if row["MatchedGroup"] != "None":
            return marker_map.get(row["MatchedGroup"], "o")
        return marker_map.get(row["SiteName"], "^")

    return marker_map.get(row["SiteName"], "o")


def get_lat_plot_x(row):
    x = float(row["Latitude"])

    if row["Dataset"] == "ERA5" and row["MatchedGroup"] == "California":
        return x - LAT_OFFSET
    if row["Dataset"] == "HRRR" and row["SiteName"] == "California":
        return x + LAT_OFFSET

    if row["Dataset"] == "ERA5" and row["MatchedGroup"] == "Missouri":
        return x - LAT_OFFSET
    if row["Dataset"] == "HRRR" and row["SiteName"] == "Missouri":
        return x + LAT_OFFSET

    return x


def get_elev_plot_x(row):
    x = float(row["Elevation"])

    if row["Dataset"] == "ERA5" and row["MatchedGroup"] == "California":
        return x - ELEV_OFFSET
    if row["Dataset"] == "HRRR" and row["SiteName"] == "California":
        return x + ELEV_OFFSET

    if row["Dataset"] == "ERA5" and row["MatchedGroup"] == "Missouri":
        return x - ELEV_OFFSET
    if row["Dataset"] == "HRRR" and row["SiteName"] == "Missouri":
        return x + ELEV_OFFSET

    return x


def plot_point(ax, row, xval, yval, color_val, cmap, norm):
    mk = get_marker(row)

    if row["Dataset"] == "ERA5":
        ax.scatter(
            xval,
            yval,
            c=[color_val],
            cmap=cmap,
            norm=norm,
            marker=mk,
            s=140,
            edgecolors="black",
            linewidths=1.0,
            zorder=3
        )
    else:
        edge_color = cmap(norm(color_val))

        ax.scatter(
            xval,
            yval,
            facecolors="none",
            edgecolors=[edge_color],
            marker=mk,
            s=160,
            linewidths=2.0,
            zorder=4
        )


def clean_site_name(fname):
    name = Path(fname).stem
    name = name.replace("ERA5_", "")
    return name


# ============================================================
# ERA5: COMPUTE sigma^sp
# ============================================================
era5_time_rows = []
era5_summary_rows = []

for fname in ERA5_FILES:
    fpath = DATA_DIR / fname

    if not fpath.exists():
        print(f"Missing ERA5 file: {fpath}")
        continue

    d = pd.read_csv(fpath)
    d.columns = d.columns.str.strip()

    resid_col = find_col(d, ["Residual(ERA5)", "Residual(ERA-5)", "Residual", "Residual_ERA5", "ERA5 Residual"])
    station_col = find_col(d, ["ID", "Name", "Station", "GNSS Station", "SiteStation"])
    lat_col = find_col(d, ["Lat", "Latitude", "LAT"])
    elev_col = find_col(d, ["Hgt_m", "Hgt", "Elevation", "Elev", "Height", "Height_m", "ELV", "Elevation_m"])
    date_col = find_col(d, ["Date"])
    time_col = find_col(d, ["Time", "times"])

    if resid_col is None:
        raise KeyError(f"{fname}: residual column not found. Columns: {d.columns.tolist()}")
    if station_col is None:
        raise KeyError(f"{fname}: station column not found. Columns: {d.columns.tolist()}")
    if lat_col is None:
        raise KeyError(f"{fname}: latitude column not found. Columns: {d.columns.tolist()}")
    if elev_col is None:
        raise KeyError(f"{fname}: elevation column not found. Columns: {d.columns.tolist()}")
    if date_col is None and time_col is None:
        raise KeyError(f"{fname}: no date/time columns found. Columns: {d.columns.tolist()}")

    d[station_col] = standardize_station_names(d[station_col])
    d[resid_col] = pd.to_numeric(d[resid_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[elev_col] = pd.to_numeric(d[elev_col], errors="coerce")
    d["DateTime"] = combine_datetime_columns(d, date_col, time_col)

    d = d.dropna(subset=[station_col, resid_col, lat_col, elev_col, "DateTime"]).copy()

    if d.empty:
        continue

    site_name = clean_site_name(fname)
    pair_group = era5_to_hrrr_group.get(Path(fname).stem, "None")

    station_meta = (
        d.groupby(station_col, as_index=False)
        .agg(
            Latitude=(lat_col, "mean"),
            Elevation=(elev_col, "mean")
        )
    )

    site_lat = station_meta["Latitude"].mean()
    site_elev = station_meta["Elevation"].mean()

    time_group = (
        d.groupby("DateTime")
        .agg(
            Sigma_sp=(
                resid_col,
                lambda x: np.std(x.dropna(), ddof=1)
                if x.dropna().shape[0] >= MIN_STATIONS
                else np.nan
            ),
            N_stations=(resid_col, lambda x: x.dropna().shape[0])
        )
        .reset_index()
    )

    time_group = time_group[time_group["N_stations"] >= MIN_STATIONS].copy()
    time_group = time_group.dropna(subset=["Sigma_sp"])

    if time_group.empty:
        print(f"WARNING: {site_name} had no valid time steps with at least {MIN_STATIONS} stations.")
        continue

    time_group["Dataset"] = "ERA5"
    time_group["SiteName"] = site_name
    time_group["MatchedGroup"] = pair_group
    time_group["Latitude"] = site_lat
    time_group["Elevation"] = site_elev
    time_group["Has_HRRR_Match"] = pair_group != "None"

    era5_time_rows.append(time_group)

    era5_summary_rows.append({
        "Dataset": "ERA5",
        "SiteName": site_name,
        "MatchedGroup": pair_group,
        "Has_HRRR_Match": pair_group != "None",
        "Latitude": site_lat,
        "Elevation": site_elev,
        "N_valid_times": len(time_group),
        "Mean_sigma_sp": time_group["Sigma_sp"].mean(),
        "Min_sigma_sp": time_group["Sigma_sp"].min(),
        "Max_sigma_sp": time_group["Sigma_sp"].max(),
    })

if not era5_time_rows:
    raise ValueError("No valid ERA5 sigma^sp results were created.")

era5_times = pd.concat(era5_time_rows, ignore_index=True)
era5_summary = pd.DataFrame(era5_summary_rows)


# ============================================================
# HRRR: COMPUTE sigma^sp
# ============================================================
hrrr_time_rows = []
hrrr_summary_rows = []

for fname in HRRR_FILES:
    fpath = DATA_DIR / fname

    if not fpath.exists():
        print(f"Missing HRRR file: {fpath}")
        continue

    d = pd.read_csv(fpath)
    d.columns = d.columns.str.strip()

    station_col = find_col(d, ["GNSS Station", "Station", "ID", "Name"])
    resid_col = find_col(d, ["Residual (GNSS)", "Residual(GNSS)", "Residual(HRRR)", "Residual", "Residual_HRRR", "HRRR Residual"])
    lat_col = find_col(d, ["Lat", "Latitude", "LAT"])
    elev_col = find_col(d, ["Hgt_m", "Hgt", "Elevation", "Elev", "Height", "Height_m", "Hgt_m(Raider)"])
    date_col = find_col(d, ["Date"])
    time_col = find_col(d, ["Time", "times"])

    if station_col is None:
        raise KeyError(f"{fname}: station column not found. Columns: {d.columns.tolist()}")
    if resid_col is None:
        raise KeyError(f"{fname}: residual column not found. Columns: {d.columns.tolist()}")
    if lat_col is None:
        raise KeyError(f"{fname}: latitude column not found. Columns: {d.columns.tolist()}")
    if elev_col is None:
        raise KeyError(f"{fname}: elevation column not found. Columns: {d.columns.tolist()}")
    if date_col is None and time_col is None:
        raise KeyError(f"{fname}: no date/time columns found. Columns: {d.columns.tolist()}")

    d[station_col] = standardize_station_names(d[station_col])
    d[resid_col] = pd.to_numeric(d[resid_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[elev_col] = pd.to_numeric(d[elev_col], errors="coerce")
    d["DateTime"] = combine_datetime_columns(d, date_col, time_col)

    d = d.dropna(subset=[station_col, resid_col, lat_col, elev_col, "DateTime"]).copy()

    if d.empty:
        continue

    region = hrrr_file_to_group.get(fname, Path(fname).stem.replace("HRRR_", ""))

    station_meta = (
        d.groupby(station_col, as_index=False)
        .agg(
            Latitude=(lat_col, "mean"),
            Elevation=(elev_col, "mean")
        )
    )

    region_lat = station_meta["Latitude"].mean()
    region_elev = station_meta["Elevation"].mean()

    time_group = (
        d.groupby("DateTime")
        .agg(
            Sigma_sp=(
                resid_col,
                lambda x: np.std(x.dropna(), ddof=1)
                if x.dropna().shape[0] >= MIN_STATIONS
                else np.nan
            ),
            N_stations=(resid_col, lambda x: x.dropna().shape[0])
        )
        .reset_index()
    )

    time_group = time_group[time_group["N_stations"] >= MIN_STATIONS].copy()
    time_group = time_group.dropna(subset=["Sigma_sp"])

    if time_group.empty:
        print(f"WARNING: HRRR {region} had no valid time steps with at least {MIN_STATIONS} stations.")
        continue

    time_group["Dataset"] = "HRRR"
    time_group["SiteName"] = region
    time_group["MatchedGroup"] = region
    time_group["Latitude"] = region_lat
    time_group["Elevation"] = region_elev
    time_group["Has_HRRR_Match"] = True

    hrrr_time_rows.append(time_group)

    hrrr_summary_rows.append({
        "Dataset": "HRRR",
        "SiteName": region,
        "MatchedGroup": region,
        "Has_HRRR_Match": True,
        "Latitude": region_lat,
        "Elevation": region_elev,
        "N_valid_times": len(time_group),
        "Mean_sigma_sp": time_group["Sigma_sp"].mean(),
        "Min_sigma_sp": time_group["Sigma_sp"].min(),
        "Max_sigma_sp": time_group["Sigma_sp"].max(),
    })

if not hrrr_time_rows:
    raise ValueError("No valid HRRR sigma^sp results were created.")

hrrr_times = pd.concat(hrrr_time_rows, ignore_index=True)
hrrr_summary = pd.DataFrame(hrrr_summary_rows)


# ============================================================
# COMBINED OUTPUTS
# ============================================================
all_times = pd.concat([era5_times, hrrr_times], ignore_index=True)
summary = pd.concat([era5_summary, hrrr_summary], ignore_index=True)

all_times.to_csv(OUT_CSV_ALL_TIMES, index=False)
summary.to_csv(OUT_CSV_SUMMARY, index=False)

print("\nSaved:")
print(" -", OUT_CSV_ALL_TIMES)
print(" -", OUT_CSV_SUMMARY)

print("\n" + "=" * 100)
print("SITE/GROUP SUMMARY FOR sigma^sp")
print("=" * 100)
print(summary.sort_values(["Dataset", "SiteName"]).to_string(index=False))


# ============================================================
# PLOT DATA
# ============================================================
plot_df = summary.copy()

if plot_df.empty:
    raise ValueError("No points remain for plotting.")

plot_df["Lat_plot"] = plot_df.apply(get_lat_plot_x, axis=1)
plot_df["Elev_plot"] = plot_df.apply(get_elev_plot_x, axis=1)


# ============================================================
# BUILD FIGURE
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Panel A
ax = axes[0]

ax.text(
    0.02,
    0.98,
    "(a)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold"
)

norm_a = Normalize(vmin=plot_df["Elevation"].min(), vmax=plot_df["Elevation"].max())
cmap_a = plt.colormaps["viridis"]

for _, row in plot_df.iterrows():
    plot_point(
        ax=ax,
        row=row,
        xval=row["Lat_plot"],
        yval=row["Mean_sigma_sp"],
        color_val=row["Elevation"],
        cmap=cmap_a,
        norm=norm_a
    )

x_lat = plot_df["Latitude"].to_numpy(float)
y_sig = plot_df["Mean_sigma_sp"].to_numpy(float)

a_fit, sigma_fit = fit_gaussian_mu0(x_lat, y_sig)

if a_fit is not None:
    xs = np.linspace(np.nanmin(x_lat) - 5, np.nanmax(x_lat) + 5, 1000)
    ys = gaussian_mu0(xs, a_fit, sigma_fit)
    ax.plot(xs, ys, "--", color="black", linewidth=2.0, zorder=2)

ax.set_xlabel("Latitude (°)", fontweight="bold")
ax.set_ylabel(r"$\sigma^{sp}$ (m)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_xlim(*nice_limits(plot_df["Lat_plot"]))
ax.set_ylim(*nice_limits(y_sig))

sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
sm_a.set_array([])
cbar_a = fig.colorbar(sm_a, ax=ax, fraction=0.046, pad=0.03)
cbar_a.set_label("Elevation (m)", fontweight="bold")

# Panel B
ax = axes[1]

ax.text(
    0.02,
    0.98,
    "(b)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold"
)

norm_b = Normalize(vmin=plot_df["Latitude"].min(), vmax=plot_df["Latitude"].max())
cmap_b = plt.colormaps["plasma"]

for _, row in plot_df.iterrows():
    plot_point(
        ax=ax,
        row=row,
        xval=row["Elev_plot"],
        yval=row["Mean_sigma_sp"],
        color_val=row["Latitude"],
        cmap=cmap_b,
        norm=norm_b
    )

era5_plot = plot_df[
    plot_df["Dataset"] == "ERA5"
].dropna(subset=["Elevation", "Mean_sigma_sp"])

hrrr_plot = plot_df[
    plot_df["Dataset"] == "HRRR"
].dropna(subset=["Elevation", "Mean_sigma_sp"])

if len(era5_plot) >= 2:
    X = era5_plot[["Elevation"]].to_numpy(float)
    y = era5_plot["Mean_sigma_sp"].to_numpy(float)

    mdl = LinearRegression().fit(X, y)
    xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    ax.plot(xs, mdl.predict(xs), "--", color="black", linewidth=2.0, zorder=2)

if len(hrrr_plot) >= 2:
    X = hrrr_plot[["Elevation"]].to_numpy(float)
    y = hrrr_plot["Mean_sigma_sp"].to_numpy(float)

    mdl = LinearRegression().fit(X, y)
    xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    ax.plot(xs, mdl.predict(xs), "-", color="black", linewidth=2.0, zorder=2)

ax.set_xlabel("Elevation (m)", fontweight="bold")
ax.set_ylabel(r"$\sigma^{sp}$ (m)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_xlim(*nice_limits(plot_df["Elev_plot"]))
ax.set_ylim(*nice_limits(plot_df["Mean_sigma_sp"]))

sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
sm_b.set_array([])
cbar_b = fig.colorbar(sm_b, ax=ax, fraction=0.046, pad=0.03)
cbar_b.set_label("Latitude (°)", fontweight="bold")


# ============================================================
# LEGEND
# ============================================================
legend_handles = [
    Line2D([0], [0], marker='D', linestyle='None', markerfacecolor='lightgray',
           markeredgecolor='black', markersize=9, label='California'),
    Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='lightgray',
           markeredgecolor='black', markersize=9, label='Missouri'),
    Line2D([0], [0], marker='P', linestyle='None', markerfacecolor='lightgray',
           markeredgecolor='black', markersize=9, label='Utah'),
    Line2D([0], [0], marker='^', linestyle='None', markerfacecolor='lightgray',
           markeredgecolor='black', markersize=9, label='Australia / unmatched ERA5'),
    Line2D([0], [0], marker='v', linestyle='None', markerfacecolor='lightgray',
           markeredgecolor='black', markersize=9, label='Chile'),
    Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='lightgray',
           markeredgecolor='black', markersize=9, label='ERA5 filled symbols'),
    Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none',
           markeredgecolor='black', markeredgewidth=2.0, markersize=9, label='HRRR open symbols'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2,
           label='Latitude fit / ERA5 elevation fit'),
    Line2D([0], [0], color='black', linestyle='-', linewidth=2,
           label='HRRR elevation fit'),
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
plt.show()

print(f"\nSaved figure to:\n{OUT_PNG}")
