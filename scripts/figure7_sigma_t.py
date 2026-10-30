import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

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

OUT_PNG = OUTPUT_DIR / "Figure7_sigma_t_station_only_OPEN_HRRR_FILLED_ERA5.png"
ALL_OUT = OUTPUT_DIR / "Figure7_sigma_t_station_only_outputs.csv"
STATIONS_USED_OUT = OUTPUT_DIR / "Figure7_stations_used.csv"

era5_to_hrrr_group = {
    "ERA5_Missouri": "Missouri",
    "ERA5_California": "California",
}

region_lookup = {
    "HRRR_Missouri.csv": "Missouri",
    "HRRR_California.csv": "California",
    "HRRR_Utah.csv": "Utah",
}

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


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


def std_sample(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()

    if len(x) < 2:
        return np.nan

    return np.std(x, ddof=1)


group_marker_map = {
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


def marker_for_site(site_name, matched_group):
    if matched_group not in [None, "None"]:
        return group_marker_map.get(matched_group, "o")
    return group_marker_map.get(site_name, "o")


def marker_for_region(region):
    return group_marker_map.get(region, "o")


def plot_station(ax, row, xval, yval, color_val, cmap, norm):
    if row["Dataset"] == "ERA5":
        mk = marker_for_site(row["Name"], row["MatchedGroup"])
        ax.scatter(
            xval,
            yval,
            c=[color_val],
            cmap=cmap,
            norm=norm,
            marker=mk,
            s=110,
            edgecolors="black",
            linewidths=0.9,
            alpha=0.9,
            zorder=3,
        )
    else:
        mk = marker_for_region(row["Name"])
        edge_color = cmap(norm(color_val))
        ax.scatter(
            xval,
            yval,
            facecolors="none",
            edgecolors=[edge_color],
            marker=mk,
            s=130,
            linewidths=2.0,
            alpha=1.0,
            zorder=4,
        )


def clean_site_name(fname):
    return Path(fname).stem.replace("ERA5_", "")


# ============================================================
# ERA5: sigma^t per station
# ============================================================
era5_station_rows = []

for fname in ERA5_FILES:
    fpath = DATA_DIR / fname

    if not fpath.exists():
        print("Missing ERA5 file:", fpath)
        continue

    d = pd.read_csv(fpath)
    d.columns = d.columns.str.strip()

    resid_col = find_col(d, ["Residual(ERA5)", "Residual(ERA-5)", "Residual", "Residual_ERA5", "ERA5 Residual"])
    station_col = find_col(d, ["ID", "Name", "Station", "GNSS Station", "SiteStation"])
    lat_col = find_col(d, ["Lat", "Latitude", "LAT"])
    elev_col = find_col(d, ["Hgt_m", "Hgt", "Elevation", "Elev", "Height", "Height_m", "ELV", "Elevation_m"])

    if resid_col is None:
        raise KeyError(f"{fname}: residual column not found. Found columns: {d.columns.tolist()}")
    if station_col is None:
        raise KeyError(f"{fname}: station column not found. Found columns: {d.columns.tolist()}")
    if lat_col is None or elev_col is None:
        raise KeyError(f"{fname}: latitude/elevation column not found. Found columns: {d.columns.tolist()}")

    d[station_col] = d[station_col].astype(str).str.strip()
    d[resid_col] = pd.to_numeric(d[resid_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[elev_col] = pd.to_numeric(d[elev_col], errors="coerce")

    d = d.dropna(subset=[station_col, resid_col, lat_col, elev_col]).copy()

    if d.empty:
        continue

    site_name = clean_site_name(fname)

    g = (
        d.groupby(station_col, as_index=False)
        .agg(
            Sigma_t=(resid_col, std_sample),
            N_times=(resid_col, lambda x: pd.Series(x).notna().sum()),
            Latitude=(lat_col, "mean"),
            Elevation=(elev_col, "mean"),
        )
    )

    g = g.dropna(subset=["Sigma_t"]).copy()

    if g.empty:
        continue

    g["Dataset"] = "ERA5"
    g["Name"] = site_name
    g["Station"] = g[station_col].astype(str).str.strip()
    g["MatchedGroup"] = era5_to_hrrr_group.get(Path(fname).stem, "None")
    g["Has_HRRR_Match"] = g["MatchedGroup"] != "None"

    era5_station_rows.append(
        g[
            [
                "Dataset",
                "Name",
                "Station",
                "Sigma_t",
                "N_times",
                "Latitude",
                "Elevation",
                "MatchedGroup",
                "Has_HRRR_Match",
            ]
        ]
    )

if not era5_station_rows:
    raise ValueError("No ERA5 sigma^t station data could be created.")

era5_station = pd.concat(era5_station_rows, ignore_index=True)


# ============================================================
# HRRR: sigma^t per station
# ============================================================
hrrr_station_rows = []

for fname in HRRR_FILES:
    fpath = DATA_DIR / fname

    if not fpath.exists():
        print("Missing HRRR file:", fpath)
        continue

    d = pd.read_csv(fpath)
    d.columns = d.columns.str.strip()

    station_col = find_col(d, ["ID", "Name", "Station", "GNSS Station"])
    resid_col = find_col(d, ["Residual (GNSS)", "Residual(GNSS)", "Residual(HRRR)", "Residual", "GNSS Residual"])
    lat_col = find_col(d, ["Latitude", "Lat", "LAT"])
    elev_col = find_col(d, ["Elevation", "Hgt_m(Raider)", "Hgt_m", "Hgt", "Elev"])

    if station_col is None:
        raise KeyError(f"{fname}: station column not found. Found columns: {d.columns.tolist()}")
    if resid_col is None:
        raise KeyError(f"{fname}: residual column not found. Found columns: {d.columns.tolist()}")
    if lat_col is None or elev_col is None:
        raise KeyError(f"{fname}: latitude/elevation column not found. Found columns: {d.columns.tolist()}")

    d[station_col] = d[station_col].astype(str).str.strip()
    d[resid_col] = pd.to_numeric(d[resid_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[elev_col] = pd.to_numeric(d[elev_col], errors="coerce")

    d = d.dropna(subset=[station_col, resid_col, lat_col, elev_col]).copy()

    if d.empty:
        continue

    region_name = region_lookup.get(fname, Path(fname).stem.replace("HRRR_", ""))

    g = (
        d.groupby(station_col, as_index=False)
        .agg(
            Sigma_t=(resid_col, std_sample),
            N_times=(resid_col, lambda x: pd.Series(x).notna().sum()),
            Latitude=(lat_col, "mean"),
            Elevation=(elev_col, "mean"),
        )
    )

    g = g.dropna(subset=["Sigma_t"]).copy()

    if g.empty:
        continue

    g["Dataset"] = "HRRR"
    g["Name"] = region_name
    g["Station"] = g[station_col].astype(str).str.strip()
    g["MatchedGroup"] = region_name
    g["Has_HRRR_Match"] = True

    hrrr_station_rows.append(
        g[
            [
                "Dataset",
                "Name",
                "Station",
                "Sigma_t",
                "N_times",
                "Latitude",
                "Elevation",
                "MatchedGroup",
                "Has_HRRR_Match",
            ]
        ]
    )

if not hrrr_station_rows:
    raise ValueError("No HRRR sigma^t station data could be created.")

hrrr_station = pd.concat(hrrr_station_rows, ignore_index=True)


# ============================================================
# COMBINE, SAVE, AND PRINT ALL STATIONS USED
# ============================================================
combined_station_out = pd.concat([era5_station, hrrr_station], ignore_index=True)

combined_station_out = combined_station_out.sort_values(
    ["Dataset", "Name", "Station"]
).reset_index(drop=True)

combined_station_out.to_csv(ALL_OUT, index=False)
combined_station_out.to_csv(STATIONS_USED_OUT, index=False)

print("\nSaved station-level output file:")
print(ALL_OUT)

print("\nSaved stations-used file:")
print(STATIONS_USED_OUT)

print("\nERA5 station rows:", len(era5_station))
print("HRRR station rows:", len(hrrr_station))
print("Total stations used:", len(combined_station_out))

print("\n" + "=" * 100)
print("ALL STATIONS USED IN FIGURE 6")
print("=" * 100)
print(combined_station_out.to_string(index=False))

# ============================================================
# FIGURE 7: STATION-LEVEL SIGMA_T DATA EXPORT
# Latitude and elevation plots removed as requested.
# ============================================================

print("\nFigure 7 latitude and elevation panels have been removed.")
print("Station-level sigma_t results remain available in:")
print(ALL_OUT)
print(STATIONS_USED_OUT)

def plot_panel_a(ax):
    elev_all = combined_station_out["Elevation"].to_numpy(float)
    norm = Normalize(vmin=np.nanmin(elev_all), vmax=np.nanmax(elev_all))
    cmap = plt.colormaps["viridis"]

    for _, row in combined_station_out.iterrows():
        plot_station(
            ax=ax,
            row=row,
            xval=row["Latitude"],
            yval=row["Sigma_t"],
            color_val=row["Elevation"],
            cmap=cmap,
            norm=norm,
        )

    x_comb = combined_station_out["Latitude"].to_numpy(float)
    y_comb = combined_station_out["Sigma_t"].to_numpy(float)

    good = np.isfinite(x_comb) & np.isfinite(y_comb)
    x_comb = x_comb[good]
    y_comb = y_comb[good]

    if len(x_comb) >= 3:
        p0 = [np.nanmax(y_comb), max(np.nanstd(x_comb), 1.0)]

        try:
            popt, _ = curve_fit(
                gaussian_mu0,
                x_comb,
                y_comb,
                p0=p0,
                bounds=([0.0, 1e-6], [np.inf, np.inf]),
                maxfev=20000,
            )

            xs = np.linspace(np.nanmin(x_comb) - 5, np.nanmax(x_comb) + 5, 1000)
            ys = gaussian_mu0(xs, *popt)

# FIT LINE REMOVED: ax.plot(xs, ys, linestyle="--", color="black", linewidth=2.0, zorder=2)

            print("\nPanel A station-level latitude Gaussian fit:")
            print(f"a = {popt[0]:.6f}")
            print(f"sigma = {popt[1]:.6f}")
        except Exception as e:
            print("Panel A Gaussian fit failed:", e)

    ax.set_xlabel("Latitude (°)", fontweight="bold")
    ax.set_ylabel(r"$\sigma^t$ (m)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(*nice_limits(x_comb))
    ax.set_ylim(*nice_limits(y_comb))

    return norm, cmap


# ============================================================
# PANEL B
# ============================================================
def plot_panel_b(ax):
    lat_all = combined_station_out["Latitude"].to_numpy(float)
    norm = Normalize(vmin=np.nanmin(lat_all), vmax=np.nanmax(lat_all))
    cmap = plt.colormaps["plasma"]

    for _, row in combined_station_out.iterrows():
        plot_station(
            ax=ax,
            row=row,
            xval=row["Elevation"],
            yval=row["Sigma_t"],
            color_val=row["Latitude"],
            cmap=cmap,
            norm=norm,
        )

    se = combined_station_out[
        combined_station_out["Dataset"] == "ERA5"
    ].dropna(subset=["Elevation", "Sigma_t"])

    if len(se) >= 2:
        X = se[["Elevation"]].to_numpy(float)
        y = se["Sigma_t"].to_numpy(float)

        mdl = LinearRegression().fit(X, y)
        xs = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

# FIT LINE REMOVED: ax.plot(xs, mdl.predict(xs), "--", color="black", linewidth=2.0, zorder=2)

        print("\nPanel B station-level ERA5 elevation fit:")
        print(f"slope = {mdl.coef_[0]:.8f}")
        print(f"intercept = {mdl.intercept_:.8f}")
        print(f"R2 = {mdl.score(X, y):.6f}")

    sh = combined_station_out[
        combined_station_out["Dataset"] == "HRRR"
    ].dropna(subset=["Elevation", "Sigma_t"])

    if len(sh) >= 2:
        X = sh[["Elevation"]].to_numpy(float)
        y = sh["Sigma_t"].to_numpy(float)

        mdl = LinearRegression().fit(X, y)
        xs = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

# FIT LINE REMOVED: ax.plot(xs, mdl.predict(xs), "-", color="black", linewidth=2.0, zorder=2)

        print("\nPanel B station-level HRRR elevation fit:")
        print(f"slope = {mdl.coef_[0]:.8f}")
        print(f"intercept = {mdl.intercept_:.8f}")
        print(f"R2 = {mdl.score(X, y):.6f}")

    x_all = combined_station_out["Elevation"].to_numpy(float)
    y_all = combined_station_out["Sigma_t"].to_numpy(float)

    ax.set_xlabel("Elevation (m)", fontweight="bold")
    ax.set_ylabel(r"$\sigma^t$ (m)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(*nice_limits(x_all))
    ax.set_ylim(*nice_limits(y_all))

    return norm, cmap


# ============================================================
# BUILD FIGURE
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

axes[0].text(
    0.02,
    0.98,
    "(a)",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold",
)

norm_a, cmap_a = plot_panel_a(axes[0])

axes[1].text(
    0.02,
    0.98,
    "(b)",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold",
)

norm_b, cmap_b = plot_panel_b(axes[1])

sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
sm_a.set_array([])
cbar_a = fig.colorbar(sm_a, ax=axes[0], fraction=0.046, pad=0.03)
cbar_a.set_label("Elevation (m)", fontweight="bold")

sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
sm_b.set_array([])
cbar_b = fig.colorbar(sm_b, ax=axes[1], fraction=0.046, pad=0.03)
cbar_b.set_label("Latitude (°)", fontweight="bold")


# Station-group markers.
site_labels = [('Alaska', 'o'), ('Australia', '^'), ('California', 'D'), ('Chile', 'v'), ('Germany', '<'), ('Guadeloupe', '>'), ('Italy', 'X'), ('Japan-1', 'h'), ('Japan-2', '*'), ('Missouri', 's'), ('Utah', 'P')]

site_handles = [
    Line2D([0], [0], marker=marker, linestyle="None",
           markerfacecolor="lightgray", markeredgecolor="black",
           markersize=8, label=label)
    for label, marker in site_labels
]

# Filled/open symbol convention; no fitted-line legend entries.
style_handles = [
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor="black", markeredgecolor="black",
           markersize=8, label="ERA5 (filled symbols)"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor="none", markeredgecolor="black",
           markeredgewidth=1.8, markersize=8,
           label="HRRR (open symbols)"),
]

legend_sites = fig.legend(
    handles=site_handles, loc="upper center",
    bbox_to_anchor=(0.5, 0.995), ncol=6,
    frameon=True, fontsize=9, title="Station group"
)
fig.add_artist(legend_sites)

fig.legend(
    handles=style_handles, loc="upper center",
    bbox_to_anchor=(0.5, 0.925), ncol=2,
    frameon=True, fontsize=9
)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close('all')

print(f"\nSaved figure to:\n{OUT_PNG}")
