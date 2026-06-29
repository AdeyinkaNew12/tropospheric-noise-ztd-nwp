import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# =========================
# SETTINGS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "Supplement_Original_Data"
FIGURE_DIR = BASE_DIR / "figures"

FIGURE_DIR.mkdir(exist_ok=True)

FILES = [
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

NROWS, NCOLS = 5, 2
FIGSIZE = (8.3, 11.7)
DPI = 300
OUTNAME = "figure1_era5_1to1_all_sites.png"

FONT_FAMILY = "DejaVu Sans"
AX_LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 6.5
LEGEND_TITLE_SIZE = 7
CBAR_LABEL_SIZE = 7
CBAR_TICK_SIZE = 6.5
MARKER_SIZE = 18
EDGE_LW = 0.5
ONE2ONE_LW = 1.0
GRID_ALPHA = 0.30
PANEL_LABEL_SIZE = 10

RESID_COL_STRICT = "Residual(ERA5)"


def _norm(s: str) -> str:
    return re.sub(r"[()\s_\-]", "", str(s)).lower()


def pick_col(df, candidates):
    norm_map = {_norm(c): c for c in df.columns}

    for cand in candidates:
        k = _norm(cand)
        if k in norm_map:
            return norm_map[k]

    for cand in candidates:
        k = _norm(cand)
        for nk, real in norm_map.items():
            if k in nk:
                return real

    return None


def load_site_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    id_col = pick_col(df, ["ID", "Station", "Site", "Name", "NAME"])

    x_col = pick_col(
        df,
        [
            "GNSS(ZTD)", "GNSS ZTD", "GNSS_ZTD", "GNSSZTD",
            "ZTD(GNSS)", "ZTD GNSS",
            "GPS ZTD", "GPS_ZTD", "GPS-ZTD", "GPSZTD",
            "ZTD",
        ],
    )

    c_col = RESID_COL_STRICT

    y_col = pick_col(
        df,
        [
            "ERA5 ZTD", "ERA-5 ZTD", "ERA 5 ZTD", "ERA5 ZTD (m)", "ERA ZTD",
            "TotalDelay(Raider)ERA5", "Total Delay(Raider)ERA5",
            "TotalDelay(ERA5)", "Total Delay(ERA5)",
            "TotalDelay", "Total Delay",
        ],
    )

    missing = [
        k for k, v in {"ID": id_col, "GNSS/GPS ZTD": x_col}.items()
        if v is None
    ]

    if missing:
        raise KeyError(
            f"Missing columns in {path.name}: {', '.join(missing)}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    if c_col not in df.columns:
        raise KeyError(
            f"Missing required residual column '{c_col}' in {path.name}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    df[id_col] = df[id_col].astype(str).str.strip().str.upper()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[c_col] = pd.to_numeric(df[c_col], errors="coerce")

    if y_col is None:
        df["ERA5_ZTD_RECON"] = df[x_col] - df[c_col]
        y_col = "ERA5_ZTD_RECON"
        print(f"Warning: {path.name}: ERA5 ZTD missing; reconstructed as GNSS - Residual(ERA5).")
    else:
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    dfc = df.dropna(subset=[id_col, x_col, y_col, c_col]).copy()
    return dfc, id_col, x_col, y_col, c_col


# =========================
# BUILD FIGURE
# =========================
plt.rcParams.update({"font.family": FONT_FAMILY, "font.weight": "bold"})

fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, constrained_layout=True)
axes = axes.flatten()

panel_letters = list("abcdefghijklmnopqrstuvwxyz")
markers = ["o", "s", "^", "D", "P", "*", "X", "v", "<", ">"]

for i, (ax, fname) in enumerate(zip(axes, FILES)):
    path = DATA_DIR / fname

    if not path.exists():
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"Missing:\n{fname}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        continue

    try:
        dfc, id_col, x_col, y_col, c_col = load_site_csv(path)
    except Exception as e:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"Error:\n{fname}\n{e}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
        )
        continue

    stations = sorted(dfc[id_col].unique())
    marker_map = {st: markers[j % len(markers)] for j, st in enumerate(stations)}

    vmin, vmax = float(dfc[c_col].min()), float(dfc[c_col].max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["viridis"]

    legend_handles = []
    first_scatter = None

    for st in stations:
        sub = dfc[dfc[id_col] == st]
        if sub.empty:
            continue

        sc = ax.scatter(
            sub[x_col],
            sub[y_col],
            c=sub[c_col],
            cmap=cmap,
            norm=norm,
            marker=marker_map[st],
            s=MARKER_SIZE,
            alpha=0.9,
            edgecolors="black",
            linewidth=EDGE_LW,
        )

        if first_scatter is None:
            first_scatter = sc

        mean_res = float(sub[c_col].mean())
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_map[st],
                color="w",
                markerfacecolor=cmap(norm(mean_res)),
                markeredgecolor="black",
                markersize=4.5,
                label=st,
            )
        )

    lo = float(min(dfc[x_col].min(), dfc[y_col].min()))
    hi = float(max(dfc[x_col].max(), dfc[y_col].max()))
    pad = 0.03 * (hi - lo) if hi > lo else 0.01

    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        color="black",
        linewidth=ONE2ONE_LW,
    )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(
        f"({panel_letters[i]})",
        loc="left",
        pad=2,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )

    ax.set_xlabel("GNSS ZTD (m)", fontsize=AX_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("ERA5 ZTD (m)", fontsize=AX_LABEL_SIZE, fontweight="bold")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)

    leg = ax.legend(
        handles=legend_handles
        + [Line2D([0], [0], ls="--", color="black", lw=ONE2ONE_LW, label="1:1 line")],
        title="Station",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    leg.get_title().set_fontweight("bold")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    if first_scatter is not None:
        cbar = fig.colorbar(first_scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Residual (m)", fontsize=CBAR_LABEL_SIZE, fontweight="bold")
        cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

        for tt in cbar.ax.get_yticklabels():
            tt.set_fontweight("bold")

for j in range(len(FILES), len(axes)):
    axes[j].set_axis_off()

out_png = FIGURE_DIR / OUTNAME
plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
plt.show()

print(f"Saved combined figure to:\n{out_png}")
