import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

# =========================
# SETTINGS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "Supplement_Original_Data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    DATA_DIR / "HRRR_Missouri.csv",
    DATA_DIR / "HRRR_California.csv",
    DATA_DIR / "HRRR_Utah.csv",
]

OUTNAME = "ZTD_1to1_Missouri_California_Utah_STACKED.png"

FIGSIZE = (9.2, 10.2)
FONT_FAMILY = "DejaVu Sans"
AX_LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 10
LEGEND_TITLE_SIZE = 11
CBAR_LABEL_SIZE = 11
CBAR_TICK_SIZE = 10

MARKER_SIZE = 55
EDGE_LW = 0.8
ONE2ONE_LW = 2.0
GRID_ALPHA = 0.28
PANEL_LABEL_SIZE = 16

MARKERS = ["o", "s", "^", "D", "P", "*", "X", "v", "<", ">"]
CMAP = plt.colormaps["viridis"]


def _norm(s):
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

    x_col = pick_col(df, [
        "GNSS(ZTD)", "GNSS ZTD", "GNSS_ZTD", "GNSSZTD",
        "ZTD(GNSS)", "ZTD GNSS",
        "GPS ZTD", "GPS_ZTD", "GPS-ZTD", "GNSS"
    ])

    y_col = pick_col(df, [
        "HRRR", "HRRR ZTD", "HRRR(ZTD)", "HRRRZTD",
        "TotalDelay(HRRR)", "TotalDelayHRRR",
        "TotalDelay(Raider)HRRR",
        "TotalDelay(Raider)",
        "totalDelay", "TotalDelay", "totaldelay",
        "Total Delay"
    ])

    c_col = pick_col(df, [
        "Residual(HRRR)", "Residual HRRR",
        "Residual (GNSS)", "Residual(GNSS)",
        "GNSS-HRRR", "GPS-HRRR",
        "Residual", "Residual(m)", "Residual (m)"
    ])

    missing = [
        k for k, v in {
            "ID": id_col,
            "GNSS": x_col,
            "HRRR": y_col,
            "Residual": c_col
        }.items()
        if v is None
    ]

    if missing:
        raise KeyError(
            f"Missing columns in {path.name}: {', '.join(missing)}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    df[id_col] = df[id_col].astype(str).str.strip().str.upper()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[c_col] = pd.to_numeric(df[c_col], errors="coerce")

    dfc = df.dropna(subset=[id_col, x_col, y_col, c_col]).copy()

    print("\nLoaded:", path.name)
    print("Rows used:", len(dfc))
    print("Station column:", id_col)
    print("GNSS column:", x_col)
    print("HRRR column:", y_col)
    print("Residual column:", c_col)
    print("Stations:", sorted(dfc[id_col].unique()))

    return dfc, id_col, x_col, y_col, c_col


if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data folder not found:\n{DATA_DIR}\n\n"
        "Expected folder: data/Supplement_Original_Data/"
    )


plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1.0, 0.045],
    height_ratios=[1.0, 1.0, 1.0]
)

panel_labels = ["(a) Missouri", "(b) California", "(c) Utah"]

for i, path in enumerate(FILES):
    ax = fig.add_subplot(gs[i, 0])
    cax = fig.add_subplot(gs[i, 1])

    if not path.exists():
        ax.set_axis_off()
        cax.set_axis_off()
        ax.text(
            0.5, 0.5,
            f"Missing:\n{path.name}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )
        continue

    dfc, id_col, x_col, y_col, c_col = load_site_csv(path)
    stations = sorted(dfc[id_col].unique())

    marker_map = {
        st: MARKERS[j % len(MARKERS)]
        for j, st in enumerate(stations)
    }

    vmin = dfc[c_col].min()
    vmax = dfc[c_col].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    legend_handles = []
    first_scatter = None

    for st in stations:
        sub = dfc[dfc[id_col] == st]

        sc = ax.scatter(
            sub[x_col],
            sub[y_col],
            c=sub[c_col],
            cmap=CMAP,
            norm=norm,
            marker=marker_map[st],
            s=MARKER_SIZE,
            alpha=0.90,
            edgecolors="black",
            linewidth=EDGE_LW
        )

        if first_scatter is None:
            first_scatter = sc

        mean_res = sub[c_col].mean()

        legend_handles.append(
            Line2D(
                [0], [0],
                marker=marker_map[st],
                color="w",
                markerfacecolor=CMAP(norm(mean_res)),
                markeredgecolor="black",
                markersize=8,
                label=st
            )
        )

    lo = min(dfc[x_col].min(), dfc[y_col].min())
    hi = max(dfc[x_col].max(), dfc[y_col].max())
    pad = 0.06 * (hi - lo) if hi > lo else 0.02

    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        color="black",
        linewidth=ONE2ONE_LW
    )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(
        panel_labels[i],
        loc="left",
        pad=6,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel("GNSS (m)", fontsize=AX_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("HRRR (m)", fontsize=AX_LABEL_SIZE, fontweight="bold")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)

    legend_handles.append(
        Line2D(
            [0], [0],
            ls="--",
            color="black",
            lw=ONE2ONE_LW,
            label="1:1 line"
        )
    )

    leg = ax.legend(
        handles=legend_handles,
        title="Station",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.4,
        handletextpad=0.6
    )

    leg.get_title().set_fontweight("bold")

    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    if first_scatter is not None:
        cb = fig.colorbar(first_scatter, cax=cax)

        cb.set_label(
            "Residual (m)",
            fontsize=CBAR_LABEL_SIZE,
            fontweight="bold",
            labelpad=8
        )

        cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)

        for tt in cb.ax.get_yticklabels():
            tt.set_fontweight("bold")

        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


out_png = OUTPUT_DIR / OUTNAME

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close('all')

print(f"✅ Saved figure to:\n{out_png}")
