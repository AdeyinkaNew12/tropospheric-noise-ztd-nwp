import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path

# =========================
# SETTINGS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "Supplement_Original_Data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

MONTH_STYLE_KEYS = ["missouri", "california", "guadeloupe"]

NROWS, NCOLS = 5, 2
FIGSIZE = (16, 18)
DPI = 300
OUTNAME = "Residual_TimeSeries_ALL_SITES_ERA5_PRO.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

MARKERS = ["o", "s", "D", "^", "P", "X", "v", "<", ">", "*"]
MSIZE = 18
EDGE_LW = 0.35
ALPHA = 0.65


# =========================
# HELPERS
# =========================
def _norm(s: str) -> str:
    return re.sub(r"[()\s_\-]", "", str(s).lower())


def find_col(df, candidates):
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


def parse_time_to_timedelta(series: pd.Series) -> pd.TimedeltaIndex:
    s = series.astype(str).str.strip()
    has_colon = s.str.contains(":", na=False)

    td = pd.Series([pd.NaT] * len(s), index=s.index, dtype="timedelta64[ns]")

    if has_colon.any():
        s2 = s.copy().str.replace(r"^\s*\d+\s+days\s+", "", regex=True)
        td.loc[has_colon] = pd.to_timedelta(s2.loc[has_colon], errors="coerce")

    if (~has_colon).any():
        sec = pd.to_numeric(s.loc[~has_colon], errors="coerce")
        td.loc[~has_colon] = pd.to_timedelta(sec, unit="s", errors="coerce")

    return pd.to_timedelta(td)


def load_one_site(path: Path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    id_col = find_col(df, ["ID", "Station", "Site", "Name", "D", "Stn", "StationID"])

    resid_col = find_col(df, [
        "Residual(ERA-5)", "Residual(ERA5)", "Residual (ERA5) (m)",
        "Residual(ERA5) (m)", "Residual_ERA5", "Residual ERA5",
        "Residual-ERA5", "Residual- ERA5", "GNSS-ERA5", "GPS-ERA5",
        "Residual"
    ])

    date_col = find_col(df, ["Date"])
    time_col = find_col(df, ["Time", "times"])

    missing = [k for k, v in {
        "Station": id_col,
        "Residual": resid_col,
        "Date": date_col,
        "Time": time_col
    }.items() if v is None]

    if missing:
        raise KeyError(
            f"{path.name} missing: {', '.join(missing)} | columns={df.columns.tolist()}"
        )

    df[id_col] = df[id_col].astype(str).str.strip().str.upper()
    df[resid_col] = pd.to_numeric(df[resid_col], errors="coerce")
    date_str = df[date_col].astype(str).str.strip()

    if path.name == "ERA5_California.csv":
        df[date_col] = pd.to_datetime(
            date_str, format="%Y%m%d", errors="coerce"
        )
    else:
        df[date_col] = pd.to_datetime(
            date_str, errors="coerce"
        )

    td = parse_time_to_timedelta(df[time_col])
    df["Datetime"] = df[date_col] + td

    dfc = df.dropna(subset=["Datetime", resid_col, id_col]).copy()
    dfc = dfc[dfc[id_col] != "NAN"]
    dfc = dfc.sort_values("Datetime")

    return dfc, id_col, resid_col


def apply_year_style(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.get_offset_text().set_visible(False)

    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_ha("center")


def apply_month_style(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.get_offset_text().set_visible(False)

    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_ha("center")


def is_month_style(fname: str) -> bool:
    f = fname.lower()
    return any(k in f for k in MONTH_STYLE_KEYS)


# =========================
# CHECK DATA
# =========================
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"\nData folder not found:\n{DATA_DIR}\n\n"
        "Create this folder inside the repository:\n"
        "data/Supplement_Original_Data/\n\n"
        "Then place the ERA5 CSV files there."
    )


# =========================
# BUILD FIGURE
# =========================
fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
axes = axes.flatten()
letters = list("abcdefghijklmnopqrstuvwxyz")

fig.subplots_adjust(
    left=0.07,
    right=0.82,
    top=0.97,
    bottom=0.05,
    wspace=0.55,
    hspace=0.45
)

for i, fname in enumerate(FILES):
    ax = axes[i]
    path = DATA_DIR / fname

    if not path.exists():
        ax.set_axis_off()
        ax.text(
            0.5, 0.5,
            f"Missing:\n{fname}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )
        continue

    try:
        dfc, id_col, resid_col = load_one_site(path)
    except Exception as e:
        ax.set_axis_off()
        ax.text(
            0.5, 0.5,
            f"Error in:\n{fname}\n{e}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )
        continue

    stations = sorted(dfc[id_col].unique())

    colors = plt.cm.tab10.colors
    color_map = {st: colors[j % len(colors)] for j, st in enumerate(stations)}
    marker_map = {st: MARKERS[j % len(MARKERS)] for j, st in enumerate(stations)}

    for st in stations:
        sub = dfc[dfc[id_col] == st]

        ax.scatter(
            sub["Datetime"],
            sub[resid_col],
            s=MSIZE,
            marker=marker_map[st],
            facecolor=color_map[st],
            edgecolor="black",
            linewidth=EDGE_LW,
            alpha=ALPHA
        )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Residual (ERA-5) (m)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)

    if fname == "ERA5_California.csv":
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2023-01-01")
        )
        ax.tick_params(axis="x", labelrotation=0)
    elif is_month_style(fname):
        apply_month_style(ax)
    else:
        apply_year_style(ax)

    ax.text(
        0.0, 1.06,
        f"({letters[i]})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.15"
        )
    )

    handles = [
        Line2D(
            [0], [0],
            marker=marker_map[st],
            color="w",
            markerfacecolor=color_map[st],
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label=st
        )
        for st in stations
    ]

    in_left_col = (i % 2 == 0)
    anchor_x = 1.02 if in_left_col else 1.06

    leg = ax.legend(
        handles=handles,
        title="Station",
        loc="center left",
        bbox_to_anchor=(anchor_x, 0.5),
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.3
    )

    leg.get_title().set_fontweight("bold")

    for t in leg.get_texts():
        t.set_fontweight("bold")

for j in range(len(FILES), len(axes)):
    axes[j].set_axis_off()

out_path = OUTPUT_DIR / OUTNAME
fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
plt.close('all')

print(f"✅ Saved combined figure to:\n{out_path}")
