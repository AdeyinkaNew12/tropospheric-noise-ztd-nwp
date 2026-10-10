import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# =========================
# SETTINGS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "Supplement_Original_Data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    (DATA_DIR / "HRRR_Missouri.csv", "(a) Missouri", "normal"),
    (DATA_DIR / "HRRR_California.csv", "(b) California", "compressed"),
    (DATA_DIR / "HRRR_Utah.csv", "(c) Utah", "normal"),
]

OUTNAME = "Residual_TimeSeries_3Sites_VERTICAL_HRRR_CLEAN_CA_COMPRESSED_ONLY_SYMBOLS.png"

FIGSIZE = (11.5, 9.2)

POINT_SIZE = 18
ALPHA = 0.65
EDGE_LW = 0.35

FONT_FAMILY = "DejaVu Sans"
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
LEGEND_TITLE_SIZE = 11
PANEL_LABEL_SIZE = 14
GRID_ALPHA = 0.22

MARKERS = {
    "ROL1": "o", "ROL2": "s", "ROL3": "^", "ROL4": "D",
    "P665": "o", "P666": "s", "P667": "^", "P668": "D",
    "UTSL": "o", "UTSO": "s", "UTWV": "^", "ZLC1": "D",
}


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


def build_datetime(df):
    date_col = pick_col(df, ["Date"])
    time_col = pick_col(df, ["Time", "times"])
    datetime_col = pick_col(df, ["Datetime", "DateTime"])

    if date_col is not None and time_col is not None:
        return pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            errors="coerce"
        )

    if datetime_col is not None:
        return pd.to_datetime(df[datetime_col], errors="coerce")

    if date_col is not None:
        return pd.to_datetime(df[date_col], errors="coerce")

    raise KeyError("Date/Time or Datetime column not found.")


def load_residual_timeseries(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    id_col = pick_col(df, ["ID", "Station", "Name", "SITE"])

    res_col = pick_col(df, [
        "Residual(HRRR)", "Residual HRRR",
        "Residual (GNSS)", "Residual(GNSS)",
        "GNSS-HRRR", "GPS-HRRR",
        "Residual", "Residual(m)", "Residual (m)"
    ])

    if id_col is None:
        raise KeyError(f"Missing station column in {path.name}")

    if res_col is None:
        raise KeyError(f"Missing residual column in {path.name}")

    df["Datetime_plot"] = build_datetime(df)
    df[res_col] = pd.to_numeric(df[res_col], errors="coerce")
    df[id_col] = df[id_col].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["Datetime_plot", res_col, id_col]).copy()
    df = df.sort_values("Datetime_plot")

    unique_dates = sorted(df["Datetime_plot"].dt.normalize().unique())
    date_to_x = {d: i for i, d in enumerate(unique_dates)}
    df["x_compressed"] = df["Datetime_plot"].dt.normalize().map(date_to_x)

    print("\nLoaded:", path.name)
    print("Rows used:", len(df))
    print("Station column:", id_col)
    print("Residual column:", res_col)
    print("Stations:", sorted(df[id_col].unique()))

    return df, id_col, res_col, unique_dates


def clean_normal_date_axis(ax, dt_min, dt_max):
    span_days = (dt_max - dt_min).days
    pad = pd.Timedelta(days=max(2, int(span_days * 0.04)))

    ax.set_xlim(dt_min - pad, dt_max + pad)

    if span_days > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif span_days > 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(35)
        lbl.set_ha("right")
        lbl.set_fontweight("bold")


def clean_compressed_date_axis(ax, unique_dates):
    import numpy as np

    n = len(unique_dates)

    if n == 0:
        return

    ax.set_xlim(-1, n)

    # Six evenly spaced dates, including first and last
    tick_count = min(6, n)

    tick_positions = np.unique(
        np.linspace(0, n - 1, tick_count)
        .round()
        .astype(int)
    ).tolist()

    # Full YYYY-MM-DD format for ALL dates
    tick_labels = [
        pd.to_datetime(unique_dates[i]).strftime("%Y-%m")
        for i in tick_positions
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(35)
        lbl.set_ha("right")
        lbl.set_fontweight("bold")


fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=False)

fig.subplots_adjust(
    left=0.12,
    right=0.82,
    top=0.96,
    bottom=0.10,
    hspace=0.58
)

for ax, (path, tag, axis_type) in zip(axes, PANELS):

    if not path.exists():
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"Missing:\n{path.name}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )
        continue

    df, id_col, res_col, unique_dates = load_residual_timeseries(path)
    stations = sorted(df[id_col].unique())

    for st in stations:
        sub = df[df[id_col] == st]

        if axis_type == "compressed":
            x_values = sub["x_compressed"]
        else:
            x_values = sub["Datetime_plot"]

        ax.scatter(
            x_values,
            sub[res_col],
            s=POINT_SIZE,
            alpha=ALPHA,
            edgecolors="black",
            linewidths=EDGE_LW,
            marker=MARKERS.get(st, "o"),
            label=st
        )

    ax.set_title(tag, loc="left", fontsize=PANEL_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Residual (m)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    for t in ax.get_yticklabels():
        t.set_fontweight("bold")

    if axis_type == "compressed":
        clean_compressed_date_axis(ax, unique_dates)
    else:
        clean_normal_date_axis(
            ax,
            df["Datetime_plot"].min(),
            df["Datetime_plot"].max()
        )

    leg = ax.legend(
        title="Station",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        frameon=True
    )

    leg.get_title().set_fontweight("bold")

    for txt in leg.get_texts():
        txt.set_fontweight("bold")


out_png = OUTPUT_DIR / OUTNAME

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close('all')

print(f"\n✅ Saved to:\n{out_png}")
