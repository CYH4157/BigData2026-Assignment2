"""Question 3 figure: daily records per activity with the detected collection gaps shaded.

A gap = a run of >= 6 hours without any record of that activity (data.missing_hours).
Run from the repository root:  python src/q3_data_gaps.py   (writes outputs/figures/q3_data_gaps.{png,pdf})
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ACTS, activity_log, load_db, missing_hours  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
START = pd.Timestamp("2025-12-01")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LINE, GAP = "#2a78d6", "#e34948"
LABEL_HOURS = 48  # annotate gaps at least this long (push panel only)


def spans(miss, hours):
    """Contiguous True runs of a boolean hourly array -> list of (start, end, length in hours)."""
    out, i = [], 0
    while i < len(miss):
        if miss[i]:
            j = i
            while j < len(miss) and miss[j]:
                j += 1
            out.append((hours[i], hours[j - 1] + pd.Timedelta(hours=1), j - i))
            i = j
        else:
            i += 1
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = activity_log(load_db())
    log = log[log.t >= START]
    end = log.t.max().floor("h") + pd.Timedelta(hours=1)
    hours = pd.date_range(START, end, freq="h", inclusive="left")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(len(ACTS), 1, figsize=(14, 11), sharex=True, facecolor=SURFACE)
    for ax, a in zip(axes, ACTS):
        t = log.t[log.act == a]
        daily = t.dt.floor("D").value_counts().reindex(pd.date_range(START, end.floor("D"), freq="D"), fill_value=0)
        miss = missing_hours(t, START, end)
        gaps = spans(miss, hours)
        for s, e, n in gaps:
            ax.axvspan(s, e, color=GAP, alpha=0.22, lw=0)
        ax.plot(daily.index, daily.to_numpy(), color=LINE, lw=1.8)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, daily.max() * 1.25)
        share = miss.mean()
        ax.set_title(f"{a}   (daily records;  {share:.1%} of hours in gaps)", loc="left", fontsize=12,
                     color=INK, pad=4)
        if a == "push":
            for s, e, n in gaps:
                if n >= LABEL_HOURS:
                    ax.annotate(f"{n} h", xy=(s + (e - s) / 2, daily.max() * 1.08), ha="center", fontsize=9,
                                color=INK2)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Collection gaps in the data  (shaded: ≥ 6 consecutive hours with no record of that activity)",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.005, f"GH Archive data up to {log.seen.max():%Y-%m-%d %H:%M} UTC (course preprocess.py). In Jul-Aug the record counts of all non-push activities collapse, "
             "so short gaps there may be missing data or very low activity.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    for fmt in ["png", "pdf"]:
        fig.savefig(OUT / f"q3_data_gaps.{fmt}", dpi=200, facecolor=SURFACE)
    print(f"saved {OUT / 'q3_data_gaps'}.(png|pdf)")
    for a in ACTS:
        t = log.t[log.act == a]
        g = [x for x in spans(missing_hours(t, START, end), hours) if x[2] >= LABEL_HOURS]
        print(a, "gaps >= 24h:", [(f"{s:%m-%d %H:%M}", f"{e:%m-%d %H:%M}", n) for s, e, n in g])


if __name__ == "__main__":
    main()
