"""Question 3 figure: weekly positive rate of each activity (share of sample users who did it in the 7-day
label window), from outputs/analysis.json. Shows how much the label distribution drifts over time.

Run from the repository root after analysis.py:  python src/q3_drift.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from data import ACTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#e8871e", "#3f9e5a", "#8a5cc2", "#c94f7c", "#333333"]


def main():
    res = json.loads((ROOT / "outputs" / "analysis.json").read_text())
    rate = pd.DataFrame(res["weekly_positive_rate"]).T
    rate.index = pd.to_datetime(rate.index)
    valid = pd.to_datetime(res["valid_weeks"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(valid.min(), valid.max() + pd.Timedelta(days=7), color="#e4e3df", alpha=0.6, lw=0)
    ax.text(valid.min(), 0.89, " validation", fontsize=10, color=INK2, va="top")
    for a, c in zip(ACTS, COLORS):
        ax.plot(rate.index, rate[a], color=c, lw=2, label=a)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("positive rate (share of users)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(ncol=6, loc="upper left", bbox_to_anchor=(0, 1.12), frameon=False, fontsize=9)
    ax.set_title("Weekly positive rate per activity (label window starts each Sat 16:00 UTC)",
                 loc="left", fontsize=13, color=INK, pad=28)
    fig.text(0.01, 0.01, f"GH Archive data up to {res['data_end']} UTC; users with ≥ 5 records in the 90 days "
             "before each cutoff.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for fmt in ["png", "pdf"]:
        fig.savefig(OUT / f"q3_drift.{fmt}", dpi=200, facecolor=SURFACE)
    print(f"saved {OUT / 'q3_drift'}.(png|pdf)")


if __name__ == "__main__":
    main()
