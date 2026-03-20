"""Produces all charts and figures for the analysis.

Generates regime timeline plots, PCA biplots, asset return heatmaps by
regime, and equity curve comparisons. All figures are saved to outputs/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Colors chosen so each regime reads unambiguously at a glance.
REGIME_COLORS: dict[str, str] = {
    "Expansion":   "#2ecc71",  # green
    "Contraction": "#e74c3c",  # red
    "Stagflation": "#f39c12",  # amber/orange
    "Risk-off":    "#3498db",  # blue
}

# Historical events to annotate. Alternating top/bottom placement avoids
# label collisions for events that fall within a few months of each other.
EVENTS: list[dict] = [
    {"date": "2001-03", "label": "Dot-com recession",    "pos": "top"},
    {"date": "2001-09", "label": "9/11",                  "pos": "bottom"},
    {"date": "2007-12", "label": "Great Recession begins", "pos": "top"},
    {"date": "2008-09", "label": "Lehman collapse",        "pos": "bottom"},
    {"date": "2009-06", "label": "Recovery begins",        "pos": "top"},
    {"date": "2020-03", "label": "COVID crash",            "pos": "bottom"},
    {"date": "2022-03", "label": "Fed starts hiking",      "pos": "top"},
    {"date": "2022-06", "label": "Inflation peaks",        "pos": "bottom"},
]


def plot_regime_timeline(
    regime_df: pd.DataFrame,
    regime_col: str = "regime",
    save: bool = True,
) -> plt.Figure:
    """Creates a horizontal color-coded timeline of macro regimes over time.

    Each month is rendered as a filled vertical band colored by its regime.
    Historical event markers are overlaid as dashed vertical lines with
    rotated labels, alternating above and below the band to avoid collisions.

    Args:
        regime_df: DataFrame with a DatetimeIndex and a column of regime names.
        regime_col: Name of the column holding regime labels.
        save: If True, saves the figure to outputs/regime_timeline.png.

    Returns:
        The matplotlib Figure object for further customization or display.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 4))

    # ── Regime color bands ────────────────────────────────────────────────────
    # Iterate over each month and fill a one-month-wide vertical span.
    # Using approximate 31-day width per band at monthly frequency.
    dates = regime_df.index
    for i, date in enumerate(dates):
        regime = regime_df.loc[date, regime_col]
        color = REGIME_COLORS.get(regime, "#cccccc")

        # End of this band is the next date's start, or 31 days for the last month.
        band_end = dates[i + 1] if i < len(dates) - 1 else date + pd.DateOffset(months=1)
        ax.axvspan(date, band_end, facecolor=color, alpha=0.75, linewidth=0)

    # ── Event annotation lines ────────────────────────────────────────────────
    # y positions for top/bottom labels expressed in axes-fraction coordinates
    # so they stay fixed regardless of data range.
    label_y = {"top": 0.97, "bottom": 0.03}
    va_map   = {"top": "top", "bottom": "bottom"}

    chart_start = dates[0]
    chart_end   = dates[-1] + pd.DateOffset(months=1)

    for event in EVENTS:
        event_date = pd.Timestamp(event["date"])
        # Skip events that fall outside the rendered date range — they would
        # land off-chart and bleed into the axis spine area.
        if not (chart_start <= event_date <= chart_end):
            continue
        pos = event["pos"]

        ax.axvline(
            x=event_date,
            color="#444444",
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            zorder=5,
        )
        ax.text(
            x=event_date,
            y=label_y[pos],
            s=f"  {event['label']}",
            transform=ax.get_xaxis_transform(),  # x in data coords, y in axes fraction
            rotation=90,
            fontsize=8.5,
            color="#222222",
            va=va_map[pos],
            ha="left",
            zorder=6,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=color, alpha=0.75, label=regime)
        for regime, color in REGIME_COLORS.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # ── Chart chrome ──────────────────────────────────────────────────────────
    ax.set_xlim(dates[0], dates[-1] + pd.DateOffset(months=1))
    ax.set_ylim(0, 1)
    ax.set_yticks([])  # y-axis carries no meaning; suppress ticks and labels

    ax.set_xlabel("Date", fontsize=11, labelpad=8)
    ax.set_title("Macro Regime Timeline (2002–2024)", fontsize=14, fontweight="bold", pad=12)

    # Remove all spines except the bottom axis line for a clean look.
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")

    ax.tick_params(axis="x", labelsize=10, colors="#444444")

    fig.tight_layout()

    if save:
        out_path = OUTPUTS_DIR / "regime_timeline.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig
