"""Produces all charts and figures for the analysis.

Generates regime timeline plots, PCA biplots, asset return heatmaps by
regime, and equity curve comparisons. All figures are saved to outputs/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import config

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


# Asset colors kept consistent across both new charts so the two figures
# share a visual vocabulary — same asset always reads as the same color.
ASSET_COLORS: dict[str, str] = {
    "SPY": "#4C72B0",  # slate blue
    "TLT": "#55A868",  # teal green
    "GLD": "#C49A00",  # gold
    "DBC": "#C44E52",  # terracotta
}

REGIME_ORDER = ["Expansion", "Contraction", "Stagflation", "Risk-off"]
TICKERS = list(config.ASSETS.keys())  # ["SPY", "TLT", "GLD", "DBC"]


def plot_return_distributions(
    regime_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_col: str = "regime",
    save: bool = True,
) -> plt.Figure:
    """Plots monthly return distributions per asset, faceted by regime.

    Renders a 2×2 grid of violin plots — one subplot per regime — with four
    violins per subplot (one per asset). Each subplot title is tinted in the
    regime's canonical color to visually link this chart to the timeline.

    Args:
        regime_df: DataFrame with a DatetimeIndex and a column of regime labels.
        returns_df: DataFrame with monthly asset returns (decimal).
        regime_col: Name of the column in regime_df holding regime labels.
        save: If True, saves to outputs/return_distributions.png.

    Returns:
        The matplotlib Figure object.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle(
        "Monthly Return Distributions by Regime",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # Align regime labels with returns on the shared date index.
    combined = regime_df[[regime_col]].join(returns_df[TICKERS], how="inner").dropna()

    positions = [1, 2, 3, 4]

    for ax, regime in zip(axes.flat, REGIME_ORDER):
        subset = combined[combined[regime_col] == regime]
        data = [subset[ticker].values for ticker in TICKERS]

        parts = ax.violinplot(
            data,
            positions=positions,
            showmedians=True,
            showextrema=True,
            widths=0.65,
        )

        # Color each violin body by its asset and style the stat lines.
        for body, ticker in zip(parts["bodies"], TICKERS):
            body.set_facecolor(ASSET_COLORS[ticker])
            body.set_edgecolor("#333333")
            body.set_linewidth(0.6)
            body.set_alpha(0.82)

        for partname in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[partname].set_edgecolor("#333333")
            parts[partname].set_linewidth(1.2)

        # Zero-return reference line so positive/negative regimes read clearly.
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)

        # Subplot title tinted in the regime's color to echo the timeline chart.
        ax.set_title(
            f"{regime}  (n={len(subset)})",
            fontsize=12,
            fontweight="bold",
            color=REGIME_COLORS[regime],
            pad=6,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(TICKERS, fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.tick_params(axis="y", labelsize=9)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.set_axisbelow(True)

    # Shared y-axis label on the left column only.
    for ax in axes[:, 0]:
        ax.set_ylabel("Monthly Return", fontsize=10)

    # Legend for asset colors placed outside the grid at the bottom.
    legend_patches = [
        mpatches.Patch(facecolor=color, label=ticker, edgecolor="#333333", linewidth=0.6)
        for ticker, color in ASSET_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()

    if save:
        out_path = OUTPUTS_DIR / "return_distributions.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


def plot_allocation_heatmap(save: bool = True) -> plt.Figure:
    """Renders a heatmap of regime allocation weights colored by risk-on tilt.

    Cells are colored by their risk-on contribution (weight × asset risk score)
    rather than raw weight, so warm colors signal risk-on positioning and cool
    colors signal defensive positioning. The actual allocation percentage is
    printed in every cell. The 60/40 baseline is shown as a muted bottom row
    for reference.

    Asset risk scores: SPY +1.0, DBC +0.5, GLD −0.5, TLT −1.0.

    Args:
        save: If True, saves to outputs/allocation_heatmap.png.

    Returns:
        The matplotlib Figure object.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    allocations = config.REGIME_ALLOCATIONS
    baseline    = config.BASELINE_ALLOCATIONS

    row_labels = REGIME_ORDER + ["60/40 Baseline"]

    # Build raw weight matrix — baseline row fills missing assets with 0.
    weights = np.zeros((len(row_labels), len(TICKERS)))
    for i, regime in enumerate(REGIME_ORDER):
        for j, ticker in enumerate(TICKERS):
            weights[i, j] = allocations[regime].get(ticker, 0.0)
    for j, ticker in enumerate(TICKERS):
        weights[-1, j] = baseline.get(ticker, 0.0)

    # Risk-on score: how much each asset contributes to portfolio risk appetite.
    # Equity = fully risk-on, commodities = mildly risk-on,
    # gold = mildly defensive, bonds = fully defensive.
    risk_scores = np.array([1.0, -1.0, -0.5, 0.5])  # SPY, TLT, GLD, DBC order
    risk_matrix = weights * risk_scores

    # Symmetric color scale so the midpoint (neutral) is always yellow.
    abs_max = np.abs(risk_matrix).max()

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        risk_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max,
    )

    # ── Cell text annotations ─────────────────────────────────────────────────
    for i in range(len(row_labels)):
        for j in range(len(TICKERS)):
            w = weights[i, j]
            label = f"{w:.0%}" if w > 0 else "—"

            # Choose black or white text based on cell luminance so it always
            # reads legibly against both warm and cool backgrounds.
            norm_val = (risk_matrix[i, j] + abs_max) / (2 * abs_max)
            cell_rgb = plt.cm.RdYlGn(norm_val)[:3]
            luminance = 0.299 * cell_rgb[0] + 0.587 * cell_rgb[1] + 0.114 * cell_rgb[2]
            text_color = "#111111" if luminance > 0.45 else "#f5f5f5"

            # Baseline row rendered at reduced opacity to read as reference.
            alpha = 0.65 if i == len(row_labels) - 1 else 1.0
            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=text_color, alpha=alpha,
            )

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xticks(range(len(TICKERS)))
    ax.set_xticklabels(TICKERS, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    ax.tick_params(length=0)  # suppress tick marks; the grid cell borders suffice

    # Dashed separator between regime rows and the baseline reference row.
    ax.axhline(len(REGIME_ORDER) - 0.5, color="#555555", linewidth=1.2, linestyle="--")

    ax.set_title(
        "Regime Allocation Weights  (color = risk-on tilt)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Colorbar as a legend for the risk-on scale.
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Risk-on contribution", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    if save:
        out_path = OUTPUTS_DIR / "allocation_heatmap.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig
