"""Orchestrates the full macro regime detection pipeline.

Runs each stage in sequence: data loading → feature engineering →
regime modeling → backtesting → visualization. Acts as the single
entry point for reproducing the full analysis.
"""

import numpy as np
import pandas as pd

import config
from backtester import compare_strategies
from data_loader import load_all_data
from feature_engineering import engineer_features
from regime_models import fit_hmm, fit_kmeans, fit_pca, label_regimes
from visualizations import (
    plot_allocation_heatmap,
    plot_regime_timeline,
    plot_return_distributions,
    plot_return_distributions_box,
)

# Seed numpy before any model calls so K-Means and HMM produce the same
# cluster/state assignments on every run, keeping config label mappings stable.
np.random.seed(config.RANDOM_SEED)


def _section(title: str) -> None:
    """Prints a formatted section header to mark pipeline progress."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _build_kmeans_labels(macro_df: pd.DataFrame, kmeans_result: pd.DataFrame) -> dict[int, str]:
    """Derives the K-Means integer→name mapping from cluster characteristics.

    K-Means cluster IDs are assigned arbitrarily, so we can't hardcode them.
    Instead we rank clusters by macro conditions and assign names heuristically:
      - Expansion:   highest GDP growth
      - Contraction: highest unemployment (and typically negative/near-zero GDP)
      - Stagflation: highest CPI among the remaining two clusters
      - Risk-off:    the leftover cluster

    Args:
        macro_df: Raw macro DataFrame used to compute per-cluster means.
        kmeans_result: DataFrame with a 'regime' column of integer cluster IDs.

    Returns:
        Mapping from cluster integer to regime name string.
    """
    labeled = macro_df.join(kmeans_result[["regime"]], how="inner")
    means = labeled.groupby("regime")[["gdp_growth", "cpi", "unemployment"]].mean()

    expansion_id   = int(means["gdp_growth"].idxmax())
    contraction_id = int(means["unemployment"].idxmax())
    # Guard against the rare case where both heuristics point to the same cluster.
    if contraction_id == expansion_id:
        contraction_id = int(means["unemployment"].nlargest(2).index[-1])

    remaining      = [i for i in means.index if i not in (expansion_id, contraction_id)]
    stagflation_id = int(means.loc[remaining, "cpi"].idxmax())
    risk_off_id    = [
        i for i in means.index
        if i not in (expansion_id, contraction_id, stagflation_id)
    ][0]

    return {
        expansion_id:   "Expansion",
        contraction_id: "Contraction",
        stagflation_id: "Stagflation",
        risk_off_id:    "Risk-off",
    }


def run_pipeline() -> None:
    """Executes every stage of the macro regime detection pipeline end to end."""

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _section("Loading data")
    macro_df, returns_df = load_all_data()

    # ── 2. Feature engineering ────────────────────────────────────────────────
    _section("Engineering features")
    features = engineer_features(macro_df)

    # ── 3. PCA ────────────────────────────────────────────────────────────────
    _section("PCA — dimensionality reduction")
    pc_df, pca, scaler = fit_pca(features)

    # ── 4. K-Means clustering ─────────────────────────────────────────────────
    _section("K-Means clustering")
    kmeans_result = fit_kmeans(pc_df)

    kmeans_mapping = _build_kmeans_labels(macro_df, kmeans_result)
    print(f"\nK-Means regime mapping: {kmeans_mapping}")
    kmeans_labeled = label_regimes(kmeans_result, kmeans_mapping, column="regime")

    # ── 5. HMM ────────────────────────────────────────────────────────────────
    _section("HMM — hidden Markov model (Viterbi decoding)")
    hmm_result, best_hmm = fit_hmm(pc_df)

    print(f"\nHMM regime mapping: {config.HMM_REGIME_LABELS}")
    hmm_labeled = label_regimes(hmm_result, config.HMM_REGIME_LABELS, column="hmm_regime")

    regime_counts = hmm_labeled["hmm_regime"].value_counts()
    print("\nMonths per HMM regime:")
    for regime, count in regime_counts.items():
        pct = count / len(hmm_labeled) * 100
        sparse_flag = "  *** sparse — fewer than 6 months ***" if count < 6 else ""
        print(f"  {regime:<14} {count:>4} months  ({pct:.1f}%){sparse_flag}")

    # ── 6. Backtest ───────────────────────────────────────────────────────────
    _section("Backtesting — regime strategy vs 60/40 baseline")
    # HMM labels feed the backtester: Viterbi decoding models regime persistence
    # and transition dynamics, making it more appropriate for sequential
    # portfolio allocation decisions than the static K-Means assignments.
    regime_returns, baseline_returns = compare_strategies(
        hmm_labeled, returns_df, regime_col="hmm_regime"
    )

    # ── 7. Visualizations ─────────────────────────────────────────────────────
    _section("Generating visualizations")

    # Timeline uses K-Means labels — K-Means produces cleaner, more contiguous
    # regime bands visually since it doesn't enforce temporal smoothness.
    plot_regime_timeline(kmeans_labeled, regime_col="regime")

    # Return distributions and heatmap use HMM labels to stay consistent
    # with the backtester's view of regimes.
    plot_return_distributions(hmm_labeled, returns_df, regime_col="hmm_regime")
    plot_return_distributions_box(hmm_labeled, returns_df, regime_col="hmm_regime")
    plot_allocation_heatmap()

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Pipeline complete. Outputs saved to outputs/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_pipeline()
