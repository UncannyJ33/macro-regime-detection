"""Simulates regime-conditional asset allocation strategies.

Takes regime labels and asset returns, applies allocation rules that shift
weights based on the current regime, and computes portfolio performance
metrics (CAGR, Sharpe, max drawdown) versus a buy-and-hold benchmark.
"""

import numpy as np
import pandas as pd

import config


def run_backtest(
    regime_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    regime_col: str = "regime",
) -> pd.Series:
    """Simulates a regime-conditional portfolio month by month.

    For each month, looks up the current regime label, applies the
    corresponding allocation weights, and computes the weighted return.
    Months where any allocated asset has missing returns are dropped —
    this typically means the backtest starts when DBC data begins (~2006).

    Args:
        regime_df: DataFrame with a column of string regime labels.
        returns_df: DataFrame with monthly asset returns (decimal, e.g. 0.03).
        allocations: Mapping from regime name to {ticker: weight} dict.
        regime_col: Name of the column in regime_df holding regime labels.

    Returns:
        Monthly portfolio return series indexed by date, named "regime_strategy".
    """
    tickers = list(next(iter(allocations.values())).keys())
    combined = regime_df[[regime_col]].join(returns_df[tickers], how="inner").dropna()

    # Build a weight matrix: each row gets weights corresponding to its regime.
    weight_df = pd.DataFrame(
        [allocations[regime] for regime in combined[regime_col]],
        index=combined.index,
        columns=tickers,
    )
    port_returns = (combined[tickers] * weight_df).sum(axis=1)
    return port_returns.rename("regime_strategy")


def run_baseline(returns_df: pd.DataFrame) -> pd.Series:
    """Simulates a static 60/40 (SPY/TLT) portfolio for the same period.

    Uses BASELINE_ALLOCATIONS from config. Months with any NaN in the
    allocated assets are dropped so the period aligns cleanly with run_backtest.

    Args:
        returns_df: DataFrame with monthly asset returns.

    Returns:
        Monthly portfolio return series indexed by date, named "baseline_60_40".
    """
    weights = config.BASELINE_ALLOCATIONS
    tickers = list(weights.keys())
    asset_returns = returns_df[tickers].dropna()

    # Weighted sum across columns for every row at once — faster than iterrows.
    weight_series = pd.Series(weights)
    port_returns = asset_returns.mul(weight_series).sum(axis=1)
    return port_returns.rename("baseline_60_40")


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    """Computes standard portfolio performance metrics from a monthly return series.

    Args:
        returns: Monthly portfolio returns (decimal, not percent).

    Returns:
        Dict with keys: Total Return, Annualized Return, Annualized Volatility,
        Sharpe Ratio, Max Drawdown, Calmar Ratio.
    """
    n_months = len(returns)

    total_return = (1 + returns).prod() - 1

    # Compound annualized growth rate from the total return over n_months.
    ann_return = (1 + total_return) ** (12 / n_months) - 1

    # Monthly std × √12 converts to annualized volatility.
    ann_vol = returns.std() * np.sqrt(12)

    # Sharpe: annualized excess return divided by annualized volatility.
    monthly_rf = config.RISK_FREE_RATE / 12
    excess_monthly = returns.mean() - monthly_rf
    sharpe = (excess_monthly * 12) / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown: largest peak-to-trough decline in the cumulative return curve.
    cum_returns = (1 + returns).cumprod()
    rolling_peak = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_peak) / rolling_peak
    max_drawdown = drawdowns.min()  # Negative value; worst single trough

    # Calmar: annualized return relative to worst drawdown depth.
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        "Total Return":          total_return,
        "Annualized Return":     ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio":          sharpe,
        "Max Drawdown":          max_drawdown,
        "Calmar Ratio":          calmar,
    }


def compare_strategies(
    regime_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_col: str = "regime",
) -> tuple[pd.Series, pd.Series]:
    """Runs regime strategy and 60/40 baseline, prints a side-by-side metrics table.

    Aligns both strategies to the same date range before computing metrics so
    comparisons are apples-to-apples. Returns both return series for downstream
    use (e.g. plotting equity curves).

    Args:
        regime_df: DataFrame with a column of string regime labels.
        returns_df: DataFrame with monthly asset returns.
        regime_col: Name of the column in regime_df holding regime labels.

    Returns:
        Tuple of (regime_returns, baseline_returns) as monthly pd.Series.
    """
    allocations = config.REGIME_ALLOCATIONS
    regime_returns = run_backtest(regime_df, returns_df, allocations, regime_col)
    baseline_returns = run_baseline(returns_df)

    # Restrict both to their shared period so metrics reflect identical windows.
    common_idx = regime_returns.index.intersection(baseline_returns.index)
    regime_returns = regime_returns.loc[common_idx]
    baseline_returns = baseline_returns.loc[common_idx]

    regime_metrics = compute_metrics(regime_returns)
    baseline_metrics = compute_metrics(baseline_returns)

    # ── Print allocation rules ─────────────────────────────────────────────────
    print("\nRegime Allocation Rules")
    print("=" * 55)
    print(f"{'Regime':<14} {'SPY':>7} {'TLT':>7} {'GLD':>7} {'DBC':>7}")
    print("-" * 55)
    for regime, weights in allocations.items():
        print(
            f"{regime:<14} "
            f"{weights['SPY']:>6.0%} "
            f"{weights['TLT']:>7.0%} "
            f"{weights['GLD']:>7.0%} "
            f"{weights['DBC']:>7.0%}"
        )
    print("-" * 55)
    baseline = config.BASELINE_ALLOCATIONS
    print(
        f"{'60/40 Baseline':<14} "
        f"{baseline['SPY']:>6.0%} "
        f"{baseline['TLT']:>7.0%} "
        f"{'—':>7} "
        f"{'—':>7}"
    )

    # ── Print metrics comparison ───────────────────────────────────────────────
    period_start = common_idx[0].strftime("%Y-%m")
    period_end = common_idx[-1].strftime("%Y-%m")
    print(f"\nStrategy Comparison  ({period_start} → {period_end}, {len(common_idx)} months)")
    print("=" * 58)
    print(f"{'Metric':<25} {'Regime Strategy':>16} {'60/40 Baseline':>15}")
    print("-" * 58)

    pct_metrics = {"Total Return", "Annualized Return", "Annualized Volatility", "Max Drawdown"}
    for metric in regime_metrics:
        r_val = regime_metrics[metric]
        b_val = baseline_metrics[metric]
        if metric in pct_metrics:
            r_str = f"{r_val:.2%}"
            b_str = f"{b_val:.2%}"
        else:
            r_str = f"{r_val:.2f}"
            b_str = f"{b_val:.2f}"
        print(f"{metric:<25} {r_str:>16} {b_str:>15}")

    print("=" * 58)

    return regime_returns, baseline_returns
