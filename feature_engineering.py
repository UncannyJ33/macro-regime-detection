"""Transforms raw data into model-ready features.

Computes rolling z-scores and rates of change for each macro indicator,
plus yield-spread-specific signals. Returns a clean DataFrame with no NaN
values, ready for PCA and regime modeling.
"""

import numpy as np
import pandas as pd

import config

# Raw indicators to generate zscore + roc features for.
# yield_spread is included here and gets additional features appended separately.
_INDICATORS: list[str] = [
    "gdp_growth",
    "cpi",
    "unemployment",
    "yield_10y",
    "yield_2y",
    "fed_funds_rate",
    "yield_spread",
]


def engineer_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling z-scores, rates of change, and yield spread signals.

    For each indicator in the macro DataFrame, produces:
    - A rolling z-score: (value - rolling_mean) / rolling_std, using a window
      of config.ROLLING_WINDOW months. Captures how extreme the current reading
      is relative to recent history.
    - A month-over-month rate of change (pct_change). Captures momentum.

    For yield_spread specifically, also adds:
    - yield_spread_inverted: binary flag (1 if spread < 0) marking curve inversion.
    - yield_spread_acceleration: MoM % change of yield_spread_roc itself — the
      second derivative, capturing how quickly the spread is shifting.

    Rows without enough history for the rolling window are dropped, since z-scores
    computed on partial windows are meaningless as regime signals.

    Args:
        macro_df: Raw macro DataFrame from load_macro_data(), with a monthly
            DatetimeIndex and one column per indicator.

    Returns:
        DataFrame with 16 engineered feature columns and no NaN values.
        Index is a subset of macro_df's index (first ROLLING_WINDOW rows dropped).
    """
    features = pd.DataFrame(index=macro_df.index)

    for indicator in _INDICATORS:
        series = macro_df[indicator]

        # Rolling z-score: how many standard deviations from the rolling mean.
        # min_periods=config.ROLLING_WINDOW ensures we only compute on full windows.
        rolling = series.rolling(window=config.ROLLING_WINDOW, min_periods=config.ROLLING_WINDOW)
        rolling_std = rolling.std()
        z_score = (series - rolling.mean()) / rolling_std
        # A constant series within the window yields std=0, producing NaN via 0/0.
        # By convention, a z-score of 0 is correct: the value is exactly at its mean.
        features[f"{indicator}_zscore"] = z_score.where(rolling_std != 0, 0.0)

        # Month-over-month percentage change: captures directional momentum.
        features[f"{indicator}_roc"] = series.pct_change()

    # Yield spread extras — placed after the standard features for readability.
    features["yield_spread_inverted"] = (macro_df["yield_spread"] < 0).astype(int)

    # Acceleration = rate of change of the rate of change (second derivative).
    # Signals when the spread is shifting faster — useful at regime turning points.
    features["yield_spread_acceleration"] = features["yield_spread_roc"].pct_change()

    # Drop the leading rows that lack a full rolling window for z-score computation.
    # This also cleans up the NaN in the first row of roc columns.
    features = features.iloc[config.ROLLING_WINDOW:]

    # pct_change() on yield_spread produces ±inf when the spread crosses zero
    # (denominator near zero). Replace with NaN and forward-fill so the prior
    # valid momentum reading is carried forward rather than poisoning the scaler.
    inf_count = np.isinf(features.values).sum()
    if inf_count > 0:
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()

    # Confirm no NaNs survived — acceleration has one extra NaN from chaining
    # pct_change on roc, which iloc above should have removed. Validate explicitly.
    remaining_nans = features.isna().sum().sum()
    if remaining_nans > 0:
        raise ValueError(
            f"Feature DataFrame still contains {remaining_nans} NaN values after "
            "dropping the rolling window warm-up rows. Check indicator coverage."
        )

    print(
        f"Features engineered: {features.index[0].date()} to {features.index[-1].date()} "
        f"({len(features)} observations, {len(features.columns)} features)"
    )

    return features
