"""Fetches and aligns raw macroeconomic and asset price data.

Pulls FRED macro series and yfinance price data, resamples everything
to a common monthly frequency, and returns a single merged DataFrame
ready for feature engineering.
"""

import pandas as pd
from fredapi import Fred

import config

# Maps the readable keys in config.FRED_SERIES to snake_case column names.
_COLUMN_NAMES: dict[str, str] = {
    "GDP Growth":     "gdp_growth",
    "CPI":            "cpi",
    "Unemployment":   "unemployment",
    "10Y Yield":      "yield_10y",
    "2Y Yield":       "yield_2y",
    "Fed Funds Rate": "fed_funds_rate",
}

# FRED series that are released at daily frequency and need month-end resampling.
_DAILY_SERIES: set[str] = {"10Y Yield", "2Y Yield"}

# FRED series that are released quarterly and need forward-filling to monthly.
_QUARTERLY_SERIES: set[str] = {"GDP Growth"}

# Maximum number of months to forward-fill before dropping a row as unrecoverable.
_FFILL_LIMIT: int = 3


def load_macro_data() -> pd.DataFrame:
    """Pull all FRED macro series and return a clean monthly DataFrame.

    Handles three frequency types:
    - Daily series (yields): resampled to month-end last value.
    - Quarterly series (GDP): forward-filled to monthly frequency.
    - Monthly series: used as-is.

    All series are aligned to a common monthly DatetimeIndex, trimmed to the
    date range in config, and forward-filled up to _FFILL_LIMIT months to
    bridge short FRED data gaps.

    Returns:
        DataFrame with a monthly DatetimeIndex and columns:
        gdp_growth, cpi, unemployment, yield_10y, yield_2y,
        fed_funds_rate, yield_spread.
    """
    fred = Fred(api_key=config.FRED_API_KEY)
    series_frames: list[pd.Series] = []

    for name, series_id in config.FRED_SERIES.items():
        raw: pd.Series = fred.get_series(series_id)
        col_name = _COLUMN_NAMES[name]

        if name in _DAILY_SERIES:
            # Take the last observation of each month to get a fixed monthly snapshot.
            monthly = raw.resample("ME").last()
        elif name in _QUARTERLY_SERIES:
            # Resample quarterly release to month-end, then forward-fill the gaps
            # so the most recent GDP reading propagates across the quarter.
            monthly = raw.resample("ME").last().ffill()
        else:
            # Monthly series: resample to normalize any irregular day-of-month offsets.
            monthly = raw.resample("ME").last()

        monthly.name = col_name
        series_frames.append(monthly)

    df = pd.concat(series_frames, axis=1)

    # Normalize index to month-start for consistency across all downstream code.
    # to_period("M") collapses any day offset, then to_timestamp() gives the 1st of each month.
    df.index = df.index.to_period("M").to_timestamp()
    df.index.name = "date"

    # Trim to configured date range before filling so we don't fill across boundaries.
    df = df.loc[config.START_DATE : config.END_DATE]

    # Derived column: yield spread must be computed after both yield series are aligned.
    df["yield_spread"] = df["yield_10y"] - df["yield_2y"]

    # Bridge short data gaps (e.g. holidays, late FRED releases) but don't over-impute.
    df = df.ffill(limit=_FFILL_LIMIT)

    # Drop any rows still missing data after forward-fill — likely a genuine gap.
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Warning: dropped {n_dropped} row(s) with unresolvable missing data.")

    print(
        f"Macro data loaded: {df.index[0].date()} to {df.index[-1].date()} "
        f"({len(df)} monthly observations, {len(df.columns)} indicators)"
    )

    return df
