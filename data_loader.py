"""Fetches and aligns raw macroeconomic and asset price data.

Pulls FRED macro series and yfinance price data, resamples everything
to a common monthly frequency, and returns a single merged DataFrame
ready for feature engineering.
"""

import pandas as pd
import yfinance as yf
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
    df = df.ffill(limit=config.FFILL_LIMIT)

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


def load_asset_returns() -> pd.DataFrame:
    """Pull monthly adjusted close prices for all assets and return monthly returns.

    Fetches each ticker in config.ASSETS from yfinance, resamples to month-end,
    and computes simple percentage returns. The index is normalized to month-start
    to match the macro DataFrame. Assets with shorter histories (e.g. DBC from 2006)
    will have NaN returns before their inception date — these are left in place
    rather than trimming the full DataFrame, so the backtester can handle the
    mismatch explicitly.

    Returns:
        DataFrame with a monthly DatetimeIndex and one column per ticker.
        First row (NaN from return calculation) is dropped.
    """
    tickers = list(config.ASSETS.keys())

    raw = yf.download(
        tickers,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested.
    prices: pd.DataFrame = raw["Close"]

    # Resample daily prices to month-end last value, consistent with yield series.
    monthly_prices = prices.resample("ME").last()

    # Percentage return: (this month - last month) / last month.
    returns = monthly_prices.pct_change()

    # Drop first row — it's always NaN since there's no prior month to compare.
    returns = returns.iloc[1:]

    # Normalize to month-start to match macro DatetimeIndex.
    returns.index = returns.index.to_period("M").to_timestamp()
    returns.index.name = "date"

    # Preserve column order to match config.ASSETS key order.
    returns = returns[tickers]

    print(
        f"Asset returns loaded: {returns.index[0].date()} to {returns.index[-1].date()} "
        f"({len(returns)} monthly observations, {len(returns.columns)} assets)"
    )

    return returns


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load macro indicators and asset returns, aligned to the macro date range.

    The macro DataFrame drives the index — asset returns are reindexed to match
    it rather than using an inner join. This preserves the full macro history
    (back to 2000) even though some assets (e.g. DBC) only start in 2006.
    Months before an asset's inception will show NaN in returns_df; the
    backtester handles these by only running on months where returns are available.

    Returns:
        Tuple of (macro_df, returns_df) sharing the macro DatetimeIndex.
    """
    macro_df = load_macro_data()
    returns_df = load_asset_returns()

    # Reindex returns to macro's full index — introduces NaN for pre-inception months.
    returns_df = returns_df.reindex(macro_df.index)

    first_complete = returns_df.dropna().index[0].date()
    print(
        f"Combined dataset: macro spans {macro_df.index[0].date()} to {macro_df.index[-1].date()}, "
        f"all assets available from {first_complete}"
    )

    return macro_df, returns_df
