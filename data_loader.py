"""Fetches and aligns raw macroeconomic and asset price data.

Pulls FRED macro series and yfinance price data, resamples everything
to a common monthly frequency, and returns a single merged DataFrame
ready for feature engineering.
"""
