"""Central configuration for all pipeline parameters.

All hardcoded values live here. Import from this module; never hardcode
values in other files.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into the environment before any os.environ calls

# ── API Keys ──────────────────────────────────────────────────────────────────

FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# ── Data ──────────────────────────────────────────────────────────────────────

START_DATE: str = "2000-01-01"
END_DATE: str = "2024-06-30"

# FRED series IDs for macroeconomic indicators.
# Values are the official series codes used by the FRED API.
FRED_SERIES: dict[str, str] = {
    "GDP Growth":       "A191RL1Q225SBEA",  # Real GDP growth rate (quarterly, %)
    "CPI":              "CPIAUCSL",          # Consumer Price Index (monthly)
    "Unemployment":     "UNRATE",            # Civilian unemployment rate (monthly, %)
    "10Y Yield":        "DGS10",             # 10-year Treasury constant maturity rate
    "2Y Yield":         "DGS2",              # 2-year Treasury constant maturity rate
    "Fed Funds Rate":   "FEDFUNDS",          # Effective federal funds rate
}

# Asset tickers and their human-readable descriptions.
ASSETS: dict[str, str] = {
    "SPY": "US Equities",
    "TLT": "Long-term Treasuries",
    "GLD": "Gold",
    "DBC": "Commodities",
}

# ── Model ─────────────────────────────────────────────────────────────────────

N_REGIMES: int = 4          # Number of macro regimes to identify
N_PCA_COMPONENTS: int = 4   # PCA components to retain after feature reduction
ROLLING_WINDOW: int = 24    # Rolling window in months for feature smoothing

FFILL_LIMIT: int = 3        # Max months to forward-fill FRED data gaps

KMEANS_N_INIT: int = 20     # K-Means random restarts (best inertia is kept)
ELBOW_K_MIN: int = 2        # Smallest K tested in elbow analysis
ELBOW_K_MAX: int = 8        # Largest K tested in elbow analysis

HMM_N_INIT: int = 10        # HMM random initializations (best log-likelihood kept)
HMM_N_ITER: int = 200       # Max EM iterations per HMM fit

# ── Backtester ────────────────────────────────────────────────────────────────

RISK_FREE_RATE: float = 0.02  # Annual risk-free rate used in Sharpe ratio calculation

# Allocation weights by regime name. Keys must exactly match the labels passed to
# label_regimes() so the backtester can look up weights by name.
REGIME_ALLOCATIONS: dict[str, dict[str, float]] = {
    "Expansion":   {"SPY": 0.70, "TLT": 0.15, "GLD": 0.10, "DBC": 0.05},
    "Contraction": {"SPY": 0.20, "TLT": 0.50, "GLD": 0.20, "DBC": 0.10},
    "Stagflation": {"SPY": 0.20, "TLT": 0.20, "GLD": 0.40, "DBC": 0.20},
    "Risk-off":    {"SPY": 0.10, "TLT": 0.60, "GLD": 0.25, "DBC": 0.05},
}

# Static 60/40 benchmark used as comparison baseline.
BASELINE_ALLOCATIONS: dict[str, float] = {"SPY": 0.60, "TLT": 0.40}
