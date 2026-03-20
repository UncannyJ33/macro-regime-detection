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

RANDOM_SEED: int = 42       # Seeds numpy before model fits for reproducible cluster IDs

# Maps HMM Viterbi state integers to human-readable regime names.
# Only states 0, 1, and 3 appear — state 2 is intentionally absent because
# fit_hmm() collapses it: the model assigned state 2 to a single month
# (April 2020, the COVID crash), which is too short to be a meaningful regime
# and immediately transitions back to state 1. That month gets absorbed into
# state 1 (Stagflation) instead. As a result, "Contraction" has no
# corresponding HMM state; the Contraction allocation rule in REGIME_ALLOCATIONS
# is defined but will never be triggered when running with HMM labels.
HMM_REGIME_LABELS: dict[int, str] = {
    0: "Expansion",    # most common state: moderate growth, low unemployment, normal conditions
    1: "Stagflation",  # stagnant growth (~0% GDP), highest CPI, elevated unemployment
    3: "Risk-off",     # post-crisis recovery: highest unemployment legacy, lowest CPI,
                       # steepest yield curve (Fed still accommodative), GDP rebounding
}

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
