# Macro Regime Detection

Unsupervised ML that identifies macroeconomic regimes from FRED data and routes a four-asset portfolio accordingly — tested against 18 years of live market returns.

![Regime Timeline](outputs/regime_timeline.png)

## What This Project Does

The pipeline pulls six macroeconomic indicators from FRED (GDP growth, CPI, unemployment, the yield curve, and the Fed funds rate), engineers 16 rolling features capturing both levels and momentum, and reduces them to principal components. K-Means and a Gaussian Hidden Markov Model independently classify each month into one of four regimes: Expansion, Contraction, Stagflation, and Risk-off. A regime-conditional backtester then holds a different asset mix (SPY, TLT, GLD, DBC) for each regime and compares performance against a static 60/40 benchmark over 220 months.

## Why It Matters

Macro funds do exactly this: they form a view on the economic environment and tilt allocations before the market catches up, rather than reacting after the fact. The interesting result here isn't that the strategy beats 60/40 on returns — it barely does — it's that it achieves the same compound growth with a meaningfully smaller drawdown, which is where regime awareness is supposed to show up.

## Key Results

Backtest period: March 2006 – June 2024 (220 months). HMM Viterbi labels feed the backtester.

| Metric | Regime Strategy | 60/40 Baseline |
|---|---|---|
| Total Return | 315.20% | 310.09% |
| Annualized Return | 8.07% | 8.00% |
| Annualized Volatility | 10.37% | 10.36% |
| Sharpe Ratio | 0.61 | 0.60 |
| Max Drawdown | **−22.53%** | −28.45% |
| Calmar Ratio | **0.36** | 0.28 |

The regime strategy achieves similar returns at similar volatility but survives a 6-percentage-point shallower max drawdown — the primary benefit of regime-conditional defensiveness.

## Methodology

### Data Sources
- **FRED API**: GDP growth (quarterly, forward-filled), CPI, unemployment, 10Y and 2Y Treasury yields, Fed funds rate — 2000–2024.
- **yfinance**: Adjusted monthly close prices for SPY, TLT, GLD, DBC. DBC (commodities ETF) starts in 2006, so the effective backtest period begins there.

### Feature Engineering
Each indicator generates two features: a 24-month rolling z-score (how extreme is the current reading relative to recent history?) and a month-over-month rate of change (which direction is it moving?). The yield spread gets two extras: an inversion flag and a second-derivative acceleration term. That produces 16 features total.

### Dimensionality Reduction
StandardScaler → PCA. Four components are retained, collectively explaining the majority of variance. A second StandardScaler pass normalizes PC variances before clustering, since PCA components are ordered by explained variance rather than cluster-relevance.

### Regime Models: K-Means vs HMM
Both models run on the same PCA-reduced features.

**K-Means** clusters independently — each month is assigned to the nearest centroid without any temporal constraint. This produces visually clean, contiguous-looking regime bands in the timeline because macro variables are autocorrelated, not because the model enforces it. Regime labels are assigned post-hoc by inspecting cluster centroids: highest GDP growth → Expansion, highest unemployment → Contraction, highest CPI among the remaining two → Stagflation, the last cluster → Risk-off.

**Gaussian HMM** models regime transitions explicitly: the probability of entering a state depends on the current state, capturing persistence and switching dynamics that K-Means cannot. Multiple random initializations are run and the best log-likelihood fit is kept. Viterbi decoding recovers the single most probable state sequence. The HMM labels feed the backtester because they more accurately reflect how regimes transition in practice.

### Backtest Design
Each month, the portfolio holds a fixed allocation determined by that month's HMM regime label. No look-ahead: the regime is identified from data available through that month. Allocations:

| Regime | SPY | TLT | GLD | DBC |
|---|---|---|---|---|
| Expansion | 70% | 15% | 10% | 5% |
| Contraction | 20% | 50% | 20% | 10% |
| Stagflation | 20% | 20% | 40% | 20% |
| Risk-off | 10% | 60% | 25% | 5% |

Sharpe is computed against a 2% annual risk-free rate.

### Known Limitations
The allocation rules were designed after observing the full sample, so they embed look-ahead bias — the weights reflect knowledge of which asset classes performed well in each regime rather than a truly out-of-sample hypothesis. A walk-forward optimization (fitting the model on a rolling window and testing on the next unseen period) would give a cleaner read on whether regime detection adds value before committing capital.

## Project Structure

```
macro-regime-detection/
├── config.py               # All parameters: dates, tickers, K, window sizes, allocations
├── data_loader.py          # FRED + yfinance ingestion, monthly alignment
├── feature_engineering.py  # Rolling z-scores, rates of change, yield spread signals
├── regime_models.py        # PCA, K-Means, HMM fitting, elbow analysis, PCA loadings
├── backtester.py           # Regime-conditional portfolio simulation and metrics
├── visualizations.py       # Regime timeline, return distributions, allocation heatmap
├── main.py                 # Pipeline entry point — runs all stages in sequence
├── outputs/                # Generated charts (regime_timeline.png, etc.)
└── requirements.txt        # Python dependencies
```

## Quick Start

```bash
git clone https://github.com/<you>/macro-regime-detection.git
cd macro-regime-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your FRED API key (free at fred.stlouisfed.org)
echo "FRED_API_KEY=your_key_here" > .env

python main.py
```

Outputs are written to `outputs/`. Runtime is roughly 60–90 seconds, dominated by FRED API calls and HMM initialization.

## Tech Stack

Python 3.10 · pandas · scikit-learn · hmmlearn · fredapi · yfinance · matplotlib
