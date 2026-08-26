"""DeepStatArb --- statistical arbitrage on cointegrated NSE pairs.

The headline deliverable is not the equity curve. It is a survival curve:
what fraction of cointegrated pairs are still cointegrated N months later.

Subpackages
-----------
``data``      download, cache and quality-check the NSE price panel
``coint``     Engle-Granger screening, BH FDR control, half-life, survival
``spread``    static / rolling OLS and the Kalman time-varying hedge ratio
``signal``    entry, exit, stop-loss and time-stop rules
``filter``    features, labels, the XGBoost filter, the LSTM ablation
``backtest``  walk-forward engine, Indian cost stack, metrics
``validity``  leak detection, random-pair null, deflated Sharpe, bootstrap
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
