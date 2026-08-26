"""Backtest engine --- spec Sec 6.

Planned modules
---------------
``engine.py``   walk-forward loop, 36/6/6 months            [Day 10-12]
``costs.py``    Indian cost stack, liquidity-bucketed slippage [Day 10-12]
``metrics.py``  Sharpe, drawdown, turnover, trade count      [Day 7-9]
``runlog.py``   automatic configuration log (Sec 12)         [Day 10-12]

The alignment rule lives in exactly one place in ``engine.py``: a signal
computed from the close of t produces a position that acts from t+1 onward.
"""
