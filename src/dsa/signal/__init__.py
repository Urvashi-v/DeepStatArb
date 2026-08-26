"""Trading rules --- spec Sec 5.

Planned modules
---------------
``rules.py``  entry, exit, stop-loss, time-stop, pair kill switch  [Day 7-9]

Thresholds come from ``config/backtest.yaml`` and are fixed a priori. Any sweep
over them is a trial and must be paid for by the deflated Sharpe (Sec 10.2).

Note: this package is named ``signal`` to match the spec's file tree. It does
not shadow the standard library module of that name --- Python 3 uses absolute
imports, so ``import signal`` elsewhere still resolves to the stdlib.
"""
