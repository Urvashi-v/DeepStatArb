"""Spread models --- spec Sec 2.5, 3.

Planned modules
---------------
``ols.py``     static and rolling hedge ratio, trailing z-score  [Day 7-9]
``kalman.py``  time-varying hedge ratio via Kalman filter        [Day 13-15]

The Kalman innovation IS the spread and e/sqrt(F) IS the z-score, both computed
from information up to t-1 before seeing the price at t --- no lookahead by
construction (spec Sec 3.2).
"""
