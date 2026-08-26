"""Statistical validity --- spec Sec 10, 14.

Modules
-------
``leakage.py``       the leak detector. Built Day 1, used every day after.
``null_control.py``  random-pair null, several hundred runs   [Day 26-28]
``deflated.py``      deflated Sharpe across the config log     [Day 26-28]
``bootstrap.py``     stationary block bootstrap intervals      [Day 26-28]

Spec Sec 10.3: "If your net-of-cost Sharpe comes out above roughly 1.5 on
daily-frequency pairs trading over liquid Indian equities, stop and hunt for a
bug before celebrating."
"""

from dsa.validity.leakage import (
    LeakageError,
    LeakageReport,
    assert_causal,
    check_causal,
)

__all__ = ["LeakageError", "LeakageReport", "assert_causal", "check_causal"]
