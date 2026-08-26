"""Cointegration --- spec Sec 2, 4, 7.

Planned modules
---------------
``screen.py``    Engle-Granger both directions + BH FDR control  [Day 4-6]
``halflife.py``  OU / AR(1) half-life estimation                 [Day 4-6]
``survival.py``  Kaplan-Meier shelf-life analysis (the headline)  [Day 16-18]

Non-negotiable: use ``statsmodels.tsa.stattools.coint``, never ``adfuller`` on
OLS residuals. Beta was estimated from the same data, so textbook ADF critical
values reject far too often; ``coint`` applies the MacKinnon / Engle-Granger
values that account for it (spec Sec 2.3).
"""
