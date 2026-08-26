"""Data layer --- spec Sec 11.

Contract: nothing outside this subpackage downloads anything. Analysis code
loads from the parquet cache in ``data/clean`` and never re-downloads.

Planned modules
---------------
``download.py``  yfinance / jugaad-data pulls, cached to ``data/raw``  [Day 2]
``quality.py``   gap, split-artefact and outlier checks                [Day 3]
``panel.py``     one call that returns the clean, aligned price panel  [Day 3]
"""
