"""The ML filter --- spec Sec 8, 9.

The model's role is narrow and deliberate: the strategy generates entry
candidates and the model decides which to SKIP. It never generates trades.

Planned modules
---------------
``features.py``  feature builder, past-only by construction   [Day 19-22]
``labels.py``    forward-looking labels, horizon from half-life [Day 19-22]
``gbm.py``       XGBoost filter, temporal splits               [Day 19-22]
``lstm.py``      sequence ablation, optional arm               [Day 23-25]

The rule that governs this whole subpackage: labels look forward, features look
backward. Every function in ``features.py`` must pass
``dsa.validity.leakage.assert_causal``.
"""
