# DeepStatArb

Statistical arbitrage on cointegrated NSE pairs — and a measurement of how long
a statistical relationship actually survives.

The strategy at the centre of this project is forty years old and largely
arbitraged away. That is not a problem with the project; it is the reason the
project is worth building. Because the honest expected result is known in
advance, every hour goes into measurement quality rather than into chasing a
number.

**Headline deliverable:** a Kaplan-Meier survival curve — what fraction of
cointegrated pairs are still cointegrated *N* months after formation.

---

## Status

**Day 4 of ~20 complete: research universe.**

The NSE price panel is downloaded and the universe is frozen. No *strategy*
experiment has been run: every strategy results slot below is empty on purpose
and will stay empty until a real run fills it. The data-layer numbers that do
appear are counts from an actual download, reproducible with one command.

| Stage | Days | Status |
|---|---|---|
| Project foundation | 1 | **done** |
| Data acquisition pipeline | 2 | **done** |
| Data quality gates | 3 | **done** |
| Research universe | 4 | **done** |
| Cointegration screening + FDR | 5–6 | not started |
| Spread, signal, first backtest | 7–9 | not started |
| Walk-forward engine + Indian costs | 10–12 | not started |
| Kalman time-varying hedge ratio | 13–15 | not started |
| Shelf-life survival analysis | 16–18 | not started |
| ML filter (XGBoost) | 19–22 | not started |
| LSTM ablation | 23–25 | not started (optional arm) |
| Statistical validity | 26–28 | not started |
| Dashboard and writeup | 29–30 | not started |

---

## Results

> Every number here is produced by a script that writes to `reports/`, never
> typed in by hand. A `NOT YET MEASURED` entry means exactly that: no run has
> produced it. Numbers marked *Measured* name the command that regenerates them.

### The selection funnel

Measured (Day 2) — from `scripts/build_dataset.py`:

```
NSE F&O individual securities                        208
  -> with an NSE sector classification               208   (100% coverage)
  -> with a complete download and no FATAL finding   203
  -> eligible on >=1 formation date (frozen universe) 200

candidate pairs, unrestricted                     19,900
  -> restricted to same sector                     2,115   (89.4% removed)
```

**The frozen list is not the tradeable universe.** Eligibility is recomputed at
each formation date from a trailing window, so it varies:

```
eligible names   2018-01-01   105    (52% of the frozen list)
                 2021-01-01   125
                 2026-01-01   199   (100%)
                 range 105-199, median 150
```

Not yet measured — needs the cointegration screen (Day 4–6):

```
  -> raw p < 0.05                          NOT YET MEASURED
  -> surviving BH FDR control at 10%       NOT YET MEASURED
  -> tradeable half-life (5-60 days)       NOT YET MEASURED
```

At 5% with 2,115 same-sector tests, roughly 106 pairs would pass by chance alone
with no true cointegration anywhere. That is what the BH FDR control is for.

The funnel matters more than any equity curve, because it is the difference
between a discovery and a coincidence. Testing all 19,900 unrestricted pairs at
the 5% level would yield roughly 995 "cointegrated" pairs in a world where none
are. The same-sector prior removes 89.4% of the search space before a single
test is run; BH FDR control and out-of-sample re-testing handle what is left.

### Survival of cointegration — the headline

```
median survival (months)                 NOT YET MEASURED
same-sector vs cross-sector              NOT YET MEASURED
```

### Strategy

```
Sharpe (net, 95% bootstrap CI)           NOT YET MEASURED
breakeven cost level (bps round trip)    NOT YET MEASURED
percentile vs random-pair null           NOT YET MEASURED
max drawdown / turnover / trade count    NOT YET MEASURED
```

**Expected outcome, stated before the fact:** modest or negative Sharpe after
honest costs. That is the documented finding in the literature (Do & Faff 2010),
and the report is built around it. A net-of-cost Sharpe above roughly 1.5 on
daily-frequency pairs trading in liquid Indian equities would be treated as a
bug report, not a discovery.

---

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

```bash
python scripts/check_env.py
```

Build the price panel (~2 minutes of downloads, resumable):

```bash
python scripts/check_data_connection.py
```

```bash
python scripts/build_dataset.py --stage all
```

```bash
python scripts/run_quality_report.py --universe-only
```

```bash
python scripts/build_universe.py
```

Optional extras: `pip install -e ".[dl]"` for the LSTM ablation, `".[app]"` for
the dashboard, `".[all]"` for everything.

---

## How this project avoids lying to itself

Eight failure modes make a pairs-trading backtest look better than reality.
Every one of them produces a *better* result, which is why they survive — nobody
debugs a good number. Each is met with a mechanical guard rather than good
intentions.

| # | The lie | The guard | Where |
|---|---|---|---|
| 1 | Full-sample z-score | `zscore_shift >= 1` refused at config load | `config.py`, `test_config.py` |
| 2 | Unadjusted prices | `price_field: close` refused at config load | `config.py` |
| 3 | No multiple-testing control | BH FDR + sector prior + OOS re-test | `coint/screen.py` *(Day 4-6)* |
| 4 | Trading at the signal price | `lag_bars >= 1`, refused at config load | `config.py` |
| 5 | Survivorship bias | Declared in `universe.yaml` with its direction | `config/universe.yaml` |
| 6 | Random train/test split | `split: temporal_only` enforced | `config.py` |
| 7 | Ignoring the short-sale constraint | `short_leg.decided` must be set explicitly | `config/costs.yaml` |
| 8 | Refitting inside the trading window | Frozen config + the leak detector | `validity/leakage.py` |

### The leak detector

Built on day one rather than day twenty-five, because a leak found the week
before an interview is a leak that has already contaminated every number.

```python
from dsa.validity import assert_causal

assert_causal(my_feature_fn, price_data)   # raises if it reads the future
```

It computes a feature on the full history and on truncated copies, and asserts
the overlapping values agree. If a feature computed on `data[:100]` differs at
position 99 from the same feature computed on `data[:200]`, then future rows
changed the past — that is a leak by definition, with no reasoning about the
code required.

It catches full-sample statistics, centred windows, `shift(-1)`, globally
fitted scalers and hedge ratios. It deliberately does **not** catch same-bar
execution, which is a different bug guarded separately by `lag_bars`. Its known
limitation — sparse leaks such as a `bfill` across an occasional one-day hole
can evade the default checkpoints — is documented and pinned by a test, rather
than left to be discovered.

### Reproducibility

- **Config is frozen and hashed.** Every run is identified by a 12-character
  digest of every parameter it used. The deflated Sharpe ratio needs an honest
  count of trials; that count is only trustworthy if runs are identified by
  content rather than by memory.
- **Named random streams.** One master seed, with each component drawing from
  an independently derived stream (`spawn_rng(seed, "null_control", run=17)`).
  Adding a new component tomorrow does not perturb the 500-run null
  distribution reported yesterday.
- **Provenance capture.** Git commit, dirty-tree flag, package versions and
  platform are recorded next to every run.

---

## Layout

```
config/       universe, costs, backtest, models, base  <- frozen, versioned, hashed
src/dsa/
  data/       download, quality checks, the clean panel       [Day 2-3]
  coint/      Engle-Granger screen, half-life, survival       [Day 4-6, 16-18]
  spread/     OLS and Kalman hedge ratios                     [Day 7-9, 13-15]
  signal/     entry, exit, stops                              [Day 7-9]
  filter/     features, labels, XGBoost, LSTM                 [Day 19-25]
  backtest/   walk-forward engine, Indian costs, metrics      [Day 10-12]
  validity/   leak detector, null control, deflated Sharpe    [Day 1, 26-28]
tests/        pytest suite; test_no_lookahead.py is the important one
data/raw/     downloaded, never edited      (gitignored)
data/clean/   adjusted, gap-checked parquet (gitignored)
figures/      regenerated by make, never by hand
reports/      run logs, tables, the writeup
app/          streamlit dashboard                             [Day 29-30]
scripts/      entry points
```

---

## The data layer

167 NSE F&O-eligible names, 2,862 trading days (2015-01-01 to 2026-07-31), built
from two published NSE files and Yahoo Finance. **No credentials of any kind are
required.**

```
data/raw/ohlcv/*.parquet      208 tickers as received, write-once, 524,484 rows
data/raw/reference/*.csv      the NSE F&O and sector files, byte-for-byte
data/clean/panel_*.parquet    adjusted, aligned panels (dates x symbols)
```

The download is resumable: a manifest records what completed, so an interrupted
pull costs the tickers still outstanding rather than all 208.

### Three things about this feed worth knowing

Each was verified against real data, not assumed:

1. **Yahoo's `Close` is already split-adjusted.** The 2:1 RELIANCE split on
   2017-09-07 shows as −0.56%, not −50%. So `data/raw` is *as received*, not
   *as traded* — no free feed offers a genuinely unadjusted NSE series. The
   consequence: the ">20% single-day move" screen in spec §11.2 will **not**
   surface splits here, because the provider already handled them. What it will
   surface is genuine moves and splits the provider handled *badly*, which is
   the case worth finding.
2. **Back-adjustment puts future information into past levels.** The adjusted
   2015-01-01 RELIANCE close is 0.93184 × the split-adjusted close, and that
   factor is the product of every dividend paid *after* 2015. Strictly, the
   2015 series could not have been computed in 2015. The spec mandates adjusted
   prices and the alternative — dividend jumps in the spread — is worse, so this
   is accepted and declared. Worth knowing that a hedge ratio fitted on
   back-adjusted prices is not exactly the one a trader would have fitted live.
3. **Gaps are left as NaN, never forward-filled.** Forward-filling a halted
   stock invents a flat price, and a flat stretch reads to an ADF test as an
   unusually well-behaved spread — manufacturing exactly the false pairs the
   FDR control exists to remove.

### Quality report

`scripts/run_quality_report.py` writes `reports/data_quality/`. Twelve checks:
duplicate timestamps, chronological order, missing values, invalid prices,
zero volume, stale prices, extreme moves, missing sessions, gaps, coverage,
phantom sessions, padded sessions, and ticker consistency.

**It flags; it never deletes.** Silently dropping suspicious observations is
itself a form of lookahead — you would be choosing which days to keep using
knowledge of what the price did on them, and the days that look wrong are
disproportionately the days something real happened. A test asserts the input
frames come out byte-identical.

Measured on the frozen universe (167 names, 2,857 sessions):

```
FATAL findings                                   0
padded non-trading days found and excluded       5
extreme moves >20%                             118   (59 isolated, 59 market-wide)
stale-price runs                                 2   (13 and 17 sessions)
zero-volume runs beyond 3 sessions               2
coverage over listed life                     100%   for all 167 names
tradeable coverage (lowest)                  99.31%   DALBHARAT.NS
```

The five padded days are the finding that mattered most. Yahoo fills NSE
non-trading dates with a carried-forward close and zero volume; four have
**zero** tickers trading, and 2025-03-18 is a partial provider failure where
201 of 203 names got a padded row while two got real data. Left in place each
one inserts a zero return into nearly every series, which biases volatility
down and makes every spread look flatter — and therefore more mean-reverting —
than it is. Session inference now measures participation by *volume*, not by
the presence of a price, which is what separates them out.

---

## The research universe

Two levels, deliberately separated — this is what reconciles §11.4's "freeze the
universe" with the rule against lookahead.

| Level | What it is | Frozen? |
|---|---|---|
| **Research universe** | the 200 names ever considered | yes — versioned + hashed |
| **Point-in-time eligibility** | who qualifies at formation date *T*, from data ≤ *T* | recomputed per window |

The membership *list* is frozen so an experiment is reproducible; *eligibility
at a date* is computed causally, and it is eligibility the walk-forward loop
consults.

`config/selection.yaml` holds the rules (input, hand-written).
`config/universe.yaml` holds the frozen result (output, generated) plus a copy
of the rules that produced it. Universe hash: `75a798c41d03`, version 2. The
version bumps only when the hash changes, so it counts real changes rather than
script runs.

### The lookahead this replaced

The Day-2 filter computed median turnover over the **entire** 2015–2026 sample
and used it to admit a name to the 2015 universe. A stock thin until 2022
passed on its full-sample median and would then have been traded in 2015. That
is the universe-construction analogue of the full-sample z-score, and just as
invisible: it makes the equity curve better, not worse. `finalise_universe` now
raises `NotImplementedError` rather than being deleted quietly, so any surviving
call site fails loudly.

A test appends future data to the panel and requires every eligibility decision
already made to come back bit-identical — on synthetic panels and on the real
NSE panel.

### A measured issue you should know about

A **fixed nominal** turnover floor is not scale-invariant over eleven years.
Median daily turnover across the F&O universe went ₹48cr (2018) → ₹158cr (2026),
so the ₹25cr floor excluded 54 names in 2018 and **zero** in 2026. That
confounds the era analysis (Days 26–28): early windows would trade half as many
names as late ones. Setting `max_eligible_per_date` in `selection.yaml` switches
to a relative rule (top *N* by trailing turnover); measured effect on universe
size stability: sd 33.2 → 1.2. Left off by default — the choice is the
researcher's, and must be made once, before results are looked at.

---

## Known limitations

Stated here from the start rather than discovered at the end.

1. **Survivorship bias.** A universe built from today's F&O list and
   backtested to 2015 contains only the names that survived and stayed liquid.
   The direction of the bias is optimistic. Partial reconstruction from
   archived NSE circulars will be attempted; whatever remains is declared.
2. **The short leg.** NSE cash equity cannot be shorted for delivery. A pairs
   strategy holding a short leg for two weeks is not implementable in cash
   equities. The method — single-stock futures, intraday only, or cash with an
   explicit caveat — is recorded in `config/costs.yaml` and is currently
   **undecided**, which the config enforces rather than allowing a silent
   default.
3. **Cost rates are unverified.** The rates in `costs.yaml` carry
   `verified: false`. STT on futures and exchange transaction charges have both
   changed in recent years and will be checked against primary NSE/SEBI sources
   before any cost-sensitivity number is reported.
4. **Free data has gaps and bad prints.** Series will be cross-checked against
   a second source, and single-day moves beyond 20% flagged for inspection.
5. **Daily frequency only.** Nothing here speaks to what happens at higher
   frequency, where reversion is faster but costs and market impact dominate.

---

## References

- Gatev, Goetzmann & Rouwenhorst (2006), *Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule*, RFS — the foundational empirical study.
- Do & Faff (2010), *Does Simple Pairs Trading Still Work?* — documents the
  decline in profitability and the effect of realistic costs.
- Engle & Granger (1987), *Co-integration and Error Correction*, Econometrica.
- Johansen (1991), *Estimation and Hypothesis Testing of Cointegration Vectors*.
- MacKinnon (2010), *Critical Values for Cointegration Tests* — the corrected
  critical values that make an Engle-Granger test on estimated residuals valid.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, JPM.
- Chan, E., *Algorithmic Trading: Winning Strategies and Their Rationale*.

---

## Data and credentials

No API keys, tokens or paid subscriptions are required. NSE daily OHLCV comes
from `yfinance` (unauthenticated) with `auto_adjust=True`. If a keyed source is
ever introduced, it goes in `.env` (gitignored) with a documented entry in
`.env.example` — never in `config/`, which is tracked.
