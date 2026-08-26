# Data quality report

generated   : 2026-08-25T10:32:41+00:00
config hash : f7155d5c6dba
symbols     : 167
sessions    : 2857 (2015-01-01 .. 2026-07-31)

**Nothing below has been deleted, repaired or interpolated.** Every
finding is a flag for inspection. Dropping suspicious observations
would itself be a form of lookahead: the days that look wrong are
disproportionately the days something real happened.

## Verdict

**No FATAL findings.** No duplicated bars, no non-monotonic indexes,
no non-positive prices, no impossible OHLC. The panel is structurally sound.

**5 padded non-trading day(s)** were identified and excluded from the session calendar. They are listed under `padded_session` below and must be dropped before returns are computed.

941 observation(s) flagged for inspection across 68 warning group(s).

## Counts by check

| check | FATAL | WARN | INFO | total | what it means |
|---|---:|---:|---:|---:|---|
| `zero_volume` | 0 | 28 | 927 | 955 | bars with no shares traded |
| `padded_session` | 0 | 824 | 0 | 824 | a non-trading date the provider filled with a carried-forward close |
| `extreme_move` | 0 | 59 | 59 | 118 | single-day move beyond the inspection threshold |
| `stale_price` | 0 | 30 | 0 | 30 | an unchanged close across several sessions |

## WARN

### `extreme_move` --- single-day move beyond the inspection threshold

- **ADANIENT.NS** 2015-06-03: -38.7% single-day move with no corporate action reported nearby. Isolated --- only 10% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIENT.NS** 2017-04-26: -20.9% single-day move with no corporate action reported nearby. Isolated --- only 2% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIENT.NS** 2018-09-06: +20.9% single-day move with no corporate action reported nearby. Isolated --- only 5% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIENT.NS** 2020-08-25: +23.7% single-day move with no corporate action reported nearby. Isolated --- only 3% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIENT.NS** 2023-02-02: -26.7% single-day move with no corporate action reported nearby. Isolated --- only 5% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIENT.NS** 2023-02-08: +20.0% single-day move with no corporate action reported nearby. Isolated --- only 4% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIENT.NS** 2024-11-21: -22.6% single-day move with no corporate action reported nearby. Isolated --- only 4% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ADANIGREEN.NS** 2024-11-29: +21.8% single-day move with no corporate action reported nearby. Isolated --- only 1% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ANGELONE.NS** 2020-10-27: +20.0% single-day move with no corporate action reported nearby. Isolated --- only 6% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **ASHOKLEY.NS** 2020-03-19: -25.2% single-day move; nearest corporate action 2020-03-19 (dividend 0.25, split 0.00), 0 session(s) away. Yahoo pre-adjusts NSE splits, so a move this large next to a reported action suggests the adjustment was applied on the wrong date or not at all. 39% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **AUROPHARMA.NS** 2020-02-19: +20.3% single-day move; nearest corporate action 2020-02-17 (dividend 1.75, split 0.00), 2 session(s) away. Yahoo pre-adjusts NSE splits, so a move this large next to a reported action suggests the adjustment was applied on the wrong date or not at all. Isolated --- only 6% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **CDSL.NS** 2017-07-03: -49.9% single-day move with no corporate action reported nearby. Isolated --- only 4% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **CGPOWER.NS** 2016-02-03: -21.7% single-day move with no corporate action reported nearby. Isolated --- only 8% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **CONCOR.NS** 2025-01-01: -20.9% single-day move with no corporate action reported nearby. Isolated --- only 1% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **DIVISLAB.NS** 2016-12-23: -22.3% single-day move with no corporate action reported nearby. Isolated --- only 2% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **GLENMARK.NS** 2019-11-18: +21.4% single-day move with no corporate action reported nearby. Isolated --- only 2% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **GODREJCP.NS** 2021-05-12: +21.9% single-day move with no corporate action reported nearby. Isolated --- only 5% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **HINDZINC.NS** 2024-05-21: +25.6% single-day move; nearest corporate action 2024-05-15 (dividend 10.00, split 0.00), 3 session(s) away. Yahoo pre-adjusts NSE splits, so a move this large next to a reported action suggests the adjustment was applied on the wrong date or not at all. Isolated --- only 8% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2017-01-30: +25.3% single-day move with no corporate action reported nearby. Isolated --- only 3% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2019-07-29: -27.0% single-day move with no corporate action reported nearby. Isolated --- only 6% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2019-10-24: -23.0% single-day move with no corporate action reported nearby. Isolated --- only 8% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2019-11-14: -20.3% single-day move with no corporate action reported nearby. Isolated --- only 4% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2019-11-15: +23.7% single-day move with no corporate action reported nearby. Isolated --- only 5% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2019-11-18: +21.9% single-day move with no corporate action reported nearby. Isolated --- only 2% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- **IDEA.NS** 2019-11-19: +36.0% single-day move with no corporate action reported nearby. Isolated --- only 4% of the universe moved more than 5% that day, which makes a name-specific event or a bad print more likely.
- _... and 34 more (see issues.csv)_

### `padded_session` --- a non-trading date the provider filled with a carried-forward close

- **-** 2025-03-18: only 1 of 167 listed tickers traded and only 1 moved at all. This is a PARTIAL provider failure rather than a clean holiday: most names got a padded row while a few got real data, so the date looks like a thin session instead of an absent one. NOT DELETED --- the date is excluded from the inferred session calendar, and downstream stages should drop it before computing returns. Left in place it inserts a zero return into nearly every series, which biases volatility down and makes every spread look flatter, and therefore more mean-reverting, than it is.
- **-** 2026-01-15: no listed ticker traded and none moved: an exchange holiday the provider filled with a carried-forward close and zero volume. NOT DELETED --- the date is excluded from the inferred session calendar, and downstream stages should drop it before computing returns. Left in place it inserts a zero return into nearly every series, which biases volatility down and makes every spread look flatter, and therefore more mean-reverting, than it is.
- **-** 2026-05-01: no listed ticker traded and none moved: an exchange holiday the provider filled with a carried-forward close and zero volume. NOT DELETED --- the date is excluded from the inferred session calendar, and downstream stages should drop it before computing returns. Left in place it inserts a zero return into nearly every series, which biases volatility down and makes every spread look flatter, and therefore more mean-reverting, than it is.
- **-** 2026-05-28: no listed ticker traded and none moved: an exchange holiday the provider filled with a carried-forward close and zero volume. NOT DELETED --- the date is excluded from the inferred session calendar, and downstream stages should drop it before computing returns. Left in place it inserts a zero return into nearly every series, which biases volatility down and makes every spread look flatter, and therefore more mean-reverting, than it is.
- **-** 2026-06-26: no listed ticker traded and none moved: an exchange holiday the provider filled with a carried-forward close and zero volume. NOT DELETED --- the date is excluded from the inferred session calendar, and downstream stages should drop it before computing returns. Left in place it inserts a zero return into nearly every series, which biases volatility down and makes every spread look flatter, and therefore more mean-reverting, than it is.

### `stale_price` --- an unchanged close across several sessions

- **DALBHARAT.NS** 2019-01-07: close unchanged at 1090.00 for 13 consecutive sessions 2019-01-07..2019-01-22. A flat stretch reads to an ADF test as an exceptionally well-behaved spread, which is how a suspended stock manufactures a false cointegration result.
- **PNBHOUSING.NS** 2021-10-20: close unchanged at 477.44 for 17 consecutive sessions 2021-10-20..2021-11-11. A flat stretch reads to an ADF test as an exceptionally well-behaved spread, which is how a suspended stock manufactures a false cointegration result.

### `zero_volume` --- bars with no shares traded

- **DALBHARAT.NS** 2019-01-07: 12 consecutive zero-volume sessions 2019-01-07..2019-01-22. Any spread built on this window is fiction regardless of what the price column says.
- **PNBHOUSING.NS** 2021-10-20: 16 consecutive zero-volume sessions 2021-10-20..2021-11-11. Any spread built on this window is fiction regardless of what the price column says.

## INFO

### `extreme_move` --- single-day move beyond the inspection threshold

- **ADANIENSOL.NS** 2023-01-27: -20.0% single-day move with no corporate action reported nearby. 12% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **ADANIENT.NS** 2019-05-20: +27.4% single-day move with no corporate action reported nearby. 36% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **ADANIENT.NS** 2023-02-01: -28.2% single-day move with no corporate action reported nearby. 12% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **ADANIPORTS.NS** 2024-06-04: -21.1% single-day move with no corporate action reported nearby. 58% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **ADANIPOWER.NS** 2018-10-10: +25.3% single-day move with no corporate action reported nearby. 26% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **ADANIPOWER.NS** 2020-03-12: -26.5% single-day move with no corporate action reported nearby. 85% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **ASHOKLEY.NS** 2020-03-26: +24.4% single-day move with no corporate action reported nearby. 42% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **AXISBANK.NS** 2020-03-23: -27.9% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BAJAJFINSV.NS** 2020-03-23: -25.9% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BAJFINANCE.NS** 2020-03-23: -23.2% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BANDHANBNK.NS** 2020-03-23: -25.0% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BANDHANBNK.NS** 2020-03-26: +39.3% single-day move with no corporate action reported nearby. 42% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BANKBARODA.NS** 2016-02-15: +22.6% single-day move with no corporate action reported nearby. 32% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BANKBARODA.NS** 2017-10-25: +31.4% single-day move with no corporate action reported nearby. 14% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BANKINDIA.NS** 2017-10-25: +34.1% single-day move with no corporate action reported nearby. 14% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BHEL.NS** 2019-10-18: +21.9% single-day move with no corporate action reported nearby. 11% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BHEL.NS** 2020-05-13: +23.7% single-day move with no corporate action reported nearby. 19% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **BHEL.NS** 2024-06-04: -20.8% single-day move with no corporate action reported nearby. 58% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **CANBK.NS** 2017-10-25: +38.7% single-day move with no corporate action reported nearby. 14% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **CANBK.NS** 2020-03-23: -20.9% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **CHOLAFIN.NS** 2020-03-23: -29.6% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **COFORGE.NS** 2020-03-23: -24.4% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **COFORGE.NS** 2020-03-25: +22.0% single-day move with no corporate action reported nearby. 35% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **FEDERALBNK.NS** 2020-03-23: -24.2% single-day move with no corporate action reported nearby. 90% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- **GLENMARK.NS** 2020-03-12: -23.1% single-day move with no corporate action reported nearby. 85% of the universe also moved more than 5% that day, so this is a market-wide move rather than a name-specific one.
- _... and 34 more (see issues.csv)_

### `zero_volume` --- bars with no shares traded

- **ABCAPITAL.NS**: 5 bar(s) with zero volume (0.23% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ADANIENSOL.NS**: 6 bar(s) with zero volume (0.22% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ADANIENT.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ADANIGREEN.NS**: 5 bar(s) with zero volume (0.25% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ADANIPORTS.NS**: 6 bar(s) with zero volume (0.21% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ADANIPOWER.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ALKEM.NS**: 4 bar(s) with zero volume (0.15% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **AMBUJACEM.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ANGELONE.NS**: 6 bar(s) with zero volume (0.42% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **APOLLOHOSP.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ASHOKLEY.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **ASIANPAINT.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **AUBANK.NS**: 5 bar(s) with zero volume (0.22% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **AUROPHARMA.NS**: 4 bar(s) with zero volume (0.14% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **AXISBANK.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BAJAJ-AUTO.NS**: 6 bar(s) with zero volume (0.21% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BAJAJFINSV.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BAJFINANCE.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BANDHANBNK.NS**: 5 bar(s) with zero volume (0.24% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BANKBARODA.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BANKINDIA.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BDL.NS**: 5 bar(s) with zero volume (0.24% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BEL.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BHARATFORG.NS**: 4 bar(s) with zero volume (0.14% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- **BHARTIARTL.NS**: 5 bar(s) with zero volume (0.17% of the series). A single one is usually a halt; a large fraction means the name was not reliably tradeable.
- _... and 142 more (see issues.csv)_

## Per-ticker coverage

`coverage` asks whether a bar was supplied; `tradeable` asks whether
anything actually changed hands. A name can show 100% coverage and
still have been untradeable for weeks, so the two are shown together.

Ten lowest-tradeability names (over their own listed life):

| symbol | first | last | sessions | observed | coverage | tradeable | zero-vol | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DALBHARAT.NS | 2018-12-21 | 2026-07-31 | 1877 | 1877 | 100.0% | 99.3% | 13 | 0 |
| PNBHOUSING.NS | 2016-11-07 | 2026-07-31 | 2404 | 2404 | 100.0% | 99.3% | 16 | 0 |
| POLYCAB.NS | 2019-04-16 | 2026-07-31 | 1800 | 1800 | 100.0% | 99.7% | 5 | 0 |
| ETERNAL.NS | 2021-07-23 | 2026-07-31 | 1240 | 1240 | 100.0% | 99.8% | 3 | 0 |
| SONACOMS.NS | 2021-06-24 | 2026-07-31 | 1260 | 1260 | 100.0% | 99.8% | 3 | 0 |
| IDFCFIRSTB.NS | 2015-11-06 | 2026-07-31 | 2647 | 2647 | 100.0% | 99.8% | 6 | 0 |
| DLF.NS | 2015-01-01 | 2026-07-31 | 2857 | 2857 | 100.0% | 99.8% | 6 | 0 |
| ZYDUSLIFE.NS | 2015-01-01 | 2026-07-31 | 2857 | 2857 | 100.0% | 99.8% | 6 | 0 |
| BRITANNIA.NS | 2015-01-01 | 2026-07-31 | 2857 | 2857 | 100.0% | 99.8% | 5 | 0 |
| GMRAIRPORT.NS | 2015-01-01 | 2026-07-31 | 2857 | 2857 | 100.0% | 99.8% | 5 | 0 |
