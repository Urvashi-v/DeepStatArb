# Research universe

generated     : 2026-08-26T12:29:09+00:00
universe hash : `75a798c41d03`
version       : 2
config hash   : `685298a85a49`

## Size

- candidates with price data: **203**
- frozen research universe: **200** names across **18** sectors
- unrestricted pairs: **19,900**
- same-sector pairs: **2,115** (89.4% of the search space removed before a single test is run)

At a 5% test level, the same-sector pair count alone would yield roughly **106** 'cointegrated' pairs in a world where none are. That is what the BH FDR control exists for (spec Sec 4.1).

## Eligible names over time

The frozen list is a single number; the set you could actually have traded
is not. Eligibility is re-evaluated at each formation date using only data
available then, so a name that became liquid in 2022 is ineligible before it.

| formation date | eligible | share of frozen universe |
|---|---:|---:|
| 2018-01-01 | 105 | 52% |
| 2018-06-29 | 110 | 55% |
| 2019-01-01 | 114 | 57% |
| 2019-07-01 | 111 | 56% |
| 2020-01-01 | 113 | 56% |
| 2020-07-01 | 110 | 55% |
| 2021-01-01 | 125 | 62% |
| 2021-07-01 | 137 | 68% |
| 2021-12-31 | 150 | 75% |
| 2022-07-01 | 153 | 76% |
| 2022-12-30 | 155 | 78% |
| 2023-06-30 | 159 | 80% |
| 2024-01-01 | 168 | 84% |
| 2024-07-01 | 182 | 91% |
| 2025-01-01 | 192 | 96% |
| 2025-07-01 | 196 | 98% |
| 2026-01-01 | 199 | 100% |

Range: **105** to **199** eligible names (median 150).

## Sectors

Classification is NSE's own `Industry` field, not one written here. That
matters: the same-sector prior (Sec 4.1) and the survival stratification
(Sec 7.2) are only defensible if the grouping came from somewhere other
than the person hoping to find pairs in it.

| sector | names | same-sector pairs |
|---|---:|---:|
| Financial Services | 52 | 1,326 |
| Capital Goods | 21 | 210 |
| Healthcare | 15 | 105 |
| Automobile and Auto Components | 14 | 91 |
| Fast Moving Consumer Goods | 14 | 91 |
| Information Technology | 12 | 66 |
| Consumer Durables | 10 | 45 |
| Metals & Mining | 10 | 45 |
| Oil Gas & Consumable Fuels | 9 | 36 |
| Power | 8 | 28 |
| Consumer Services | 7 | 21 |
| Realty | 6 | 15 |
| Services | 5 | 10 |
| Construction Materials | 5 | 10 |
| Chemicals | 5 | 10 |
| Telecommunication | 3 | 3 |
| Construction | 3 | 3 |
| Textiles | 1 | 0 |
| **total** | **200** | **2,115** |

1 sector(s) hold a single name and can therefore form no same-sector pair at all: Textiles.

## Exclusions

**3** of 203 candidates excluded.

| reason | names | what it means |
|---|---:|---|
| `never_eligible` | 3 | never cleared the point-in-time bar on any decision date |

Every excluded name, with the statistics behind the decision, is in
`exclusions.csv`. Nothing is dropped without a recorded reason.

## Liquidity

Trailing median daily turnover at the last decision date (2026-01-01), in Rs crore.
Measured over traded sessions only --- a zero-volume halt is not a session
you could have traded, whatever the price column says.

| statistic | Rs crore |
|---|---:|
| minimum | 53.8 |
| 25th percentile | 101.8 |
| 50th percentile | 158.1 |
| 75th percentile | 242.0 |
| maximum | 1,711.3 |
| **median** | **158.1** |

Ten most liquid, and ten least:

| most liquid | Rs cr | least liquid | Rs cr |
|---|---:|---|---:|
| HDFCBANK.NS | 1,711.3 | RADICO.NS | 53.8 |
| RELIANCE.NS | 1,390.1 | DALBHARAT.NS | 54.1 |
| ICICIBANK.NS | 1,251.0 | PETRONET.NS | 54.2 |
| BSE.NS | 1,236.6 | ALKEM.NS | 54.3 |
| INFY.NS | 1,137.0 | ICICIPRULI.NS | 54.4 |
| BHARTIARTL.NS | 964.2 | NAM-INDIA.NS | 58.4 |
| ETERNAL.NS | 926.5 | CROMPTON.NS | 67.0 |
| TCS.NS | 822.9 | BOSCHLTD.NS | 67.6 |
| SBIN.NS | 767.6 | BAJAJHLDNG.NS | 67.7 |
| M&M.NS | 746.7 | CONCOR.NS | 67.8 |

## Coverage

- listed history at 2026-01-01: median **11.0y**, minimum **3.0y**, maximum **11.0y**
- missing/untradeable sessions in the trailing window: median **0.00%**, worst **1.59%**
- names with a full trailing window: **197/200**

Least complete names in the trailing window:

| symbol | history (y) | missing | median turnover (Rs cr) |
|---|---:|---:|---:|
| GVT&D.NS | 11.0 | 1.59% | 94.0 |
| ETERNAL.NS | 4.4 | 0.79% | 926.5 |
| M&M.NS | 11.0 | 0.40% | 746.7 |

## Criteria applied

From `config/selection.yaml`. Every point-in-time criterion is evaluated
using only data available on or before the decision date.

| parameter | value |
|---|---|
| `exclude_quality_fatal` | True |
| `liquidity_lookback_months` | 12 |
| `max_eligible_per_date` | None |
| `max_missing_frac` | 0.02 |
| `max_size` | None |
| `min_history_years` | 3.0 |
| `min_median_turnover_inr` | 250000000 |
| `min_price_inr` | 20.0 |
| `rank_by` | median_turnover_inr |
| `require_complete_download` | True |
| `require_ever_eligible` | True |
| `require_sector` | True |
| `tie_break` | symbol |

## Bias declaration

**Survivorship: not eliminated.** The candidate pool is today's NSE F&O
list. Names dropped from F&O, delisted, or shrunk out of the segment are
absent, and their absence flatters every downstream result. Direction:
**optimistic**. Point-in-time F&O constituent history is not available
free, so this is declared rather than fixed.

**Lookahead in selection: eliminated.** Liquidity, history, price and
coverage are computed from a trailing window ending at the decision date.
Appending future data to the panel does not change any eligibility
decision already made --- asserted by a test that re-runs the selection
against a truncated panel and requires identical output.
