"""Synthetic market and news generators.

The market is geometric Brownian motion plus regime shifts and
occasional shocks, which gives the pattern detectors something
realistic to chew on. News headlines are sampled from a small
template pool and biased toward the current regime so that the
sentiment signal carries information.

The point isn't realism — it's having a self-contained dataset so the
agent can train and we can verify the pipeline end-to-end without
needing an external data feed. Replace these generators with real
adapters (yfinance, Polygon, NewsAPI, etc.) when wiring to live data.
"""

import math
import random
from typing import List, Tuple

from .patterns import Bar


_BULL_HEADLINES = [
    "Tech giants rally on strong demand for AI chips",
    "Fed signals a rate cut as inflation cools",
    "Retail sales beat expectations for third quarter",
    "New trade deal signed between major economies",
    "Energy stocks surge on supply expansion",
]

_BEAR_HEADLINES = [
    "Markets plunge on recession fears",
    "Major bank reports surprise miss; downgrade follows",
    "Sanctions escalate amid ongoing war in eastern Europe",
    "Tariff hike sparks fears of global slowdown",
    "Layoffs sweep across the tech sector",
]

_NEUTRAL_HEADLINES = [
    "Central bank holds rates steady",
    "Index closes flat after choppy session",
    "Analysts split on outlook for next quarter",
    "Quiet trading session as investors await data",
]


def generate_market(
    n_bars: int = 1500,
    start_price: float = 100.0,
    seed: int = 42,
) -> Tuple[List[Bar], List[int]]:
    """Generate `n_bars` synthetic OHLC bars and a per-bar regime tag.

    Regimes: 0 = sideways/calm, 1 = bull trend, 2 = bear trend, 3 = shock.
    The regime tag is returned so the news generator can stay in sync.
    """
    rng = random.Random(seed)
    bars: List[Bar] = []
    regimes: List[int] = []
    price = start_price
    regime = 0
    bars_left_in_regime = rng.randint(80, 180)
    drift_table = {0: 0.0001, 1: 0.0020, 2: -0.0020, 3: 0.0}
    vol_table = {0: 0.008, 1: 0.010, 2: 0.012, 3: 0.030}

    for i in range(n_bars):
        if bars_left_in_regime <= 0:
            regime = rng.choices([0, 1, 2, 3], weights=[0.35, 0.3, 0.3, 0.05])[0]
            bars_left_in_regime = rng.randint(40, 120)
        bars_left_in_regime -= 1

        drift = drift_table[regime]
        vol = vol_table[regime]
        ret = rng.gauss(drift, vol)
        new_price = max(1.0, price * math.exp(ret))

        # Construct an OHLC bar around the move.
        high = max(price, new_price) * (1 + abs(rng.gauss(0, vol / 2)))
        low = min(price, new_price) * (1 - abs(rng.gauss(0, vol / 2)))
        bars.append(Bar(open=price, high=high, low=low, close=new_price))
        regimes.append(regime)
        price = new_price

    return bars, regimes


def generate_news_stream(
    regimes: List[int],
    seed: int = 7,
    headlines_per_bar: int = 2,
) -> List[List[str]]:
    """Generate per-bar lists of headlines biased by the underlying regime."""
    rng = random.Random(seed)
    out: List[List[str]] = []
    for r in regimes:
        if r == 1:
            pool, weight_self = _BULL_HEADLINES, 0.7
        elif r == 2:
            pool, weight_self = _BEAR_HEADLINES, 0.7
        elif r == 3:
            pool, weight_self = _BEAR_HEADLINES, 0.85
        else:
            pool, weight_self = _NEUTRAL_HEADLINES, 0.6

        bar_headlines: List[str] = []
        for _ in range(headlines_per_bar):
            if rng.random() < weight_self:
                bar_headlines.append(rng.choice(pool))
            else:
                bar_headlines.append(rng.choice(_NEUTRAL_HEADLINES))
        out.append(bar_headlines)
    return out
