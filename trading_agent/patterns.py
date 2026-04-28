"""Chart and candlestick pattern detection.

Each detector returns a confidence score in [0, 1]. Bullish patterns are
positive contributions; bearish patterns are negative. The aggregated
`dominant_signal` is a value in [-1, 1] used by the agent as a feature
and by the risk manager as a "predictability" gate — trades are only
taken when |signal| exceeds a threshold.
"""

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 1e-9)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((xs[i] - mx) * (values[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-9
    return num / den


def detect_trend(bars: Sequence[Bar], lookback: int = 20) -> float:
    """Return signed trend strength in [-1, 1]."""
    if len(bars) < lookback:
        return 0.0
    closes = [b.close for b in bars[-lookback:]]
    s = _slope(closes)
    avg = sum(closes) / len(closes)
    norm = s * lookback / max(avg, 1e-9)
    return max(-1.0, min(1.0, norm * 50))


def detect_bullish_engulfing(bars: Sequence[Bar]) -> float:
    if len(bars) < 2:
        return 0.0
    p, c = bars[-2], bars[-1]
    if not p.is_bullish and c.is_bullish and c.open <= p.close and c.close >= p.open:
        return min(1.0, c.body / max(p.body, 1e-9) * 0.6)
    return 0.0


def detect_bearish_engulfing(bars: Sequence[Bar]) -> float:
    if len(bars) < 2:
        return 0.0
    p, c = bars[-2], bars[-1]
    if p.is_bullish and not c.is_bullish and c.open >= p.close and c.close <= p.open:
        return min(1.0, c.body / max(p.body, 1e-9) * 0.6)
    return 0.0


def detect_hammer(bars: Sequence[Bar]) -> float:
    """Bullish reversal candle after a downtrend."""
    if len(bars) < 6:
        return 0.0
    c = bars[-1]
    lower_wick = min(c.open, c.close) - c.low
    upper_wick = c.high - max(c.open, c.close)
    if c.body <= 0 or lower_wick < 2 * c.body or upper_wick > c.body:
        return 0.0
    prior_trend = detect_trend(bars[:-1], lookback=10)
    if prior_trend >= 0:
        return 0.0
    return min(1.0, abs(prior_trend) * (lower_wick / c.range))


def detect_shooting_star(bars: Sequence[Bar]) -> float:
    """Bearish reversal candle after an uptrend."""
    if len(bars) < 6:
        return 0.0
    c = bars[-1]
    upper_wick = c.high - max(c.open, c.close)
    lower_wick = min(c.open, c.close) - c.low
    if c.body <= 0 or upper_wick < 2 * c.body or lower_wick > c.body:
        return 0.0
    prior_trend = detect_trend(bars[:-1], lookback=10)
    if prior_trend <= 0:
        return 0.0
    return min(1.0, prior_trend * (upper_wick / c.range))


def detect_double_bottom(bars: Sequence[Bar], lookback: int = 30, tol: float = 0.01) -> float:
    if len(bars) < lookback:
        return 0.0
    window = bars[-lookback:]
    lows = [b.low for b in window]
    i1 = lows.index(min(lows[: lookback // 2]))
    second_half = lows[lookback // 2:]
    i2 = lookback // 2 + second_half.index(min(second_half))
    if i2 - i1 < 5:
        return 0.0
    diff = abs(lows[i1] - lows[i2]) / max(lows[i1], 1e-9)
    if diff > tol:
        return 0.0
    peak_between = max(b.high for b in window[i1:i2])
    if window[-1].close <= peak_between:
        return 0.0
    return max(0.0, 1.0 - diff / tol)


def detect_double_top(bars: Sequence[Bar], lookback: int = 30, tol: float = 0.01) -> float:
    if len(bars) < lookback:
        return 0.0
    window = bars[-lookback:]
    highs = [b.high for b in window]
    i1 = highs.index(max(highs[: lookback // 2]))
    second_half = highs[lookback // 2:]
    i2 = lookback // 2 + second_half.index(max(second_half))
    if i2 - i1 < 5:
        return 0.0
    diff = abs(highs[i1] - highs[i2]) / max(highs[i1], 1e-9)
    if diff > tol:
        return 0.0
    trough_between = min(b.low for b in window[i1:i2])
    if window[-1].close >= trough_between:
        return 0.0
    return max(0.0, 1.0 - diff / tol)


def detect_breakout(bars: Sequence[Bar], lookback: int = 20) -> float:
    """Positive on upside breakout of recent range, negative on downside."""
    if len(bars) < lookback + 1:
        return 0.0
    window = bars[-(lookback + 1):-1]
    hi = max(b.high for b in window)
    lo = min(b.low for b in window)
    last = bars[-1]
    rng = max(hi - lo, 1e-9)
    if last.close > hi:
        return min(1.0, (last.close - hi) / rng * 5)
    if last.close < lo:
        return -min(1.0, (lo - last.close) / rng * 5)
    return 0.0


def pattern_features(bars: Sequence[Bar]) -> dict:
    """All pattern scores. Bullish positive, bearish negative."""
    return {
        "trend": detect_trend(bars),
        "bull_engulf": detect_bullish_engulfing(bars),
        "bear_engulf": -detect_bearish_engulfing(bars),
        "hammer": detect_hammer(bars),
        "shooting_star": -detect_shooting_star(bars),
        "double_bottom": detect_double_bottom(bars),
        "double_top": -detect_double_top(bars),
        "breakout": detect_breakout(bars),
    }


def dominant_signal(bars: Sequence[Bar]) -> float:
    """Aggregate signal in [-1, 1]. Used as the predictability gate."""
    feats = pattern_features(bars)
    weights = {
        "trend": 0.25,
        "bull_engulf": 0.10,
        "bear_engulf": 0.10,
        "hammer": 0.10,
        "shooting_star": 0.10,
        "double_bottom": 0.15,
        "double_top": 0.15,
        "breakout": 0.20,
    }
    score = sum(feats[k] * weights[k] for k in feats)
    return max(-1.0, min(1.0, score))
