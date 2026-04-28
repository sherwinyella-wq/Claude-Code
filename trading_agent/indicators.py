"""Pure-Python technical indicators. No numpy/pandas required."""

from typing import List, Sequence


def sma(values: Sequence[float], period: int) -> List[float]:
    out: List[float] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        out.append(running / period if i >= period - 1 else float("nan"))
    return out


def ema(values: Sequence[float], period: int) -> List[float]:
    out: List[float] = []
    k = 2.0 / (period + 1)
    prev = None
    for i, v in enumerate(values):
        if prev is None:
            prev = v
        else:
            prev = v * k + prev * (1 - k)
        out.append(prev if i >= period - 1 else float("nan"))
    return out


def rsi(closes: Sequence[float], period: int = 14) -> List[float]:
    out: List[float] = [float("nan")] * len(closes)
    if len(closes) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[float]:
    trs: List[float] = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
    return sma(trs, period)
