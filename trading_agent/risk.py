"""Risk manager.

Hard rule from the user: every trade has a stop loss equal to half the
target profit, i.e. risk:reward = 1:2. The take-profit distance is
sized off ATR so it adapts to current volatility, and the stop is then
mechanically half of that.

Beyond the base predictability gate, this module supports several
optional confluence filters that only allow a trade when multiple
independent signals agree. Enabled by default; each can be toggled off
via the constructor to isolate their contribution to win rate.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    direction: int            # +1 long, -1 short
    entry: float
    stop_loss: float
    take_profit: float
    size: float
    bar_index: int

    def is_hit(self, high: float, low: float) -> Optional[str]:
        """Return 'tp', 'sl', or None given the bar's range."""
        if self.direction > 0:
            hit_sl = low <= self.stop_loss
            hit_tp = high >= self.take_profit
        else:
            hit_sl = high >= self.stop_loss
            hit_tp = low <= self.take_profit
        # Pessimistic: if both touched in the same bar, assume stop first.
        if hit_sl and hit_tp:
            return "sl"
        if hit_sl:
            return "sl"
        if hit_tp:
            return "tp"
        return None

    def realized_r(self, exit_price: float) -> float:
        """PnL expressed in R-multiples (1R = stop distance)."""
        risk_per_unit = abs(self.entry - self.stop_loss)
        if risk_per_unit == 0:
            return 0.0
        return self.direction * (exit_price - self.entry) / risk_per_unit


class RiskManager:
    """Sizes trades and enforces the user's 1:2 risk-reward rule."""

    RISK_REWARD_RATIO = 0.5  # stop loss = 0.5 * take-profit distance

    def __init__(
        self,
        account_equity: float = 10_000.0,
        risk_per_trade: float = 0.01,
        atr_target_mult: float = 2.0,
        min_signal: float = 0.50,
        require_trend_confluence: bool = False,
        require_sentiment_agreement: bool = False,
        volatility_guard: bool = False,
        max_atr_ratio: float = 1.8,
        rsi_extreme_guard: bool = False,
        rsi_overbought: float = 82.0,
        rsi_oversold: float = 18.0,
        overextension_guard: bool = False,
        max_atr_from_mean: float = 2.5,
    ) -> None:
        self.account_equity = account_equity
        self.risk_per_trade = risk_per_trade
        self.atr_target_mult = atr_target_mult
        self.min_signal = min_signal
        self.require_trend_confluence = require_trend_confluence
        self.require_sentiment_agreement = require_sentiment_agreement
        self.volatility_guard = volatility_guard
        self.max_atr_ratio = max_atr_ratio
        self.rsi_extreme_guard = rsi_extreme_guard
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.overextension_guard = overextension_guard
        self.max_atr_from_mean = max_atr_from_mean

    def is_predictable(
        self,
        signal: float,
        direction: int,
        trend: float = 0.0,
        sentiment: float = 0.0,
        atr_ratio: float = 1.0,
        rsi_value: float = 50.0,
        atr_from_mean: float = 0.0,
    ) -> bool:
        """Gate: pattern signal must be strong and aligned with direction,
        with layered confluence checks. `atr_from_mean` is signed distance
        of price from a short SMA measured in ATRs (positive = above)."""

        # Base signal gate.
        if direction > 0 and signal < self.min_signal:
            return False
        if direction < 0 and signal > -self.min_signal:
            return False

        # Trend alignment: LONG shouldn't fight a strong downtrend and
        # vice versa (soft guard — a modest counter-signal is allowed).
        if self.require_trend_confluence:
            if direction > 0 and trend < -0.10:
                return False
            if direction < 0 and trend > 0.10:
                return False

        # Sentiment agreement: news mood shouldn't sharply contradict direction.
        if self.require_sentiment_agreement:
            if direction > 0 and sentiment < -0.5:
                return False
            if direction < 0 and sentiment > 0.5:
                return False

        # Volatility guard: skip during shock regimes.
        if self.volatility_guard and atr_ratio > self.max_atr_ratio:
            return False

        # RSI extreme guard: don't chase overbought longs / oversold shorts.
        if self.rsi_extreme_guard and rsi_value == rsi_value:  # NaN check
            if direction > 0 and rsi_value > self.rsi_overbought:
                return False
            if direction < 0 and rsi_value < self.rsi_oversold:
                return False

        # Overextension guard: reject chasing price too far from mean.
        if self.overextension_guard:
            if direction > 0 and atr_from_mean > self.max_atr_from_mean:
                return False
            if direction < 0 and atr_from_mean < -self.max_atr_from_mean:
                return False

        return True

    def build_trade(
        self,
        direction: int,
        entry: float,
        atr_value: float,
        bar_index: int,
        world_risk: int = 0,
    ) -> Optional[Trade]:
        if atr_value <= 0 or direction == 0:
            return None

        # Target profit distance, scaled down in high-risk world states.
        risk_scale = {0: 1.0, 1: 0.7, 2: 0.4}.get(world_risk, 1.0)
        tp_distance = atr_value * self.atr_target_mult * risk_scale
        sl_distance = tp_distance * self.RISK_REWARD_RATIO  # half the profit

        if direction > 0:
            stop_loss = entry - sl_distance
            take_profit = entry + tp_distance
        else:
            stop_loss = entry + sl_distance
            take_profit = entry - tp_distance

        # Position sizing: risk a fixed fraction of equity per trade.
        risk_dollars = self.account_equity * self.risk_per_trade * risk_scale
        size = risk_dollars / max(sl_distance, 1e-9)

        return Trade(
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            bar_index=bar_index,
        )

    def update_equity(self, pnl: float) -> None:
        self.account_equity += pnl
