"""Risk manager.

Hard rule from the user: every trade has a stop loss equal to half the
target profit, i.e. risk:reward = 1:2. The take-profit distance is
sized off ATR so it adapts to current volatility, and the stop is then
mechanically half of that.

The manager also gates entries on pattern "predictability": the
aggregate pattern signal must exceed `min_signal` (default 0.35) in the
direction of the trade, otherwise the setup is rejected.
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
        min_signal: float = 0.45,
    ) -> None:
        self.account_equity = account_equity
        self.risk_per_trade = risk_per_trade
        self.atr_target_mult = atr_target_mult
        self.min_signal = min_signal

    def is_predictable(self, signal: float, direction: int) -> bool:
        """Gate: only take trades when the pattern signal is strong
        and aligned with the trade direction."""
        if direction > 0:
            return signal >= self.min_signal
        return signal <= -self.min_signal

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
