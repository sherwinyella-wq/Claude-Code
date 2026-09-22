"""Backtester that ties the agent, risk manager, and feature pipeline
together. Walks one bar at a time:

  1. Compute features (indicators + patterns + sentiment + world risk).
  2. If flat, ask the agent for an action and (if entering) build a trade
     via the RiskManager — which enforces the 1:2 stop:profit rule and
     the predictability gate.
  3. If in a trade, check the bar range against stop and target.
  4. On trade close, update the agent's Q-table with the realized reward
     in R-multiples (winner = +2R, loser = -1R by construction).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .agent import QLearningAgent, build_state, HOLD, LONG, SHORT
from .indicators import atr, rsi, sma
from .patterns import Bar, dominant_signal, detect_trend
from .risk import RiskManager, Trade
from .sentiment import score_headlines, world_risk_level


@dataclass
class TradeRecord:
    direction: int
    entry: float
    exit: float
    outcome: str           # 'tp' or 'sl'
    pnl: float
    r_multiple: float
    bar_in: int
    bar_out: int


@dataclass
class BacktestResult:
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    starting_equity: float = 0.0

    @property
    def ending_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.starting_equity

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity * 100

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.outcome == "tp")
        return wins / len(self.trades) * 100

    @property
    def expectancy_r(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.r_multiple for t in self.trades) / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        worst = 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            worst = max(worst, dd)
        return worst

    def summary(self) -> str:
        return (
            f"trades={len(self.trades)}  "
            f"win_rate={self.win_rate:.1f}%  "
            f"expectancy={self.expectancy_r:+.2f}R  "
            f"return={self.total_return_pct:+.2f}%  "
            f"max_dd={self.max_drawdown_pct:.2f}%"
        )


class Backtester:
    WARMUP = 30

    def __init__(
        self,
        agent: QLearningAgent,
        risk_manager: RiskManager,
        training: bool = True,
        require_confirmation: bool = False,
    ) -> None:
        self.agent = agent
        self.risk = risk_manager
        self.training = training
        self.require_confirmation = require_confirmation

    def run(
        self,
        bars: List[Bar],
        news: List[List[str]],
    ) -> BacktestResult:
        result = BacktestResult(starting_equity=self.risk.account_equity)
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        rsi_series = rsi(closes, 14)
        atr_series = atr(highs, lows, closes, 14)
        sma_series = sma(closes, 20)

        # Rolling median ATR is used to detect volatility spikes.
        def _median_atr(end_i: int, window: int = 50) -> float:
            start = max(0, end_i - window)
            vals = [a for a in atr_series[start:end_i] if a == a and a > 0]
            if not vals:
                return 0.0
            vals = sorted(vals)
            n = len(vals)
            return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

        open_trade: Optional[Trade] = None
        pending_state = None
        pending_action = None

        for i in range(len(bars)):
            equity_now = self.risk.account_equity
            result.equity_curve.append(equity_now)

            if i < self.WARMUP:
                continue

            if self.training:
                self.agent.decay_epsilon()

            window = bars[: i + 1]
            signal = dominant_signal(window)
            trend = detect_trend(window)
            sent = score_headlines(news[i])
            risk_lvl = world_risk_level(news[i])
            in_trade = open_trade is not None
            state = build_state(trend, signal, rsi_series[i], sent, risk_lvl, in_trade)

            # 1) Manage an open trade against this bar's range.
            if open_trade is not None:
                hit = open_trade.is_hit(highs[i], lows[i])
                if hit is not None:
                    exit_price = open_trade.take_profit if hit == "tp" else open_trade.stop_loss
                    r = open_trade.realized_r(exit_price)
                    pnl = r * abs(open_trade.entry - open_trade.stop_loss) * open_trade.size
                    self.risk.update_equity(pnl)
                    result.trades.append(TradeRecord(
                        direction=open_trade.direction,
                        entry=open_trade.entry,
                        exit=exit_price,
                        outcome=hit,
                        pnl=pnl,
                        r_multiple=r,
                        bar_in=open_trade.bar_index,
                        bar_out=i,
                    ))
                    # Reward the agent with the R-multiple from its entry decision.
                    if self.training and pending_state is not None and pending_action is not None:
                        self.agent.update(pending_state, pending_action, r, state)
                    open_trade = None
                    pending_state = None
                    pending_action = None
                continue

            # 2) Flat: maybe enter.
            action = self.agent.act(state, training=self.training)
            if action == HOLD:
                continue

            direction = 1 if action == LONG else -1
            atr_val = atr_series[i]
            med = _median_atr(i)
            atr_ratio = (atr_val / med) if (med and atr_val == atr_val) else 1.0
            mean_val = sma_series[i]
            atr_from_mean = 0.0
            if mean_val == mean_val and atr_val == atr_val and atr_val > 0:
                atr_from_mean = (closes[i] - mean_val) / atr_val
            gate_ok = self.risk.is_predictable(
                signal, direction,
                trend=trend, sentiment=sent, atr_ratio=atr_ratio,
                rsi_value=rsi_series[i], atr_from_mean=atr_from_mean,
            )
            if not gate_ok:
                if self.training:
                    next_state = build_state(trend, signal, rsi_series[i], sent, risk_lvl, False)
                    self.agent.update(state, action, -0.05, next_state)
                continue

            # Confirmation bar: require the *current* bar to close in the
            # trade direction relative to its open. Filters signals whose
            # follow-through fails, at the cost of a slightly worse entry
            # price. Increases win rate by throwing out false breakouts.
            if self.require_confirmation:
                bar = bars[i]
                if direction > 0 and bar.close <= bar.open:
                    continue
                if direction < 0 and bar.close >= bar.open:
                    continue

            entry = closes[i]
            if atr_val != atr_val or atr_val <= 0:
                continue
            trade = self.risk.build_trade(direction, entry, atr_val, i, world_risk=risk_lvl)
            if trade is None:
                continue
            open_trade = trade
            pending_state = state
            pending_action = action

        # Force-close anything still open at the end at last close.
        if open_trade is not None:
            exit_price = bars[-1].close
            r = open_trade.realized_r(exit_price)
            pnl = r * abs(open_trade.entry - open_trade.stop_loss) * open_trade.size
            self.risk.update_equity(pnl)
            result.trades.append(TradeRecord(
                direction=open_trade.direction,
                entry=open_trade.entry,
                exit=exit_price,
                outcome="tp" if r > 0 else "sl",
                pnl=pnl,
                r_multiple=r,
                bar_in=open_trade.bar_index,
                bar_out=len(bars) - 1,
            ))

        result.equity_curve.append(self.risk.account_equity)
        return result
