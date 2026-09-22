"""Ablation across confluence filters, one seed pool per config.

Confirms which filters actually improve OOS win rate and which hurt.
The current defaults in RiskManager (all confluence filters OFF) came
from running this study — the base pattern-signal gate is the only
filter with a clear positive contribution.

Usage:
    python3 ablate.py
    python3 ablate.py --seeds 50 --epochs 15
"""

import argparse
import statistics

from trading_agent import (QLearningAgent, RiskManager, Backtester,
                           generate_market, generate_news_stream)


def run(seeds: int, epochs: int, bars: int, **risk_kwargs):
    total_trades, wins, returns = 0, 0, []
    for s in range(seeds):
        bars_data, regimes = generate_market(n_bars=bars, seed=s)
        news = generate_news_stream(regimes, seed=s + 1)
        split = int(len(bars_data) * 0.7)
        agent = QLearningAgent(seed=s)
        for _ in range(epochs):
            Backtester(agent, RiskManager(**risk_kwargs), training=True).run(
                bars_data[:split], news[:split])
        r = Backtester(agent, RiskManager(**risk_kwargs), training=False).run(
            bars_data[split:], news[split:])
        total_trades += len(r.trades)
        wins += sum(1 for t in r.trades if t.outcome == "tp")
        returns.append(r.total_return_pct)
    return {
        "trades": total_trades,
        "win_rate": wins / total_trades * 100 if total_trades else 0.0,
        "avg_return": statistics.mean(returns) if returns else 0.0,
        "profitable_seeds": sum(1 for x in returns if x > 0),
    }


CONFIGS = [
    ("only base signal gate (default)", {}),
    ("+ trend confluence", {"require_trend_confluence": True}),
    ("+ sentiment agreement", {"require_sentiment_agreement": True}),
    ("+ volatility guard", {"volatility_guard": True}),
    ("+ RSI extreme guard", {"rsi_extreme_guard": True}),
    ("+ overextension guard", {"overextension_guard": True}),
    ("all filters on", {
        "require_trend_confluence": True,
        "require_sentiment_agreement": True,
        "volatility_guard": True,
        "rsi_extreme_guard": True,
        "overextension_guard": True,
    }),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bars", type=int, default=4000)
    args = p.parse_args()

    print(f"Ablation over {args.seeds} seeds, {args.epochs} epochs, {args.bars} bars\n")
    print(f"{'config':<36}{'trades':>8}{'win%':>8}{'ret%':>9}{'prof/n':>10}")
    print("-" * 71)
    for label, kwargs in CONFIGS:
        r = run(args.seeds, args.epochs, args.bars, **kwargs)
        print(f"{label:<36}{r['trades']:>8}{r['win_rate']:>7.1f} "
              f"{r['avg_return']:>+8.2f} {r['profitable_seeds']:>6}/{args.seeds}")


if __name__ == "__main__":
    main()
