"""Train the trading agent and run an out-of-sample backtest.

Usage:
    python3 train.py
    python3 train.py --epochs 12 --bars 4000 --seed 42

The dataset is split 70/30 into train and test. The agent learns over
multiple passes on the training half, then we freeze it and evaluate
on the unseen test half. The user's risk rule (stop = half the target,
i.e. 1:2 risk:reward) is enforced inside RiskManager regardless of
what the agent decides.

Research scaffold only. The market data here is synthetic. Wire up a
real feed (yfinance/Polygon/IBKR + a real news source) before drawing
any conclusions, and never trade live capital from this code without
extensive walk-forward validation, slippage modeling, and broker fees.
"""

import argparse

from trading_agent import (
    QLearningAgent,
    RiskManager,
    Backtester,
    generate_market,
    generate_news_stream,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-equity", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--save", type=str, default="agent_q.json")
    args = parser.parse_args()

    bars, regimes = generate_market(n_bars=args.bars, seed=args.seed)
    news = generate_news_stream(regimes, seed=args.seed + 1)

    split = int(len(bars) * 0.7)
    train_bars, test_bars = bars[:split], bars[split:]
    train_news, test_news = news[:split], news[split:]

    agent = QLearningAgent(seed=args.seed)

    print(f"Training on {len(train_bars)} bars, {args.epochs} epochs")
    for epoch in range(args.epochs):
        risk = RiskManager(
            account_equity=args.start_equity,
            risk_per_trade=args.risk_per_trade,
        )
        bt = Backtester(agent, risk, training=True)
        result = bt.run(train_bars, train_news)
        print(f"  epoch {epoch + 1}/{args.epochs}  eps={agent.epsilon:.3f}  {result.summary()}")

    agent.save(args.save)
    print(f"Saved Q-table -> {args.save}  (states={len(agent.q)})")

    print(f"\nEvaluating on held-out {len(test_bars)} bars (no exploration)...")
    eval_risk = RiskManager(
        account_equity=args.start_equity,
        risk_per_trade=args.risk_per_trade,
    )
    eval_bt = Backtester(agent, eval_risk, training=False)
    eval_result = eval_bt.run(test_bars, test_news)
    print(f"OUT-OF-SAMPLE: {eval_result.summary()}")

    if eval_result.trades:
        wins = [t for t in eval_result.trades if t.outcome == "tp"]
        losses = [t for t in eval_result.trades if t.outcome == "sl"]
        avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
        avg_loss_r = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0
        longs = sum(1 for t in eval_result.trades if t.direction > 0)
        shorts = len(eval_result.trades) - longs
        print(f"  longs={longs}  shorts={shorts}  "
              f"avg_win={avg_win_r:+.2f}R  avg_loss={avg_loss_r:+.2f}R")
        print("  first 5 trades:")
        for t in eval_result.trades[:5]:
            side = "LONG " if t.direction > 0 else "SHORT"
            print(f"    bar {t.bar_in:>4}->{t.bar_out:<4} {side} "
                  f"entry={t.entry:.2f} exit={t.exit:.2f} "
                  f"R={t.r_multiple:+.2f} pnl=${t.pnl:+.2f}")


if __name__ == "__main__":
    main()
