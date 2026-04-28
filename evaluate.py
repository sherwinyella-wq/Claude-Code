"""Aggregate evaluation across many random seeds.

Trains a fresh agent for each seed and records the out-of-sample
metrics. Prints per-seed results and a summary so we can see the
success rate over many independent market samples — one run is just
one observation.
"""

import argparse
import statistics

from trading_agent import (
    QLearningAgent,
    RiskManager,
    Backtester,
    generate_market,
    generate_news_stream,
)


def run_one(seed: int, epochs: int, bars: int, start_equity: float) -> dict:
    bars_data, regimes = generate_market(n_bars=bars, seed=seed)
    news = generate_news_stream(regimes, seed=seed + 1)
    split = int(len(bars_data) * 0.7)
    train_bars, test_bars = bars_data[:split], bars_data[split:]
    train_news, test_news = news[:split], news[split:]

    agent = QLearningAgent(seed=seed)
    for _ in range(epochs):
        risk = RiskManager(account_equity=start_equity)
        Backtester(agent, risk, training=True).run(train_bars, train_news)

    eval_risk = RiskManager(account_equity=start_equity)
    result = Backtester(agent, eval_risk, training=False).run(test_bars, test_news)

    return {
        "seed": seed,
        "trades": len(result.trades),
        "win_rate": result.win_rate,
        "expectancy_r": result.expectancy_r,
        "return_pct": result.total_return_pct,
        "max_dd": result.max_drawdown_pct,
        "wins": sum(1 for t in result.trades if t.outcome == "tp"),
        "losses": sum(1 for t in result.trades if t.outcome == "sl"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bars", type=int, default=4000)
    p.add_argument("--start-equity", type=float, default=10_000.0)
    args = p.parse_args()

    print(f"Evaluating across {args.seeds} seeds  "
          f"(epochs={args.epochs}, bars={args.bars})\n")
    print(f"{'seed':>5}  {'trades':>6}  {'win%':>6}  "
          f"{'exp':>7}  {'ret%':>8}  {'maxdd%':>7}")
    print("-" * 50)

    results = []
    for s in range(args.seeds):
        r = run_one(s, args.epochs, args.bars, args.start_equity)
        results.append(r)
        print(f"{r['seed']:>5}  {r['trades']:>6}  {r['win_rate']:>5.1f}  "
              f"{r['expectancy_r']:>+7.2f}R  {r['return_pct']:>+8.2f}  {r['max_dd']:>6.2f}")

    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_losses = sum(r["losses"] for r in results)
    aggregate_win_rate = total_wins / total_trades * 100 if total_trades else 0.0

    returns = [r["return_pct"] for r in results]
    expectancies = [r["expectancy_r"] for r in results if r["trades"] > 0]
    profitable_seeds = sum(1 for r in results if r["return_pct"] > 0)

    print("-" * 50)
    print(f"\nAGGREGATE OVER {len(results)} SEEDS")
    print(f"  total trades:        {total_trades}  ({total_wins}W / {total_losses}L)")
    print(f"  pooled win rate:     {aggregate_win_rate:.1f}%")
    print(f"  avg expectancy:      {statistics.mean(expectancies):+.2f}R" if expectancies else "  no trades")
    print(f"  avg return:          {statistics.mean(returns):+.2f}%")
    print(f"  median return:       {statistics.median(returns):+.2f}%")
    print(f"  stdev of returns:    {statistics.pstdev(returns):.2f}%")
    print(f"  profitable seeds:    {profitable_seeds}/{len(results)}  "
          f"({profitable_seeds/len(results)*100:.0f}%)")
    print(f"  best seed:           {max(returns):+.2f}%")
    print(f"  worst seed:          {min(returns):+.2f}%")

    breakeven_wr = 100 / 3
    print(f"\n  breakeven win rate at 1:2 R:R = {breakeven_wr:.1f}%")
    if aggregate_win_rate > breakeven_wr:
        edge = aggregate_win_rate - breakeven_wr
        print(f"  agent's edge:               +{edge:.1f} pp above breakeven")
    else:
        edge = breakeven_wr - aggregate_win_rate
        print(f"  agent below breakeven by:   -{edge:.1f} pp")


if __name__ == "__main__":
    main()
