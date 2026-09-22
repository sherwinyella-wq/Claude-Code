"""Tabular Q-learning agent.

The agent observes a discretized state composed of:

  * trend bucket          (down / sideways / up)
  * pattern bucket        (strong-bear / weak-bear / none / weak-bull / strong-bull)
  * RSI zone              (oversold / neutral / overbought)
  * news sentiment zone   (negative / neutral / positive)
  * world risk level      (calm / tense / crisis)
  * holding flag          (flat / in trade)

…and chooses among three actions: hold, enter long, enter short.
Trade exits are driven by the RiskManager (fixed stop/target), not by
the agent — the agent's job is purely to find good entries.

Learning is standard Q-learning with epsilon-greedy exploration.
Trade outcomes (in R-multiples) are used as the reward signal so the
agent inherently learns the user's 1:2 reward profile: a winner is
+2R-equivalent worth of expectancy and a loser is -1R.
"""

import json
import random
from typing import Dict, Tuple


Action = int
HOLD, LONG, SHORT = 0, 1, 2
ACTIONS = (HOLD, LONG, SHORT)


def bucket_trend(t: float) -> int:
    if t < -0.15:
        return 0
    if t > 0.15:
        return 2
    return 1


def bucket_pattern(p: float) -> int:
    """Buckets aligned with the RiskManager gate (|signal| >= 0.50):
    0 = strong bear (gate passes short), 1 = weak bear (gate blocks),
    2 = neutral (gate blocks), 3 = weak bull (gate blocks),
    4 = strong bull (gate passes long).
    """
    if p <= -0.50:
        return 0
    if p <= -0.15:
        return 1
    if p < 0.15:
        return 2
    if p < 0.50:
        return 3
    return 4


def bucket_rsi(r: float) -> int:
    if r != r:  # NaN guard
        return 1
    if r < 30:
        return 0
    if r > 70:
        return 2
    return 1


def bucket_sentiment(s: float) -> int:
    if s < -0.2:
        return 0
    if s > 0.2:
        return 2
    return 1


State = Tuple[int, int, int, int, int, int]


def build_state(
    trend: float,
    pattern_signal: float,
    rsi_value: float,
    sentiment: float,
    world_risk: int,
    in_trade: bool,
) -> State:
    return (
        bucket_trend(trend),
        bucket_pattern(pattern_signal),
        bucket_rsi(rsi_value),
        bucket_sentiment(sentiment),
        max(0, min(2, world_risk)),
        1 if in_trade else 0,
    )


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        seed: int = 0,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q: Dict[State, Dict[Action, float]] = {}
        self._rng = random.Random(seed)

    def _row(self, state: State) -> Dict[Action, float]:
        row = self.q.get(state)
        if row is None:
            # Optimistic init for entry actions encourages exploration:
            # the agent will try LONG/SHORT enough times in each state to
            # learn which ones actually produce positive expectancy.
            row = {HOLD: 0.0, LONG: 0.5, SHORT: 0.5}
            self.q[state] = row
        return row

    def act(self, state: State, training: bool = True) -> Action:
        # If already in a trade, the agent must wait — exits are managed
        # by the RiskManager. Encoding this rule here keeps the Q-table
        # focused on the entry decision.
        if state[5] == 1:
            return HOLD
        if training and self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        row = self._row(state)
        return max(row, key=row.get)

    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        row = self._row(state)
        next_row = self._row(next_state)
        target = reward + self.gamma * max(next_row.values())
        row[action] += self.alpha * (target - row[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        serializable = {
            ",".join(str(x) for x in k): v for k, v in self.q.items()
        }
        with open(path, "w") as f:
            json.dump({"epsilon": self.epsilon, "q": serializable}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.q = {
            tuple(int(x) for x in k.split(",")): {int(a): v for a, v in row.items()}
            for k, row in data["q"].items()
        }
