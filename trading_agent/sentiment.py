"""Lightweight world-events / news sentiment scoring.

A real system would plug in an LLM or a fine-tuned classifier here. For
this learner we use a keyword-weighted scorer so the agent has a real,
inspectable signal to consume. Two outputs are exposed:

  * `score_headlines(headlines)` -> sentiment in [-1, 1]
  * `world_risk_level(headlines)` -> macro risk in {0=calm, 1=tense, 2=crisis}
"""

from typing import Iterable, List, Sequence


_BULLISH_TERMS = {
    "rally": 1.0, "surge": 1.0, "beats": 0.8, "record high": 1.0,
    "growth": 0.6, "expands": 0.6, "upgrade": 0.7, "strong demand": 0.8,
    "rate cut": 0.9, "stimulus": 0.8, "ceasefire": 0.7, "deal signed": 0.7,
    "profit": 0.5, "approves": 0.4,
}

_BEARISH_TERMS = {
    "crash": -1.0, "plunge": -1.0, "miss": -0.7, "recession": -1.0,
    "layoffs": -0.6, "downgrade": -0.7, "bankruptcy": -1.0,
    "rate hike": -0.7, "inflation": -0.5, "war": -1.0, "sanctions": -0.7,
    "shutdown": -0.6, "default": -0.9, "scandal": -0.6, "fraud": -0.8,
    "tariff": -0.5,
}

_RISK_TERMS = {
    "war", "invasion", "sanctions", "pandemic", "default", "shutdown",
    "crisis", "attack", "bankruptcy", "recession",
}


def _tokenize(s: str) -> str:
    return s.lower()


def score_headlines(headlines: Iterable[str]) -> float:
    headlines = list(headlines)
    if not headlines:
        return 0.0
    total = 0.0
    weight = 0.0
    for h in headlines:
        text = _tokenize(h)
        local = 0.0
        hits = 0
        for term, w in _BULLISH_TERMS.items():
            if term in text:
                local += w
                hits += 1
        for term, w in _BEARISH_TERMS.items():
            if term in text:
                local += w
                hits += 1
        if hits:
            total += local / hits
            weight += 1
    if weight == 0:
        return 0.0
    return max(-1.0, min(1.0, total / weight))


def world_risk_level(headlines: Iterable[str]) -> int:
    hits = 0
    for h in headlines:
        text = _tokenize(h)
        for term in _RISK_TERMS:
            if term in text:
                hits += 1
                break
    if hits == 0:
        return 0
    if hits <= 2:
        return 1
    return 2
