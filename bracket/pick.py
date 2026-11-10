"""Bracket picks and scoring.

A pick is a winner for every game from round 1 up. A pick scores when the picked team really wins that game, so a
pick for a team that was knocked out earlier scores nothing. Points are per round: the first number is round 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bracket.simulate import Levels
from bracket.structure import Bracket

DEFAULT_POINTS = (10, 20, 40, 80, 160, 320)


@dataclass(frozen=True)
class Scoring:
    points: tuple = DEFAULT_POINTS


@dataclass
class Picks:
    """winners[l] for l = 1..n_rounds: the picked winner (team position) of each game at that level.
    winners[0] holds the slots after play-ins, as the picks imply them."""

    winners: list
    name: str = ""


def _build(bracket: Bracket, level0: np.ndarray, choose) -> list:
    winners = [level0]
    for _ in range(bracket.n_rounds):
        prev = winners[-1]
        winners.append(np.array([choose(prev[2 * j], prev[2 * j + 1]) for j in range(len(prev) // 2)]))
    return winners


def favorite_bracket(bracket: Bracket, P) -> Picks:
    """Each game goes to the team with the higher model probability against its picked opponent."""
    P = np.asarray(P, float)
    level0 = np.array([c[0] if len(c) == 1 else (c[0] if P[c[0], c[1]] >= 0.5 else c[1]) for c in bracket.slots])
    return Picks(_build(bracket, level0, lambda a, b: a if P[a, b] >= 0.5 else b), "model favorite")


def seed_bracket(bracket: Bracket) -> Picks:
    """Chalk: the better seed always wins; equal seeds go to the lower team position."""
    seed = np.array([s if s is not None else 99 for s in bracket.seed])
    level0 = np.array([c[0] if len(c) == 1 else min(c, key=lambda t: (seed[t], t)) for c in bracket.slots])
    return Picks(_build(bracket, level0, lambda a, b: a if (seed[a], a) <= (seed[b], b) else b), "seed chalk")


def best_expected_bracket(bracket: Bracket, levels: Levels, scoring: Scoring = Scoring()) -> Picks:
    """The consistent bracket with the most expected points, exact for the given win odds.

    Expected points are a sum over games of points * P(picked team wins that game), and a pick must come from the
    two picks feeding the game. So one pass up the tree finds the best value for each team being picked to win each
    game, and one pass down reads the picks off.
    """
    R, T = bracket.n_rounds, len(bracket.teams)
    if len(scoring.points) < R:
        raise ValueError(f"scoring has {len(scoring.points)} rounds, the bracket has {R}")
    best0 = np.full((bracket.n_slots, T), -np.inf)
    for k, cand in enumerate(bracket.slots):
        for t in cand:
            best0[k, t] = 1e-9 * levels.arrays[0][k, t]  # play-ins are not scored; break ties toward the likelier team
    best = [best0]
    for r in range(1, R + 1):
        left, right = best[r - 1][0::2], best[r - 1][1::2]
        gain = scoring.points[r - 1] * levels.arrays[r]
        from_left = gain + left + right.max(axis=1, keepdims=True)
        from_right = gain + right + left.max(axis=1, keepdims=True)
        best.append(np.maximum(from_left, from_right))
    winners = [np.empty(len(b), dtype=int) for b in best]

    def descend(r, j, team):
        winners[r][j] = team
        if r == 0:
            return
        li, ri = 2 * j, 2 * j + 1
        if np.isfinite(best[r - 1][li, team]):
            descend(r - 1, li, team)
            descend(r - 1, ri, int(np.argmax(best[r - 1][ri])))
        else:
            descend(r - 1, ri, team)
            descend(r - 1, li, int(np.argmax(best[r - 1][li])))

    descend(R, 0, int(np.argmax(best[R][0])))
    return Picks(winners, "best expected points")


def expected_points(picks: Picks, levels: Levels, scoring: Scoring = Scoring()) -> float:
    return float(sum(scoring.points[r - 1] * levels.arrays[r][np.arange(len(picks.winners[r])), picks.winners[r]].sum() for r in range(1, len(picks.winners))))


def score(picks: Picks, actual, scoring: Scoring = Scoring()) -> float:
    """Points a bracket earned. `actual[l]` are the real winners per level (from Bracket.replay); unknown games (-1) score 0."""
    total = 0.0
    for r in range(1, len(picks.winners)):
        total += scoring.points[r - 1] * float(((picks.winners[r] == actual[r]) & (actual[r] >= 0)).sum())
    return total
