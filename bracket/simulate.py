"""Win odds for every team in every round.

`exact_levels` computes them exactly for independent games. `simulate` samples brackets, optionally with a random
strength shift per team (`tau`) and with some games already decided (`fixed`). Both return `Levels`: for each level
l, an array (games, teams) holding P(team wins game j of level l). Level 0 is the slots after the First Four, so
level 0 is also "who is in round 1". Summing a level over its games gives each team's chance of being in the next
round, and the last level gives the title chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bracket.structure import REACH_COLUMNS, Bracket

P_CLIP = 1e-6


@dataclass
class Levels:
    arrays: list            # arrays[l]: (games at level l, teams)
    n_sims: int | None      # None when computed exactly
    tau: float = 0.0

    def reach(self) -> np.ndarray:
        """(teams, rounds + 1): chance of playing in round 1 ... round n, then of winning the title."""
        return np.column_stack([a.sum(axis=0) for a in self.arrays])

    def reach_se(self) -> np.ndarray:
        r = self.reach()
        return np.zeros_like(r) if self.n_sims is None else np.sqrt(r * (1 - r) / self.n_sims)


def check_matrix(P, n_teams: int) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.shape != (n_teams, n_teams):
        raise ValueError(f"probability matrix is {P.shape}, expected {(n_teams, n_teams)}")
    if np.isnan(P).any() or (P < 0).any() or (P > 1).any():
        raise ValueError("probability matrix has missing values or values outside [0, 1]")
    off = ~np.eye(n_teams, dtype=bool)
    if not np.allclose((P + P.T)[off], 1.0, atol=1e-6):
        raise ValueError("probability matrix is not symmetric: P[a, b] + P[b, a] must equal 1")
    return P


def fixed_matrix(bracket: Bracket, fixed) -> np.ndarray | None:
    """int8 matrix F: F[a, b] = 1 if a is known to beat b, -1 if b beat a, 0 if not played. `fixed` is a table or
    list of (team_a, team_b, winner) team ids."""
    if fixed is None or len(fixed) == 0:
        return None
    rows = fixed.itertuples(index=False) if isinstance(fixed, pd.DataFrame) else iter(fixed)
    idx = bracket.index
    F = np.zeros((len(bracket.teams),) * 2, dtype=np.int8)
    for row in rows:
        a, b, w = (idx[row[0]], idx[row[1]], idx[row[2]]) if not hasattr(row, "winner") else (idx[row.team_a], idx[row.team_b], idx[row.winner])
        if w not in (a, b):
            raise ValueError(f"winner {bracket.teams[w]} was not in the game {bracket.teams[a]} v {bracket.teams[b]}")
        F[a, b], F[b, a] = (1, -1) if w == a else (-1, 1)
    return F


def exact_levels(bracket: Bracket, P, fixed=None) -> Levels:
    """Exact odds for independent games. With `fixed`, games already played have their real winners."""
    T = len(bracket.teams)
    P = check_matrix(P, T).copy()
    F = fixed_matrix(bracket, fixed)
    if F is not None:
        P[F == 1], P[F == -1] = 1.0, 0.0
    level0 = np.zeros((bracket.n_slots, T))
    for k, cand in enumerate(bracket.slots):
        if len(cand) == 1:
            level0[k, cand[0]] = 1.0
        else:
            a, b = cand
            level0[k, a], level0[k, b] = P[a, b], P[b, a]
    arrays = [level0]
    for _ in range(bracket.n_rounds):
        prev = arrays[-1]
        left, right = prev[0::2], prev[1::2]
        arrays.append(left * (right @ P.T) + right * (left @ P.T))
    return Levels(arrays, None, 0.0)


def _play(a, b, rng, P, Lg, delta, F):
    if delta is None:
        p = P[a, b]
    else:
        z = Lg[a, b] + np.take_along_axis(delta, a, axis=1) - np.take_along_axis(delta, b, axis=1)
        p = 1.0 / (1.0 + np.exp(-z))
    win_a = rng.random(a.shape) < p
    if F is not None:
        f = F[a, b]
        win_a = np.where(f == 1, True, np.where(f == -1, False, win_a))
    return np.where(win_a, a, b)


def simulate(bracket: Bracket, P, n_sims: int = 100_000, tau: float = 0.0, fixed=None, seed: int = 0, chunk: int = 100_000) -> Levels:
    """Monte Carlo odds. `tau` is the spread of a random rating shift per team and simulation, in logit units;
    0 means independent games. Games in `fixed` always go to their real winner."""
    T = len(bracket.teams)
    P = check_matrix(P, T)
    if tau < 0:
        raise ValueError("tau must be zero or positive")
    F = fixed_matrix(bracket, fixed)
    Pc = np.clip(P, P_CLIP, 1 - P_CLIP)
    Lg = np.log(Pc / (1 - Pc))
    rng = np.random.default_rng(seed)
    S, R = bracket.n_slots, bracket.n_rounds
    counts = [np.zeros((S >> l, T)) for l in range(R + 1)]
    single = np.array([c[0] for c in bracket.slots])
    done = 0
    while done < n_sims:
        n = min(chunk, n_sims - done)
        done += n
        delta = rng.normal(0.0, tau, (n, T)) if tau > 0 else None
        cur = np.tile(single, (n, 1))
        for k in bracket.playin_slots:
            a, b = bracket.slots[k]
            cur[:, k] = _play(np.full((n, 1), a), np.full((n, 1), b), rng, P, Lg, delta, F)[:, 0]
        for lvl in range(R + 1):
            if lvl > 0:
                cur = _play(cur[:, 0::2], cur[:, 1::2], rng, P, Lg, delta, F)
            for j in range(cur.shape[1]):
                counts[lvl][j] += np.bincount(cur[:, j], minlength=T)
    return Levels([c / n_sims for c in counts], n_sims, float(tau))


def reach_table(bracket: Bracket, levels: Levels) -> pd.DataFrame:
    """One row per team: seed, region, and the chance of being in each round and of winning the title."""
    reach, se = levels.reach(), levels.reach_se()
    names = list(REACH_COLUMNS) if bracket.n_rounds == 6 else [f"p_round_{i + 1}" for i in range(bracket.n_rounds)] + ["p_champion"]
    out = pd.DataFrame({"team": bracket.teams, "seed": bracket.seed, "region": bracket.region})
    for i, name in enumerate(names):
        out[name] = reach[:, i]
    out["se_champion"] = se[:, -1]
    return out.sort_values("p_champion", ascending=False).reset_index(drop=True)
