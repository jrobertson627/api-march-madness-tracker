"""Win probabilities for every pair of teams in a bracket, from a Phase 2 model.

Each pair becomes a hypothetical neutral-court NCAA game as of Selection Sunday, built with the same code that
makes training rows (bracket mode), and the model predicts it. A team with no features gets a fallback probability
from its seed, and is reported so nothing is hidden.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from bracket.simulate import check_matrix
from bracket.structure import Bracket
from data.features import build_matchups
from ml.dataset import add_derived
from ml.models import LogisticModel


def seed_fallback(train: pd.DataFrame):
    """A function f(seed_a, seed_b) -> P(a wins) from a seed-gap logistic fit on the NCAA rows of `train`
    (rows of the season being predicted must not be in it)."""
    ncaa = train[(train["game_group"] == "ncaa") & train["diff_seed"].notna()]
    model = LogisticModel(C=10.0).fit(ncaa[["diff_seed"]], ncaa["a_win"].to_numpy())

    def f(seed_a, seed_b) -> float:
        return float(model.predict_proba(pd.DataFrame({"diff_seed": [float(seed_a) - float(seed_b)]}))[0])

    return f


def pair_matrix(predictor, team_asof: pd.DataFrame, bracket: Bracket, season: int, as_of, fallback=None):
    """Return (P, fallback_teams). P[a, b] is the chance team a beats team b, in the bracket's team order."""
    as_of = pd.Timestamp(as_of)
    ta = team_asof[(team_asof["season"] == season) & (team_asof["as_of"] == as_of)]
    have = set(ta["team"])
    T = len(bracket.teams)
    P = np.full((T, T), 0.5)
    ids = [t for t in bracket.teams if t in have]
    pos = bracket.index
    if len(ids) >= 2:
        pairs = pd.DataFrame(list(itertools.combinations(sorted(ids), 2)), columns=["team_a", "team_b"])
        pairs = pairs.assign(season=season, as_of=as_of, game_date=as_of + pd.Timedelta(days=1), game_group="ncaa", loc_a=0.0, overtimes=0)
        rows = add_derived(build_matchups(pairs, team_asof, min_games=0, as_of_mode="bracket"))
        p = predictor.predict(rows)["p_a_win"].to_numpy()
        a = pairs["team_a"].map(pos).to_numpy()
        b = pairs["team_b"].map(pos).to_numpy()
        P[a, b], P[b, a] = p, 1.0 - p
    missing = [t for t in bracket.teams if t not in have]
    if missing:
        if fallback is None:
            raise ValueError(f"no features at {as_of.date()} for {missing}; pass a fallback")
        for t in missing:
            i = pos[t]
            for j in range(T):
                if j != i:
                    P[i, j] = fallback(bracket.seed[i], bracket.seed[j])
                    P[j, i] = 1.0 - P[i, j]
    np.fill_diagonal(P, 0.5)
    return check_matrix(P, T), missing


def seed_matrix(bracket: Bracket, fallback) -> np.ndarray:
    """The probability matrix from seeds alone: the baseline a model has to beat."""
    T = len(bracket.teams)
    P = np.full((T, T), 0.5)
    for i in range(T):
        for j in range(T):
            if i < j:
                P[i, j] = fallback(bracket.seed[i], bracket.seed[j])
                P[j, i] = 1.0 - P[i, j]
    return check_matrix(P, T)
