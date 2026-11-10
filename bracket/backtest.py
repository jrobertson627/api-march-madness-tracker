"""Backtest the bracket simulator on past tournaments.

For each season the win probabilities come from a model trained on other seasons only. The odds are scored on what
happened: every team and round is a yes/no event (did the team reach this round), so a tournament gives about 400
events. Events within a bracket depend on each other, so the intervals one might compute from them are optimistic;
the per-season numbers are the honest unit and there are only five of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bracket import pick, upsets
from bracket.simulate import Levels, exact_levels, simulate
from bracket.structure import Bracket
from ml import metrics

POOL = (2022, 2023, 2024, 2025)  # seasons used to choose settings; 2026 is scored once at the end
TAU_GRID = (0.0, 0.1, 0.2, 0.3)


@dataclass
class Tournament:
    season: int
    bracket: Bracket
    results: pd.DataFrame     # team_a, team_b, winner, round
    actual: list              # winners per level, from Bracket.replay
    realized: np.ndarray      # (teams, rounds + 1) 0/1: reached round 1 ... reached the last round, then won the title


def build_tournament(season: int, teams: pd.DataFrame, results: pd.DataFrame) -> Tournament:
    """teams: team, seed, region. results: team_a, team_b, winner, round for every game of the tournament."""
    bracket = Bracket.from_results(teams, results)
    actual, problems = bracket.replay(results)
    if problems:
        raise ValueError(f"{season}: the bracket structure does not match the real games: {problems[:3]}")
    return Tournament(season, bracket, results, actual, bracket.realized_reach(actual))


def events(reach: np.ndarray, realized: np.ndarray):
    """Flatten the odds and outcomes for rounds 2 onward (reaching round 1 is almost always certain)."""
    return reach[:, 1:].ravel(), realized[:, 1:].ravel()


def event_scores(reach: np.ndarray, realized: np.ndarray) -> dict:
    p, y = events(reach, realized)
    return {"log_loss": metrics.log_loss(p, y), "brier": metrics.brier(p, y), "events": int(len(y))}


def choose_tau(loss: pd.DataFrame, min_seasons_better: float = 0.75) -> dict:
    """`loss` has one row per season and one column per tau (the event log loss). Pick the tau with the lowest mean
    loss, but keep it only if it beats tau = 0 in at least `min_seasons_better` of the seasons and passes a nested
    check (choose tau on the other seasons, score it on the held-out one, and beat tau = 0 on average); otherwise 0."""
    base = loss[0.0]
    best = loss.mean().idxmin()
    wins = int((loss[best] < base).sum())
    need = math.ceil(min_seasons_better * len(loss))
    nested = []
    for s in loss.index:
        rest = loss.drop(index=s)
        t = rest.mean().idxmin()
        nested.append(loss.loc[s, t])
    nested_mean = float(np.mean(nested))
    # Keep a tau above zero only if it wins in enough seasons AND choosing it on the other seasons still beats tau = 0
    # on the held-out one. A tiny average gain that does not survive that check is noise.
    chosen = float(best) if best != 0.0 and wins >= need and nested_mean < float(base.mean()) else 0.0
    return {"tau": chosen, "best_by_mean": float(best), "seasons_better": wins, "needed": need,
            "mean_loss_tau0": float(base.mean()), "mean_loss_chosen": float(loss[chosen].mean()),
            "nested_mean_loss": nested_mean}


def strategy_scores(t: Tournament, P: np.ndarray, levels: Levels, scoring: pick.Scoring = pick.Scoring()) -> pd.DataFrame:
    """Realized and expected points for the three pick strategies."""
    strategies = [pick.seed_bracket(t.bracket), pick.favorite_bracket(t.bracket, P), pick.best_expected_bracket(t.bracket, levels, scoring)]
    return pd.DataFrame([{"season": t.season, "strategy": s.name, "realized_points": pick.score(s, t.actual, scoring),
                          "expected_points": pick.expected_points(s, levels, scoring)} for s in strategies])


def flag_outcomes(t: Tournament, flags: pd.DataFrame) -> pd.DataFrame:
    """Add whether each first-round underdog actually won."""
    return flags.assign(season=t.season, upset=upsets.realized_upsets(t.bracket, t.actual))


def flag_summary(all_flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in (("model picks the upset", all_flags["model_picks_upset"]), ("model above seed history", all_flags["model_above_history"]),
                       ("either flag", all_flags["flagged"]), ("not flagged", ~all_flags["flagged"]), ("all games", np.ones(len(all_flags), bool))):
        sub = all_flags[mask]
        rows.append({"group": name, "games": len(sub), "upsets": int(sub["upset"].sum()), "upset_rate": float(sub["upset"].mean()) if len(sub) else np.nan})
    return pd.DataFrame(rows)


def levels_for(bracket: Bracket, P: np.ndarray, tau: float, n_sims: int, seed: int) -> Levels:
    """Exact odds when games are independent, otherwise simulated."""
    return exact_levels(bracket, P) if tau == 0.0 else simulate(bracket, P, n_sims=n_sims, tau=tau, seed=seed)
