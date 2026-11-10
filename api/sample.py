"""A small invented set of published predictions, for tests and for trying the API without any real data.

The numbers are made up: a 68-team tournament field plus a few extra teams, with win probabilities from random
strengths. Never serve this as real output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.store import Bundle, ModelVersion, Snapshot

REGIONS = ("East", "West", "South", "Midwest")
GROUPS = ("efficiency", "form", "schedule")


def sample_field() -> pd.DataFrame:
    """68 teams: four regions with seeds 1 to 16 and four First Four games. Team ids are lower-case slugs."""
    rows = []
    for region in REGIONS:
        for s in range(1, 17):
            rows.append((f"{region[0].lower()}{s}", f"{region} {s}", s, region))
    rows += [("e16b", "East 16b", 16, "East"), ("s11b", "South 11b", 11, "South"), ("w16c", "West 16c", 16, "West"), ("m11c", "Midwest 11c", 11, "Midwest")]
    return pd.DataFrame(rows, columns=["team", "name", "seed", "region"])


def sample_bundle(season: int = 2030, model_id: str = "sample-model", in_sample: bool = False, seed: int = 0, extra: int = 4,
                  kind: str = "selection_sunday", as_of: str | None = None) -> Bundle:
    rng = np.random.default_rng(seed)
    field = sample_field()
    extras = pd.DataFrame({"team": [f"x{i}" for i in range(extra)], "name": [f"Extra {i}" for i in range(extra)]})
    teams = sorted(list(field["team"]) + list(extras["team"]))
    strength = dict(zip(teams, rng.normal(0, 1.2, len(teams))))
    rows = []
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            p = 1 / (1 + np.exp(-(strength[a] - strength[b])))
            margin = float(np.log(p / (1 - p)) * 6.0)
            total = 140.0
            eff = 0.6 * (p - 0.5)
            form = 0.25 * (p - 0.5)
            sched = 0.1 * (p - 0.5)
            factors = {"efficiency": eff, "form": form, "schedule": sched, "interaction": (p - 0.5) - eff - form - sched}
            rows.append((a, b, float(p), margin, total, (total + margin) / 2, (total - margin) / 2, factors))
    pairs = pd.DataFrame(rows, columns=["team_a", "team_b", "p_a_win", "margin_a", "total", "score_a", "score_b", "factors"])
    names = {**dict(zip(field["team"], field["name"])), **dict(zip(extras["team"], extras["name"]))}
    team_rows = [{"team": t, "name": names[t], "values": {"adj_em": round(float(strength[t]) * 8, 2), "adj_tempo": 68.0 + float(strength[t]),
                                                           "last10_margin": round(float(strength[t]) * 3, 2), "sos_elo": 1500 + 20 * float(strength[t]),
                                                           "elo": 1500 + 100 * float(strength[t])}} for t in teams]
    as_of = as_of or f"{season}-03-15"
    model = ModelVersion(model_id, "sample", "run-" + model_id, "backtest" if not in_sample else "live", [2022, 2023] + ([season] if in_sample else []),
                         "hash", "2026-11-10T00:00:00+00:00", {"log_loss": 0.55}, {}, False)
    snap = Snapshot(model_id, season, as_of, kind, in_sample, len(teams),
                    {"seed_history": {"1-16": 0.02, "8-9": 0.5, "5-12": 0.35, "4-13": 0.2, "6-11": 0.4, "3-14": 0.15, "7-10": 0.4, "2-15": 0.1}},
                    "2026-11-10T00:00:00+00:00")
    return Bundle(model, snap, pairs, pd.DataFrame(team_rows), {"teams": field, "final_four": [["East", "West"], ["South", "Midwest"]], "results": sample_results()})


def sample_results() -> list:
    """Three invented results, one of them a First Four game (e16 and e16b share a slot)."""
    return [{"team_a": "s1", "team_b": "s16", "winner": "s1"}, {"team_a": "w8", "team_b": "w9", "winner": "w9"},
            {"team_a": "e16", "team_b": "e16b", "winner": "e16b"}]


def sample_backtest(model_id: str = "sample-model") -> dict:
    """An invented backtest report in the shape the Store returns: {model_id, data_hash, created_at, report}. The numbers are made up."""
    from api import backtest as bt

    scores = pd.DataFrame([(s, m, 0.0, ll, 0.1, 408) for s in (2022, 2023) for m, ll in (("coin", 0.33), ("seed", 0.28), ("eff", 0.27), ("model", 0.26))],
                          columns=["season", "method", "tau", "log_loss", "brier", "events"])
    exp = pd.DataFrame([(name, "loso", "game", season, n, ll, 0.2, acc) for name, ll, acc in (("coin", 0.693, 0.5), ("seed", 0.58, 0.7), ("eff", 0.57, 0.69), ("model", 0.56, 0.7))
                        for season, n in ((2022, 67), (2023, 67), ("all", 134))], columns=["name", "protocol", "mode", "season", "n", "log_loss", "brier", "accuracy"])
    specs = [("coin flip", "coin"), ("seed", "seed"), ("efficiency gap", "eff"), ("model", "model")]
    flags = pd.DataFrame([(2022, "East", "e1", "e16", 1, 16, 0.05, False, False, False, False), (2022, "East", "e8", "e9", 8, 9, 0.52, True, False, True, True),
                          (2023, "West", "w5", "w12", 5, 12, 0.30, False, True, True, False), (2023, "West", "w6", "w11", 6, 11, 0.35, False, False, False, False)],
                         columns=["season", "region", "favorite", "underdog", "favorite_seed", "underdog_seed", "p_underdog_model", "model_picks_upset", "model_above_history", "flagged", "upset"])
    rows, summary = bt.upset_flags(flags, {"e1": "East 1", "e16": "East 16"})
    pair = bt.paired_row({"games": 134, "mean_diff": -0.012, "lo": -0.03, "hi": 0.004, "season_wins": 2, "seasons": 2}, "model", "efficiency gap")
    picks = [{"season": 2022, "strategy": "best expected points", "realized_points": 520.0, "expected_points": 760.5},
             {"season": 2022, "strategy": "model favorite", "realized_points": 580.0, "expected_points": 745.2}]
    locked = {"season": 2024, "advancement": bt.advancement_rows(scores[scores["season"] == 2023].assign(season=2024)),
              "games_scored": [dict(r, season=2024) for r in bt.game_rows(exp, specs) if r["season"] == 2023],
              "picks": bt.pick_rows([dict(p, season=2024) for p in picks]), "paired": [pair]}
    report = bt.build_report(seasons=[2022, 2023], advancement=bt.advancement_rows(scores), games_scored=bt.game_rows(exp, specs), picks=bt.pick_rows(picks),
                             flags=rows, flag_summary=summary, paired=[pair], locked=locked)
    return {"model_id": model_id, "data_hash": "samplehash", "created_at": "2026-11-10T00:00:00+00:00", "report": report}
