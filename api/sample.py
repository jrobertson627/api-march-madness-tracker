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
    return Bundle(model, snap, pairs, pd.DataFrame(team_rows), {"teams": field, "final_four": [["East", "West"], ["South", "Midwest"]]})
