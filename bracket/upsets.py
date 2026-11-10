"""First-round upset flags: where the model disagrees with the seeds.

For each round-of-64 game the underdog is the team with the larger seed number. A game is flagged when the model
gives the underdog at least `pick` (the model would pick the upset) or at least `gap` more than the historical
rate for that seed pairing (the model sees something the seeds do not). The history comes only from other seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bracket.simulate import Levels
from bracket.structure import Bracket

SEED_PAIRS = ((1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15))


def seed_history(games: pd.DataFrame, smoothing: float = 1.0) -> dict:
    """Upset rate per round-of-64 seed pairing. `games` needs seed_a, seed_b, a_win and round; use other seasons
    only. The rate is smoothed toward 0.5 by `smoothing` pseudo-games each way."""
    g = games[games["round"] == 1].dropna(subset=["seed_a", "seed_b"])
    out = {}
    for hi, lo in SEED_PAIRS:
        fav_is_a = (g["seed_a"] == hi) & (g["seed_b"] == lo)
        fav_is_b = (g["seed_b"] == hi) & (g["seed_a"] == lo)
        sel = g[fav_is_a | fav_is_b]
        upsets = int(((fav_is_a[sel.index]) & (sel["a_win"] == 0)).sum() + ((fav_is_b[sel.index]) & (sel["a_win"] == 1)).sum())
        out[(hi, lo)] = (upsets + smoothing) / (len(sel) + 2 * smoothing)
    return out


def first_round_table(bracket: Bracket, levels: Levels, history: dict, pick: float = 0.5, gap: float = 0.15) -> pd.DataFrame:
    """One row per round-of-64 game with the model's and history's chance that the underdog wins, and the flags."""
    if bracket.n_rounds != 6:
        raise ValueError("first_round_table needs a 64-slot bracket")
    win = levels.arrays[1]
    rows = []
    for j in range(32):
        left, right = bracket.slots[2 * j], bracket.slots[2 * j + 1]
        seeds = {k: bracket.seed[bracket.slots[k][0]] for k in (2 * j, 2 * j + 1)}
        fav, dog = sorted((2 * j, 2 * j + 1), key=lambda k: seeds[k])
        pair = (seeds[fav], seeds[dog])
        p_model = float(sum(win[j, t] for t in bracket.slots[dog]))
        p_hist = history.get(pair, np.nan)
        rows.append({
            "region": bracket.region[bracket.slots[fav][0]],
            "favorite_seed": pair[0], "underdog_seed": pair[1],
            "favorite": "/".join(bracket.teams[t] for t in bracket.slots[fav]),
            "underdog": "/".join(bracket.teams[t] for t in bracket.slots[dog]),
            "p_underdog_model": p_model, "p_underdog_history": p_hist,
            "model_picks_upset": p_model >= pick,
            "model_above_history": (p_model - p_hist) >= gap if not np.isnan(p_hist) else False,
            "game": j,
        })
    out = pd.DataFrame(rows)
    out["flagged"] = out["model_picks_upset"] | out["model_above_history"]
    return out


def realized_upsets(bracket: Bracket, actual) -> np.ndarray:
    """For each round-of-64 game, did the underdog (larger seed number) win? Needs `replay` output."""
    out = np.zeros(32, dtype=bool)
    for j in range(32):
        w = actual[1][j]
        s = [bracket.seed[bracket.slots[k][0]] for k in (2 * j, 2 * j + 1)]
        dog_slot = 2 * j + int(np.argmax(s))
        out[j] = w >= 0 and w in bracket.slots[dog_slot] and s[0] != s[1]
    return out
