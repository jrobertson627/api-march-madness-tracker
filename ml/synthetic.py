"""Simulated matchup rows with a known skill-to-outcome relationship.

Used by tests and by docs/modeling.md so the modeling code can be checked
without the real matrix. Columns mirror the Phase 1 matchup file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(
    seasons=(2022, 2023, 2024, 2025),
    n_regular: int = 500,
    n_conf: int = 60,
    n_ncaa: int = 40,
    seed: int = 0,
    slope: float = 0.7,
    noise: float = 11.0,
) -> pd.DataFrame:
    """margin_a = slope * skill_gap + 3 * loc_a + noise. Team A wins when margin_a > 0."""
    rng = np.random.default_rng(seed)
    parts = []
    for season in seasons:
        for group, n in (("regular", n_regular), ("conf_tourney", n_conf), ("ncaa", n_ncaa)):
            gap = rng.normal(0, 8, n)
            loc = rng.choice([1, -1, 0], n, p=[0.4, 0.4, 0.2]) if group == "regular" else np.zeros(n)
            margin = slope * gap + 3.0 * loc + rng.normal(0, noise, n)
            margin = np.where(margin == 0, 1.0, margin)
            ot = ((np.abs(margin) <= 2) & (rng.random(n) < 0.6)).astype(int)
            total = 140 + rng.normal(0, 12, n) + 5 * ot
            elo_gap = 20 * gap + rng.normal(0, 40, n)
            tempo_a, tempo_b = 68 + rng.normal(0, 3, n), 68 + rng.normal(0, 3, n)
            off_a, off_b = 105 + rng.normal(0, 4, n), 105 + rng.normal(0, 4, n)
            df = pd.DataFrame(
                {
                    "season": season,
                    "game_group": group,
                    "loc_a": loc,
                    "margin_a": margin,
                    "overtimes": ot,
                    "total": total,
                    "diff_adj_em": gap + rng.normal(0, 1, n),
                    "diff_elo": elo_gap,
                    "diff_sos_elo": rng.normal(0, 30, n),
                    "diff_last10_margin": 0.3 * gap + rng.normal(0, 4, n),
                    "diff_form_trend": rng.normal(0, 3, n),
                    "diff_win_pct": 0.01 * gap + rng.normal(0, 0.05, n),
                    "proj_margin": slope * gap + 3.0 * loc,
                    "proj_total": 140 + rng.normal(0, 4, n),
                    "elo_expected_a": 1 / (1 + 10 ** (-(elo_gap + 90 * loc) / 400)),
                    "seed_a": np.where(group == "ncaa", np.clip(8 - np.round(gap / 3), 1, 16), np.nan),
                    "seed_b": np.where(group == "ncaa", 8.0, np.nan),
                    "prior_season_available_a": 1,
                    "adj_off_a": off_a, "adj_off_b": off_b,
                    "adj_def_a": 100 + rng.normal(0, 4, n), "adj_def_b": 100 + rng.normal(0, 4, n),
                    "adj_tempo_a": tempo_a, "adj_tempo_b": tempo_b,
                    "tempo_a": tempo_a, "tempo_b": tempo_b,
                }
            )
            df["game_date"] = pd.Timestamp(f"{season}-01-01") + pd.to_timedelta(np.arange(n) % 90, unit="D")
            df["as_of"] = df["game_date"] - pd.Timedelta(days=1)
            parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["diff_seed"] = out["seed_a"] - out["seed_b"]
    out["a_win"] = (out["margin_a"] > 0).astype(int)
    out["score_a"] = (out["total"] + out["margin_a"]) / 2
    out["score_b"] = (out["total"] - out["margin_a"]) / 2
    out["trainable"] = 1
    out["both_have_features"] = 1
    out["game_id"] = [f"g{i}" for i in range(len(out))]
    out["team_a"] = [f"a{i}" for i in range(len(out))]
    out["team_b"] = [f"b{i}" for i in range(len(out))]
    from ml.dataset import add_derived

    return add_derived(out)
