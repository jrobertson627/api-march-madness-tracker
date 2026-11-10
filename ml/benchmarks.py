"""Score outside predictions (betting lines or ratings) the same way as our models.

The file needs one row per game with `season`, `game_date`, `team_a`, `team_b`
(the same team ids as the matchup file, canonical_source_path) and EITHER
`p_a_win` (probability team A wins) OR `spread_a` (points team A is favored by,
positive when A is the favorite). Rows may list the teams in either order; they
are put in the canonical order (team_a is the smaller id) here.

Lines are not in the database yet, so this is tested on simulated lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import metrics
from ml.models import normal_cdf

KEY = ["season", "game_date", "team_a", "team_b"]


def spread_to_prob(spread_a, sigma: float) -> np.ndarray:
    """P(A wins) when A is favored by `spread_a` and results scatter around the spread with std `sigma`."""
    return normal_cdf(np.asarray(spread_a, float) / sigma)


def estimate_sigma(spread_a, margin_a) -> float:
    """Spread of (actual margin - spread). Estimate it on seasons other than the ones being scored."""
    return float(np.std(np.asarray(margin_a, float) - np.asarray(spread_a, float)))


def load_benchmark(df_or_path, sigma: float = 11.0) -> pd.DataFrame:
    """Read a benchmark file and return canonical rows with p_bench. `sigma` is only used for spreads."""
    df = pd.read_csv(df_or_path, parse_dates=["game_date"]) if not isinstance(df_or_path, pd.DataFrame) else df_or_path.copy()
    if "p_a_win" in df:
        p = df["p_a_win"].astype(float)
    elif "spread_a" in df:
        p = pd.Series(spread_to_prob(df["spread_a"], sigma), index=df.index)
    else:
        raise ValueError("benchmark needs a p_a_win or spread_a column")
    flip = df["team_a"] > df["team_b"]
    out = df[["season", "game_date"]].copy()
    out["team_a"] = np.where(flip, df["team_b"], df["team_a"])
    out["team_b"] = np.where(flip, df["team_a"], df["team_b"])
    out["p_bench"] = np.where(flip, 1 - p, p)
    return out.drop_duplicates(KEY)


def score_benchmark(bench: pd.DataFrame, ncaa: pd.DataFrame) -> dict:
    """Metrics of the benchmark on the NCAA games it covers, plus how much of the matrix it covers."""
    m = ncaa.merge(bench, on=KEY, how="inner")
    out = {"games": len(m), "coverage": len(m) / max(len(ncaa), 1)}
    if len(m):
        out |= {"log_loss": metrics.log_loss(m["p_bench"], m["a_win"]), "brier": metrics.brier(m["p_bench"], m["a_win"]),
                "accuracy": metrics.accuracy(m["p_bench"], m["a_win"])}
    return out


def compare_to_model(bench: pd.DataFrame, oof: pd.DataFrame, n: int = 10_000) -> dict:
    """Paired comparison of a model's out-of-fold predictions against the benchmark on shared games.
    `oof` needs game_date, team_a, team_b, season, p and a_win."""
    if "team_a" not in oof:  # game_id is "season|date|team_a|team_b"
        parts = oof["game_id"].str.split("|", expand=True)
        oof = oof.assign(game_date=pd.to_datetime(parts[1]), team_a=parts[2], team_b=parts[3])
    m = oof.merge(bench, on=KEY, how="inner")
    if m.empty:
        return {"games": 0}
    r = metrics.paired_bootstrap(metrics.game_log_loss(m["p"], m["a_win"]), metrics.game_log_loss(m["p_bench"], m["a_win"]), n=n)
    return {"games": len(m), "model_log_loss": metrics.log_loss(m["p"], m["a_win"]),
            "bench_log_loss": metrics.log_loss(m["p_bench"], m["a_win"]), **r}
