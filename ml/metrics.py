"""Metrics, and the paired bootstrap used to compare two models on the same games."""

from __future__ import annotations

import numpy as np
import pandas as pd

P_CLIP = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)


def game_log_loss(p, y) -> np.ndarray:
    p, y = _clip(p), np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(p, y) -> float:
    return float(game_log_loss(p, y).mean())


def brier(p, y) -> float:
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def accuracy(p, y) -> float:
    """Share of games called right. A probability of exactly 0.5 counts as half."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1)).mean())


def reliability_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predictions: mean prediction, observed rate, games."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, min(bins, len(p)))
    return pd.DataFrame(
        {"mean_p": [p[c].mean() for c in chunks], "frac_win": [y[c].mean() for c in chunks], "games": [len(c) for c in chunks]}
    )


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins. Noisy with a few hundred games."""
    t = reliability_table(p, y, bins)
    return float((t["games"] * (t["mean_p"] - t["frac_win"]).abs()).sum() / t["games"].sum())


def mae(pred, actual) -> float:
    return float(np.abs(np.asarray(pred, float) - np.asarray(actual, float)).mean())


def rmse(pred, actual) -> float:
    return float(np.sqrt(((np.asarray(pred, float) - np.asarray(actual, float)) ** 2).mean()))


def paired_bootstrap(loss_candidate, loss_baseline, n: int = 10_000, seed: int = 0) -> dict:
    """Resample games and compare two models on the same games.

    Inputs are per-game losses. The difference is candidate minus baseline, so
    negative means the candidate is better. Returns the mean difference, the
    95% interval, and the share of resamples where the candidate is better.
    """
    d = np.asarray(loss_candidate, float) - np.asarray(loss_baseline, float)
    m = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    step = max(1, 2_000_000 // m)
    for start in range(0, n, step):
        k = min(step, n - start)
        means[start : start + k] = d[rng.integers(0, m, size=(k, m))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "share_better": float((means < 0).mean())}


def season_wins(loss_candidate, loss_baseline, seasons, margin: float = 0.005) -> int:
    """Seasons where the candidate beats the baseline by at least `margin` log loss."""
    d = pd.DataFrame({"d": np.asarray(loss_candidate, float) - np.asarray(loss_baseline, float), "s": np.asarray(seasons)})
    return int((d.groupby("s")["d"].mean() <= -margin).sum())


def keep_complex(loss_candidate, loss_baseline, seasons, n: int = 10_000, seed: int = 0, margin: float = 0.005) -> dict:
    """The decision rule: keep the more complex model only if the paired 95% interval
    excludes zero, or it wins by at least `margin` in four of five seasons."""
    b = paired_bootstrap(loss_candidate, loss_baseline, n, seed)
    wins = season_wins(loss_candidate, loss_baseline, seasons, margin)
    n_seasons = len(set(np.asarray(seasons).tolist()))
    needed = int(np.ceil(0.8 * n_seasons))
    return {**b, "season_wins": wins, "seasons": n_seasons, "keep": bool(b["hi"] < 0 or wins >= needed)}
