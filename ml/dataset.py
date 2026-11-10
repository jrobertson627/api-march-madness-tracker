"""Matchup data for modeling: feature sets, training weights, symmetry.

Team A is just the alphabetically earlier id, so a model must not learn from
the ordering. Signed features (differences, location, projected margin) flip
sign when A and B swap. `swap_features` does that, `augment` adds the swapped
copy of every training row, and models average a prediction with the
prediction for the swapped row so p(A, B) + p(B, A) = 1 exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "artifacts" / "features" / "matchups.csv"
BRACKET_PATH = ROOT / "artifacts" / "features" / "matchups_bracket.csv"

SIGNED_EXACT = ("loc_a", "proj_margin")
ONE_MINUS = ("elo_expected_a",)  # a probability for A, so swapping gives 1 - p
MIXES = {  # (regular, conference tournament, NCAA) row weights
    "equal": (1.0, 1.0, 1.0),
    "ncaa_only": (0.0, 0.0, 1.0),
    "tourney_x5": (1.0, 5.0, 5.0),
    "tourney_x20": (1.0, 20.0, 20.0),
}
OT_MINUTES = 5.0
REGULATION_MINUTES = 40.0


def is_signed(col: str) -> bool:
    return col.startswith("diff_") or col in SIGNED_EXACT


def feature_columns(name: str, columns) -> list[str]:
    """Columns for a named feature set, given the columns present in the matrix."""
    diffs = sorted(c for c in columns if c.startswith("diff_") and c != "diff_seed")
    full = [*diffs, "proj_margin", "proj_total", "loc_a", "is_ncaa", "is_conf_tourney", "prior_season_available_a"]
    sets = {
        "seed_only": ["diff_seed"],
        "seed_pair": ["seed_a", "seed_b"],
        "elo_prob": ["elo_expected_a"],
        "elo_core": ["diff_elo", "loc_a"],
        "rating_core": ["diff_adj_em", "loc_a"],
        "ratings_plus": [
            "diff_adj_em", "diff_elo", "diff_sos_elo", "diff_last10_margin", "diff_form_trend",
            "proj_margin", "proj_total", "loc_a",
        ],
        "full": full,
        "full_with_seed": [*full, "diff_seed"],
    }
    if name not in sets:
        raise KeyError(f"unknown feature set {name!r}; choose from {sorted(sets)}")
    missing = [c for c in sets[name] if c not in columns]
    if missing:
        raise KeyError(f"feature set {name!r} needs columns not in the matrix: {missing}")
    return sets[name]


def ot_factor(overtimes) -> np.ndarray:
    """Scale an overtime game's points to a 40-minute game."""
    ot = np.asarray(overtimes, dtype=float)
    return REGULATION_MINUTES / (REGULATION_MINUTES + OT_MINUTES * ot)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_ncaa"] = (df["game_group"] == "ncaa").astype(int)
    df["is_conf_tourney"] = (df["game_group"] == "conf_tourney").astype(int)
    f = ot_factor(df["overtimes"].fillna(0))
    for c in ("total", "score_a", "score_b"):
        df[f"{c}_reg"] = df[c] * f
    for c in ("adj_off", "adj_def", "adj_tempo", "tempo"):
        if f"{c}_a" in df and f"{c}_b" in df:
            df[f"sum_{c}"] = df[f"{c}_a"] + df[f"{c}_b"]
    return df


def load_matchups(path: Path | str = MATRIX_PATH, trainable_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date", "as_of"])
    if trainable_only:
        df = df[df["trainable"] == 1]
    return add_derived(df).reset_index(drop=True)


def data_stamp(path: Path | str = MATRIX_PATH, df: pd.DataFrame | None = None) -> dict:
    """Ties a result to the exact matrix that produced it."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    stamp = {"hash": h}
    if df is not None:
        stamp |= {"rows": int(len(df)), "max_game_date": str(df["game_date"].max().date())}
    return stamp


def swap_features(X: pd.DataFrame) -> pd.DataFrame:
    """The same games with team A and team B exchanged."""
    S = X.copy()
    for c in S.columns:
        if is_signed(c):
            S[c] = -S[c]
        elif c in ONE_MINUS:
            S[c] = 1.0 - S[c]
    return S


def augment(X: pd.DataFrame, y, w=None):
    """Training rows plus their swapped copies (labels flipped, weights repeated)."""
    y = np.asarray(y, dtype=float)
    X2 = pd.concat([X, swap_features(X)], ignore_index=True)
    y2 = np.concatenate([y, 1.0 - y])
    w2 = None if w is None else np.concatenate([np.asarray(w, float), np.asarray(w, float)])
    return X2, y2, w2


def training_weights(df: pd.DataFrame, mix: str) -> np.ndarray:
    if mix not in MIXES:
        raise KeyError(f"unknown mix {mix!r}; choose from {sorted(MIXES)}")
    regular, conf, ncaa = MIXES[mix]
    group = df["game_group"].to_numpy()
    return np.where(group == "ncaa", ncaa, np.where(group == "conf_tourney", conf, regular)).astype(float)
