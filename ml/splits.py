"""Train/test splits. Everything is by season, so a game and its swapped copy
always land on the same side, and a held-out season contributes no rows of any
game type to training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

TUNE_GROUPS = ("ncaa", "conf_tourney")


@dataclass(frozen=True)
class Split:
    name: str
    season: int
    train: np.ndarray  # positional row indexes into the frame
    test: np.ndarray


def _in_pool(df: pd.DataFrame, exclude_seasons) -> np.ndarray:
    return ~df["season"].isin(list(exclude_seasons)).to_numpy()


def loso(df: pd.DataFrame, exclude_seasons=()) -> Iterator[Split]:
    """Leave one season out. Test: that season's NCAA rows. Train: every row of the other seasons."""
    pool = _in_pool(df, exclude_seasons)
    season = df["season"].to_numpy()
    ncaa = (df["game_group"] == "ncaa").to_numpy()
    for s in sorted(np.unique(season[pool])):
        yield Split("loso", int(s), np.flatnonzero(pool & (season != s)), np.flatnonzero(pool & (season == s) & ncaa))


def forward_chain(df: pd.DataFrame, min_train_seasons: int = 2, exclude_seasons=()) -> Iterator[Split]:
    """Train on earlier seasons only, test on the next season's NCAA rows."""
    pool = _in_pool(df, exclude_seasons)
    season = df["season"].to_numpy()
    ncaa = (df["game_group"] == "ncaa").to_numpy()
    seasons = sorted(np.unique(season[pool]))
    for i, s in enumerate(seasons):
        if i < min_train_seasons:
            continue
        yield Split("forward", int(s), np.flatnonzero(pool & np.isin(season, seasons[:i])), np.flatnonzero(pool & (season == s) & ncaa))


def locked_split(df: pd.DataFrame, season: int) -> Split:
    """The locked-box split: train on every other season, test on `season`'s NCAA rows."""
    s = df["season"].to_numpy()
    ncaa = (df["game_group"] == "ncaa").to_numpy()
    return Split("locked", int(season), np.flatnonzero(s != season), np.flatnonzero((s == season) & ncaa))


def inner(df: pd.DataFrame, train_idx: np.ndarray) -> Iterator[Split]:
    """Leave-one-season-out inside a training set. Test: NCAA and conference-tournament rows of the inner season."""
    sub = df.iloc[train_idx]
    season = sub["season"].to_numpy()
    tune = sub["game_group"].isin(TUNE_GROUPS).to_numpy()
    for s in sorted(np.unique(season)):
        yield Split("inner", int(s), train_idx[season != s], train_idx[(season == s) & tune])
