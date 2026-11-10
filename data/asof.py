"""The as-of rule: which date's information a game may use.

A game on date D may use anything that happened before D, earlier tournament
games included, and nothing from D or later. Features are computed on a grid of
as-of dates (`asof_grid`); each game is assigned the last grid date strictly
before it (`asof_for_games`). `assert_no_leak` is the guard the build runs.

Two modes for NCAA tournament games:
  - "game"    (default): same rule as every other game, so a Sweet 16 row sees
              the round-one and round-two results.
  - "bracket": every NCAA row uses Selection Sunday, which is what a bracket
              simulation (Phase 3) needs because later rounds are unplayed.
"""

from __future__ import annotations

import pandas as pd

MODES = ("game", "bracket")


def asof_grid(first_game, cutoff, last_game, cadence: str = "weekly") -> list[pd.Timestamp]:
    """As-of dates for one season.

    weekly: every Sunday from `first_game` through `cutoff` (Selection Sunday),
            then every day after the cutoff through `last_game`.
    daily:  every day from `first_game` through `last_game`.
    """
    first, cutoff, last = (pd.Timestamp(x).normalize() for x in (first_game, cutoff, last_game))
    if cadence == "daily":
        return list(pd.date_range(first, last, freq="D"))
    if cadence != "weekly":
        raise ValueError(f"cadence must be 'weekly' or 'daily', got {cadence!r}")
    sundays = pd.date_range(first, cutoff, freq="W-SUN").union(pd.DatetimeIndex([cutoff]))
    after = pd.date_range(cutoff + pd.Timedelta(days=1), last, freq="D")
    return list(sundays.append(after))


def asof_for_games(games: pd.DataFrame, grid: dict, cutoffs: dict, mode: str = "game") -> pd.Series:
    """As-of date per game: the last grid date strictly before `game_date`.

    `games` needs season, game_date and game_group. `grid` maps season to its
    list of as-of dates and `cutoffs` maps season to Selection Sunday. Games
    earlier than the first grid date get NaT. In bracket mode NCAA games take
    the season's cutoff instead.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if games.empty:
        return pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns]")
    parts = []
    for season, idx in games.groupby("season").groups.items():
        gi = pd.DatetimeIndex(sorted(set(pd.Timestamp(d) for d in grid[season])))
        pos = gi.searchsorted(pd.DatetimeIndex(games.loc[idx, "game_date"]), side="left") - 1
        vals = pd.Series(gi.take(pos.clip(min=0)), index=idx)
        parts.append(vals.where(pos >= 0))
    out = pd.concat(parts).reindex(games.index)
    if mode == "bracket":
        ncaa = games["game_group"] == "ncaa"
        out = out.where(~ncaa, games["season"].map(cutoffs))
    return out


def assert_no_leak(frame: pd.DataFrame, as_of_col: str = "as_of", date_col: str = "game_date") -> None:
    """Raise AssertionError unless every row's as-of date is before its game date."""
    bad = frame[frame[as_of_col] >= frame[date_col]]
    assert bad.empty, f"leak: {len(bad)} rows have {as_of_col} on or after {date_col}"
