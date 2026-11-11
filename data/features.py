"""Team features as of a date, and matchup rows built from them.

Everything is a function of (team, as_of): `team_asof` has one row per team per
as-of date, and a matchup row is a join of two teams' rows at the game's as-of
date (see data.asof for how that date is chosen). Compute functions are pure
pandas and filter by date themselves; loaders are the only code that touches
the database.

Naming: a feature `f` appears in a matchup as `f_a`, `f_b` and `diff_f = f_a - f_b`.
Team A is the lexicographically smaller team id, as in the Phase 0 labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import asof as asof_mod
from data import ratings
from data.db import query
from data.snapshots import (
    NCAA_ROUNDS,
    STAT_COLS,
    FTA_POSS,
    compute_player_snapshots,
    compute_roster_features,
    compute_team_snapshots,
    cutoffs_from_games,
    load_player_games,
    load_player_seasons,
    load_team_meta,
    season_start,
)

LOC_SIGN = {"home": 1, "away": -1, "neutral": 0}
GAME_GROUPS = ("ncaa", "conf_tourney", "regular")
DEFAULT_MIN_GAMES = 8
ELO_HOME_ADV = 90.0


# --------------------------------------------------------------------------- #
# feature registry (documents every output column)
# --------------------------------------------------------------------------- #

def _f(group, definition, step, source="game_log_rows", missing="NaN", diff=True):
    return dict(group=group, definition=definition, step=step, source=source, missing=missing, diff=diff)


FEATURES: dict[str, dict] = {
    # record and scoring (Phase 0)
    "games": _f("record", "Games played through as_of, including non-D1 opponents.", "Phase 0", diff=False),
    "d1_games": _f("record", "Games against D1 opponents.", "Phase 0", diff=False),
    "wins": _f("record", "Wins through as_of.", "Phase 0", diff=False),
    "losses": _f("record", "Losses through as_of.", "Phase 0", diff=False),
    "win_pct": _f("record", "wins / games.", "Phase 0"),
    "avg_margin": _f("record", "Mean scoring margin over all games, non-D1 included.", "Phase 0"),
    # raw box-score efficiency (Phase 0)
    "tempo": _f("raw efficiency", "Estimated possessions per box-scored game.", "Phase 0", "box scores"),
    "off_eff": _f("raw efficiency", "Points scored per 100 possessions, not opponent-adjusted.", "Phase 0", "box scores"),
    "def_eff": _f("raw efficiency", "Points allowed per 100 possessions, not opponent-adjusted.", "Phase 0", "box scores"),
    "eff_margin": _f("raw efficiency", "off_eff - def_eff.", "Phase 0", "box scores"),
    "efg": _f("four factors", "Effective field goal percentage.", "p1-04", "box scores"),
    "tov_pct": _f("four factors", "Turnovers per possession-ending play.", "p1-04", "box scores"),
    "orb_pct": _f("four factors", "Offensive rebound percentage.", "p1-04", "box scores"),
    "ftr": _f("four factors", "Free throw attempts per field goal attempt.", "p1-04", "box scores"),
    "def_efg": _f("four factors", "Opponent effective field goal percentage.", "p1-04", "box scores"),
    "def_tov_pct": _f("four factors", "Opponent turnover percentage.", "p1-04", "box scores"),
    "def_orb_pct": _f("four factors", "Opponent offensive rebound percentage.", "p1-04", "box scores"),
    "def_ftr": _f("four factors", "Opponent free throw rate.", "p1-04", "box scores"),
    "fg3_rate": _f("four factors", "Three-point attempts per field goal attempt.", "p1-04", "box scores"),
    "fg3_pct": _f("four factors", "Three-point percentage.", "p1-04", "box scores"),
    "ft_pct": _f("four factors", "Free throw percentage.", "p1-04", "box scores"),
    # adjusted ratings (p1-01)
    "adj_off": _f("adjusted", "Points per 100 possessions against an average defense, venue-adjusted.", "p1-01", "box scores"),
    "adj_def": _f("adjusted", "Points allowed per 100 possessions against an average offense. Lower is better.", "p1-01", "box scores"),
    "adj_em": _f("adjusted", "adj_off - adj_def. Higher is better.", "p1-01", "box scores"),
    "adj_tempo": _f("adjusted", "Possessions per game against an average-tempo opponent.", "p1-01", "box scores"),
    "rated_games": _f("adjusted", "D1 box-scored games used in the adjusted ratings.", "p1-01", "box scores", diff=False),
    "hca": _f("adjusted", "League home-court advantage in points per 100 possessions, estimated as of the date.", "p1-01", "box scores", diff=False),
    "lg_eff": _f("adjusted", "League average points per 100 possessions as of the date.", "p1-01", "box scores", diff=False),
    "lg_tempo": _f("adjusted", "League average possessions per game as of the date.", "p1-01", "box scores", diff=False),
    # elo and schedule strength (p1-02, p1-03)
    "elo": _f("elo", "Elo rating after the last game on or before as_of. Starting values are untuned.", "p1-02"),
    "elo_games": _f("elo", "D1 games in the Elo history this season.", "p1-02", diff=False),
    "sos_elo": _f("schedule strength", "Mean opponent Elo before each game. A team's own result never feeds it.", "p1-03"),
    "sos_adj": _f("schedule strength", "Mean opponent adj_em as of the date. Carries a small effect from games against the team itself.", "p1-03", "box scores"),
    # form (p1-05)
    "last5_margin": _f("form", "Mean margin over the last 5 D1 games.", "p1-05"),
    "last10_margin": _f("form", "Mean margin over the last 10 D1 games.", "p1-05"),
    "form_trend": _f("form", "last5_margin minus the season mean D1 margin.", "p1-05"),
    # roster (p1-06)
    "n_players": _f("roster", "Players with minutes this season.", "p1-06", "player box scores", diff=False),
    "top5_min_share": _f("roster", "Share of team minutes played by the top five players.", "p1-06", "player box scores"),
    "top_player_min_share": _f("roster", "Largest single-player share of team minutes.", "p1-06", "player box scores"),
    "returning_min_pct": _f("roster", "Share of last season's minutes from players back on the same team.", "p1-06", "player season stats", "NaN without a prior season"),
    "incoming_prior_min": _f("roster", "Last-season minutes brought by players new to the team.", "p1-06", "player season stats", "NaN without a prior season"),
    "returning_bpm_wavg": _f("roster", "Minutes-weighted prior-season BPM of returning players.", "p1-06", "player season stats", "NaN without a prior season"),
    "prior_season_available": _f("roster", "1 when the previous season is in the database.", "p1-06", diff=False),
    "top_player_bpm": _f("roster", "Best prior-season BPM among returning players.", "p1-06", "player season stats", "NaN without a prior season"),
    "depth": _f("roster", "Players with at least 10% of team minutes.", "p1-06", "player box scores"),
    # coach and context (p1-07)
    "coach_seasons_at_school": _f("coach", "Seasons in the database with this coach at this school, counting this one. Starts in 2022.", "p1-07", "team_seasons"),
    "coach_prior_ncaa_apps": _f("coach", "Earlier seasons in the database with this coach and an NCAA bid. Starts in 2022.", "p1-07", "team_seasons"),
    "conference": _f("context", "Conference name, fixed for the season.", "Phase 0", "team_seasons", diff=False),
    "seed": _f("context", "NCAA seed. Known only from Selection Sunday, so NaN for earlier as-of dates and for non-tournament teams.", "Phase 0", "team_seasons"),
    # quality flags (p1-10)
    "box_games": _f("quality", "Box-scored games used for efficiency and four factors.", "p1-10", "box scores", diff=False),
    "box_coverage": _f("quality", "box_games / games.", "p1-10", "box scores", diff=False),
    "schedule_from_box": _f("quality", "1 when the team has no game log and results come from box scores.", "p1-10", diff=False),
    "max_input_date": _f("quality", "Latest game date used for this row. Always on or before as_of.", "p1-13", diff=False),
}
GLOBAL_FEATURES = ["lg_eff", "lg_tempo", "hca"]  # league-wide, taken once per matchup
SIDE_FEATURES = [c for c in FEATURES if c not in GLOBAL_FEATURES]
DIFF_FEATURES = [c for c, spec in FEATURES.items() if spec["diff"]]
ROSTER_COLS = [c for c, s in FEATURES.items() if s["group"] == "roster"]

MATCHUP_EXTRA = {
    "game_id": "Season, date and both team ids joined with |.",
    "season": "Season (ending year).",
    "game_date": "Date of the game.",
    "game_type": "Raw game_type from game_log_rows.",
    "game_group": "ncaa, conf_tourney or regular.",
    "round": "NCAA round, 0 (First Four) to 6 (final). NaN for other games.",
    "team_a": "Lexicographically smaller team id.",
    "team_b": "Other team id.",
    "overtimes": "Overtime periods played.",
    "loc_a": "+1 when team A is at home, -1 when away, 0 neutral.",
    "as_of": "Date the features are computed as of. Always before game_date.",
    "as_of_mode": "game or bracket. See data.asof.",
    "score_a": "Label. Team A final score.",
    "score_b": "Label. Team B final score.",
    "a_win": "Label. 1 when team A won.",
    "margin_a": "Label. score_a - score_b.",
    "total": "Label. score_a + score_b.",
    "swapped": "1 on rows created by with_both_orderings.",
    "proj_poss": "Projected possessions: adj_tempo_a + adj_tempo_b - lg_tempo.",
    "proj_pts_a": "Projected points for A from adjusted ratings, venue and tempo.",
    "proj_pts_b": "Projected points for B.",
    "proj_margin": "proj_pts_a - proj_pts_b.",
    "proj_total": "proj_pts_a + proj_pts_b.",
    "elo_expected_a": "Elo win probability for A, with home advantage.",
    "both_have_features": "1 when both teams have a team_asof row at this as_of.",
    "trainable": "1 when both teams have features, both have played the minimum games, and the game is not postseason other.",
}


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def classify_game_type(game_type) -> str:
    if game_type in NCAA_ROUNDS:
        return "ncaa"
    if game_type == "CTOURN":
        return "conf_tourney"
    if isinstance(game_type, str) and game_type.startswith("REG"):
        return "regular"
    return "other"


def _season_of(dates: pd.Series) -> pd.Series:
    return dates.dt.year + (dates.dt.month >= 8).astype(int)


def possessions(own: pd.DataFrame, opp_prefix: str = "o_") -> pd.Series:
    """Game possessions: the mean of both sides' estimates."""
    mine = own["fga"] - own["orb"] + own["tov"] + FTA_POSS * own["fta"]
    theirs = (
        own[f"{opp_prefix}fga"] - own[f"{opp_prefix}orb"] + own[f"{opp_prefix}tov"] + FTA_POSS * own[f"{opp_prefix}fta"]
    )
    return (mine + theirs) / 2


def load_box_games(seasons) -> pd.DataFrame:
    """One row per (D1 team, game) with own and opponent box score, all dates."""
    seasons = sorted(seasons)
    own = ", ".join(f"t.{c}" for c in STAT_COLS)
    opp = ", ".join(f"o.{c} AS o_{c}" for c in STAT_COLS)
    rows = query(
        f"""
        SELECT g.id AS game_id, g.canonical_box_score_path AS box_path, g.game_date,
               gt.team_source_path AS team, gt.team_name, gt.final_score AS pts_for,
               op.team_source_path AS opp, op.final_score AS pts_against, {own}, {opp}
        FROM games g
        JOIN game_teams gt ON gt.game_id = g.id
        JOIN game_teams op ON op.game_id = g.id AND op.id <> gt.id
        JOIN team_game_stats t ON t.game_team_id = gt.id
        JOIN team_game_stats o ON o.game_team_id = op.id
        WHERE g.game_status = 'final' AND gt.team_source_path IS NOT NULL
          AND g.game_date >= %s AND g.game_date < %s
        """,
        (season_start(seasons[0]), season_start(seasons[-1] + 1)),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = _season_of(df["game_date"])
    num = ["pts_for", "pts_against", *STAT_COLS, *[f"o_{c}" for c in STAT_COLS]]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["poss"] = possessions(df)
    return df


def load_game_facts(seasons, box: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (team, game) from game logs, joined to box-score possessions.

    Game logs are the backbone because they are complete and carry location and
    game type. `has_box` is False for games without a box score. `opp` is NaN
    for non-D1 opponents.
    """
    rows = query(
        """
        SELECT ss.ending_year AS season, s.canonical_source_path AS team,
               glr.opponent_school_path AS opp, glr.game_date, glr.location, glr.game_type,
               glr.overtimes, glr.team_score AS pts_for, glr.opponent_score AS pts_against,
               glr.canonical_box_score_path AS box_path
        FROM game_log_rows glr
        JOIN school_seasons ss ON ss.id = glr.school_season_id
        JOIN schools s ON s.id = ss.school_id
        WHERE glr.game_status = 'final' AND ss.ending_year = ANY(%s)
        """,
        (list(seasons),),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df[["pts_for", "pts_against"]] = df[["pts_for", "pts_against"]].apply(pd.to_numeric, errors="coerce")
    df["loc"] = df["location"].map(LOC_SIGN)
    df["game_group"] = df["game_type"].map(classify_game_type)
    df["overtimes"] = pd.to_numeric(df["overtimes"], errors="coerce").fillna(0).astype(int)
    if box is None:
        box = load_box_games(seasons)
    if box.empty:
        df["poss"] = np.nan
    else:
        b = box.dropna(subset=["box_path"])[["box_path", "team", "poss"]].drop_duplicates(["box_path", "team"])
        df = df.merge(b, on=["box_path", "team"], how="left")
    df["has_box"] = df["poss"].notna()
    return df.drop(columns="location")


# --------------------------------------------------------------------------- #
# games (the rows we predict) and labels
# --------------------------------------------------------------------------- #

def build_games(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per D1-vs-D1 game with canonical team order, location and labels."""
    d1 = facts[facts["opp"].notna() & facts["pts_for"].notna() & facts["pts_against"].notna()]
    flip = d1["team"] > d1["opp"]
    games = pd.DataFrame(
        {
            "season": d1["season"],
            "game_date": d1["game_date"],
            "game_type": d1["game_type"],
            "game_group": d1["game_group"],
            "round": d1["game_type"].map(NCAA_ROUNDS),
            "team_a": np.where(flip, d1["opp"], d1["team"]),
            "team_b": np.where(flip, d1["team"], d1["opp"]),
            "score_a": np.where(flip, d1["pts_against"], d1["pts_for"]),
            "score_b": np.where(flip, d1["pts_for"], d1["pts_against"]),
            "overtimes": d1["overtimes"],
            "loc_a": np.where(flip, -d1["loc"], d1["loc"]),
        }
    )
    games = games.drop_duplicates(["season", "game_date", "team_a", "team_b"])
    games["a_win"] = (games["score_a"] > games["score_b"]).astype(int)
    games["margin_a"] = games["score_a"] - games["score_b"]
    games["total"] = games["score_a"] + games["score_b"]
    games["game_id"] = (
        games["season"].astype(str) + "|" + games["game_date"].dt.strftime("%Y-%m-%d") + "|" + games["team_a"] + "|" + games["team_b"]
    )
    return games.sort_values(["season", "game_date", "team_a"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# pure feature computations
# --------------------------------------------------------------------------- #

def compute_form(facts: pd.DataFrame, season: int, as_of) -> pd.DataFrame:
    """Recent form from D1 games on or before `as_of`: last 5, last 10, and trend."""
    as_of = pd.Timestamp(as_of)
    g = facts[
        (facts["season"] == season)
        & (facts["game_date"] <= as_of)
        & facts["opp"].notna()
        & facts["pts_for"].notna()
        & facts["pts_against"].notna()
    ]
    if g.empty:
        return pd.DataFrame(columns=["team", "last5_margin", "last10_margin", "form_trend"])
    g = g.assign(margin=g["pts_for"] - g["pts_against"]).sort_values(["team", "game_date"], kind="stable")
    by = g.groupby("team")["margin"]
    out = pd.DataFrame(
        {
            "last5_margin": g.groupby("team").tail(5).groupby("team")["margin"].mean(),
            "last10_margin": g.groupby("team").tail(10).groupby("team")["margin"].mean(),
            "season_margin": by.mean(),
        }
    )
    out["form_trend"] = out["last5_margin"] - out["season_margin"]
    return out.drop(columns="season_margin").reset_index()


def roster_extras(players: pd.DataFrame) -> pd.DataFrame:
    """Depth and best returning player from a player snapshot."""
    if players.empty:
        return pd.DataFrame(columns=["team", "depth", "top_player_bpm"])
    depth = (players["min_share"] >= 0.10).groupby(players["team"]).sum().rename("depth")
    back = players[players["returning_same_team"] == 1]
    bpm = back.groupby("team")["prior_bpm"].max().rename("top_player_bpm")
    return pd.concat([depth, bpm], axis=1).reset_index()


def coach_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Coach tenure and earlier NCAA bids, counted only within the seasons in `meta`."""
    m = meta.sort_values(["team", "season"]).copy()
    m["had_bid"] = pd.to_numeric(m["ncaa_seed"], errors="coerce").notna().astype(int)
    g = m.groupby(["team", "coach_name"], dropna=False)
    m["coach_seasons_at_school"] = g.cumcount() + 1
    m["coach_prior_ncaa_apps"] = g["had_bid"].cumsum() - m["had_bid"]
    unknown = m["coach_name"].isna()
    m.loc[unknown, ["coach_seasons_at_school", "coach_prior_ncaa_apps"]] = np.nan
    return m[["season", "team", "coach_seasons_at_school", "coach_prior_ncaa_apps"]]


def team_features_asof(
    facts: pd.DataFrame,
    box: pd.DataFrame,
    players: pd.DataFrame | None,
    prior: pd.DataFrame | None,
    elo_hist: pd.DataFrame,
    season: int,
    as_of,
    shrink_games: float = 4.0,
) -> pd.DataFrame:
    """Every team's features using only games dated on or before `as_of`.

    `facts` and `box` may hold any seasons or dates; each piece filters again.
    """
    as_of = pd.Timestamp(as_of)
    sched = facts[["season", "team", "opp", "game_date", "pts_for", "pts_against"]]
    base = compute_team_snapshots(box, season, as_of, schedule=sched)
    if base.empty:
        return base
    base = base.rename(
        columns={"d_efg": "def_efg", "d_tov_pct": "def_tov_pct", "d_orb_pct": "def_orb_pct", "d_ftr": "def_ftr"}
    ).drop(columns=["team_name", "avg_pts_for", "avg_pts_against", "sos_naive", "last10_margin"])
    base["max_input_date"] = base.pop("last_game_date")

    rated = ratings.adjusted_ratings(facts, season, as_of, shrink_games=shrink_games)
    out = base.merge(rated.drop(columns=["season", "as_of"]), on="team", how="left")
    out = out.merge(ratings.elo_asof(elo_hist, season, as_of), on="team", how="left")
    out = out.merge(ratings.sos_asof(elo_hist, rated, season, as_of), on="team", how="left")
    out = out.merge(compute_form(facts, season, as_of), on="team", how="left")

    if players is not None and not players.empty:
        p = compute_player_snapshots(players, season, as_of, prior)
        if not p.empty:
            roster = compute_roster_features(p, prior).merge(roster_extras(p), on="team", how="left")
            out = out.merge(roster, on="team", how="left")
    for c in ROSTER_COLS:
        if c not in out:
            out[c] = np.nan
    return out


def build_team_asof(
    facts: pd.DataFrame,
    box: pd.DataFrame,
    players_by_season: dict,
    prior_by_season: dict,
    meta: pd.DataFrame,
    elo_hist: pd.DataFrame,
    cutoffs: dict,
    dates_by_season: dict,
    shrink_games: float = 4.0,
) -> pd.DataFrame:
    """The team_asof table for every needed (season, as_of) pair."""
    parts = []
    for season, dates in sorted(dates_by_season.items()):
        f_s = facts[facts["season"] == season]
        b_s = box[box["season"] == season]
        for d in sorted(pd.DatetimeIndex(dates)):
            parts.append(
                team_features_asof(
                    f_s, b_s, players_by_season.get(season), prior_by_season.get(season), elo_hist, season, d, shrink_games
                )
            )
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    ta = pd.concat(parts, ignore_index=True)
    m = meta.assign(seed=pd.to_numeric(meta["ncaa_seed"], errors="coerce"))[["season", "team", "conference", "seed"]]
    ta = ta.merge(m, on=["season", "team"], how="left").merge(coach_features(meta), on=["season", "team"], how="left")
    # seeds are announced on Selection Sunday. A season in progress has no cutoff (NaT), so none of its rows has a seed.
    ta["seed"] = ta["seed"].where(ta["as_of"] >= pd.to_datetime(ta["season"].map(cutoffs)))
    assert (ta["max_input_date"] <= ta["as_of"]).all(), "leak: a team_asof row used a game after its as_of"
    return ta


# --------------------------------------------------------------------------- #
# matchups
# --------------------------------------------------------------------------- #

def build_matchups(
    games: pd.DataFrame,
    team_asof: pd.DataFrame,
    min_games: int = DEFAULT_MIN_GAMES,
    as_of_mode: str = "game",
) -> pd.DataFrame:
    """One row per game: both teams at the game's as-of date, differences, projections, labels.

    `games` must carry an `as_of` column (see data.asof.asof_for_games). Rows
    whose as-of date has no team rows keep NaN features and are flagged by
    `both_have_features`.
    """
    asof_mod.assert_no_leak(games.dropna(subset=["as_of"]))
    keep = ["season", "team", "as_of", *FEATURES]
    ta = team_asof[[c for c in keep if c in team_asof]]
    side_a = ta.rename(columns={c: f"{c}_a" for c in SIDE_FEATURES} | {"team": "team_a"})
    side_b = ta.drop(columns=GLOBAL_FEATURES, errors="ignore").rename(
        columns={c: f"{c}_b" for c in SIDE_FEATURES} | {"team": "team_b"}
    )
    m = games.merge(side_a, on=["season", "team_a", "as_of"], how="left").merge(
        side_b, on=["season", "team_b", "as_of"], how="left"
    )
    diffs = {f"diff_{c}": m[f"{c}_a"] - m[f"{c}_b"] for c in DIFF_FEATURES if f"{c}_a" in m}
    m = pd.concat([m, pd.DataFrame(diffs, index=m.index)], axis=1).copy()
    m["as_of_mode"] = as_of_mode
    m["both_have_features"] = (m["adj_em_a"].notna() & m["adj_em_b"].notna()).astype(int)

    # Projections from adjusted ratings: points = (off + opp def - league avg + venue) * possessions / 100
    m["proj_poss"] = m["adj_tempo_a"] + m["adj_tempo_b"] - m["lg_tempo"]
    venue = m["hca"] * m["loc_a"]
    m["proj_pts_a"] = (m["adj_off_a"] + m["adj_def_b"] - m["lg_eff"] + venue) * m["proj_poss"] / 100
    m["proj_pts_b"] = (m["adj_off_b"] + m["adj_def_a"] - m["lg_eff"] - venue) * m["proj_poss"] / 100
    m["proj_margin"] = m["proj_pts_a"] - m["proj_pts_b"]
    m["proj_total"] = m["proj_pts_a"] + m["proj_pts_b"]
    m["elo_expected_a"] = 1.0 / (1.0 + 10.0 ** (-(m["elo_a"] - m["elo_b"] + ELO_HOME_ADV * m["loc_a"]) / 400.0))

    m["trainable"] = (
        (m["both_have_features"] == 1)
        & (m["games_a"] >= min_games)
        & (m["games_b"] >= min_games)
        & (m["game_group"] != "other")
    ).astype(int)
    m["swapped"] = 0
    return m


def with_both_orderings(m: pd.DataFrame) -> pd.DataFrame:
    """Add a copy of every row with team A and B swapped (labels and differences flipped).

    Folds must keep both orderings of one game together: group by `game_id`
    or by season.
    """
    s = m.copy()
    cols = set(s.columns)
    pairs = [(c, c[:-2] + "_b") for c in s.columns if c.endswith("_a") and c[:-2] + "_b" in cols]
    for a, b in pairs:
        s[a], s[b] = m[b], m[a]
    for c in [c for c in s.columns if c.startswith("diff_")] + ["loc_a", "margin_a", "proj_margin"]:
        s[c] = -m[c]
    s["a_win"] = 1 - m["a_win"]
    s["elo_expected_a"] = 1 - m["elo_expected_a"]
    s["swapped"] = 1
    return pd.concat([m, s], ignore_index=True)


def matchup_for(team_asof: pd.DataFrame, season: int, team_a: str, team_b: str, as_of, loc: float = 0.0) -> pd.DataFrame:
    """A matchup row for two teams on a date, without labels (Phase 4 reuse)."""
    a, b = sorted([team_a, team_b])
    flip = a != team_a
    g = pd.DataFrame(
        {
            "season": [season],
            "game_date": [pd.Timestamp(as_of) + pd.Timedelta(days=1)],
            "as_of": [pd.Timestamp(as_of)],
            "game_group": ["regular"],
            "team_a": [a],
            "team_b": [b],
            "loc_a": [-loc if flip else loc],
        }
    )
    return build_matchups(g, team_asof)


# --------------------------------------------------------------------------- #
# documentation
# --------------------------------------------------------------------------- #

def column_dictionary(team_asof_cols, matchup_cols) -> pd.DataFrame:
    """Documented columns for team_asof and matchups. Raises if any column is undocumented."""
    rows, missing = [], []
    for c in team_asof_cols:
        if c in ("season", "team", "as_of"):
            rows.append(("team_asof", c, "key", "Key column.", "", "", "", ""))
        elif c in FEATURES:
            s = FEATURES[c]
            rows.append(("team_asof", c, s["group"], s["definition"], s["source"], s["missing"], s["step"], ""))
        else:
            missing.append(("team_asof", c))
    for c in matchup_cols:
        if c in MATCHUP_EXTRA:
            rows.append(("matchups", c, "matchup", MATCHUP_EXTRA[c], "", "", "", ""))
        elif c in GLOBAL_FEATURES:
            s = FEATURES[c]
            rows.append(("matchups", c, s["group"], s["definition"], s["source"], s["missing"], s["step"], ""))
        elif c.startswith("diff_") and c[5:] in FEATURES:
            s = FEATURES[c[5:]]
            rows.append(("matchups", c, s["group"], f"Team A minus team B: {s['definition']}", s["source"], s["missing"], s["step"], ""))
        elif c[-2:] in ("_a", "_b") and c[:-2] in FEATURES:
            s = FEATURES[c[:-2]]
            rows.append(("matchups", c, s["group"], f"Team {c[-1].upper()}: {s['definition']}", s["source"], s["missing"], s["step"], ""))
        else:
            missing.append(("matchups", c))
    if missing:
        raise KeyError(f"undocumented columns: {missing}")
    return pd.DataFrame(rows, columns=["table", "column", "group", "definition", "source", "missing", "roadmap", "notes"])


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def assemble_feature_matrix(
    facts: pd.DataFrame,
    box: pd.DataFrame,
    players_by_season: dict,
    prior_by_season: dict,
    meta: pd.DataFrame,
    include=GAME_GROUPS,
    cadence: str = "weekly",
    mode: str = "game",
    shrink_games: float = 4.0,
    min_games: int = DEFAULT_MIN_GAMES,
    elo_kwargs: dict | None = None,
    through=None,
) -> dict:
    """Everything after loading: grid, as-of dates, Elo, team_asof and matchups.

    Pure pandas, so tests can run it on synthetic data. Returns a dict with
    matchups, team_asof, cutoffs and report.

    `through` drops every game dated after it, so a run can leave out a day that is still being played.

    A season with no tournament games yet (the live season) has no Selection Sunday, so it has no entry in `cutoffs`,
    its teams have no seeds, and its newest as-of date is the date of its newest game.
    """
    if through is not None:
        through = pd.Timestamp(through)
        facts, box = facts[facts["game_date"] <= through], box[box["game_date"] <= through]
    seasons = sorted(facts["season"].unique())
    games_all = build_games(facts)

    ncaa = games_all[games_all["game_group"] == "ncaa"]
    cutoffs = cutoffs_from_games(ncaa)
    last_ncaa = ncaa.groupby("season")["game_date"].max()
    last_any = facts.groupby("season")["game_date"].max()
    first_game = facts.groupby("season")["game_date"].min()
    grid = {s: asof_mod.asof_grid(first_game[s], cutoffs.get(s), last_ncaa.get(s, last_any[s]), cadence) for s in seasons}

    games = games_all[games_all["game_group"].isin(include)].copy()
    games["as_of"] = asof_mod.asof_for_games(games, grid, cutoffs, mode)
    no_asof = int(games["as_of"].isna().sum())
    games = games.dropna(subset=["as_of"]).reset_index(drop=True)
    # Selection Sunday is always built, so the bracket-time snapshot exists in either mode. A season in progress has none,
    # so its newest grid date is built instead: no game is after it, so no game would ask for it.
    dates_by_season = {
        int(s): pd.DatetimeIndex(g.unique()).union(pd.DatetimeIndex([cutoffs[int(s)] if int(s) in cutoffs else max(grid[int(s)])]))
        for s, g in games.groupby("season")["as_of"]
    }

    elo_hist = ratings.elo_history(games_all, home_adv=ELO_HOME_ADV, **(elo_kwargs or {}))
    team_asof = build_team_asof(
        facts, box, players_by_season, prior_by_season, meta, elo_hist, cutoffs, dates_by_season, shrink_games
    )
    matchups = build_matchups(games, team_asof, min_games=min_games, as_of_mode=mode)
    report = {
        "seasons": [int(s) for s in seasons],
        "cadence": cadence,
        "mode": mode,
        "games_without_asof": no_asof,
        "grid_dates": {int(s): len(g) for s, g in grid.items()},
        "team_asof_rows": len(team_asof),
    }
    return {"matchups": matchups, "team_asof": team_asof, "cutoffs": cutoffs, "report": report}


def load_raw(seasons) -> dict:
    """Everything the database provides, loaded once: facts, box, meta, players, prior.

    Keep the result to rebuild features with different settings without
    touching the database (see scripts/tune_features.py).
    """
    seasons = sorted(seasons)
    box = load_box_games(seasons)
    facts = load_game_facts(seasons, box)
    ncaa_last = facts[facts["game_group"] == "ncaa"].groupby("season")["game_date"].max()
    last_any = facts.groupby("season")["game_date"].max()  # a season in progress has no tournament games yet
    meta = load_team_meta(seasons)
    players, prior, prior_cache = {}, {}, {}
    for s in seasons:
        players[s] = load_player_games(s, ncaa_last.get(s, last_any[s]))
        if s - 1 not in prior_cache:
            prior_cache[s - 1] = load_player_seasons(s - 1)
        prior[s] = prior_cache[s - 1]
    return {"facts": facts, "box": box, "meta": meta, "players": players, "prior": prior}


def build_from_raw(raw: dict, include=GAME_GROUPS, cadence: str = "weekly", mode: str = "game", shrink_games: float = 4.0,
                   min_games: int = DEFAULT_MIN_GAMES, elo_kwargs: dict | None = None, with_players: bool = True, through=None) -> dict:
    """Assemble from loaded data. `with_players=False` skips roster features (much faster; roster columns are NaN)."""
    return assemble_feature_matrix(
        raw["facts"], raw["box"], raw["players"] if with_players else {}, raw["prior"] if with_players else {}, raw["meta"],
        include, cadence, mode, shrink_games, min_games, elo_kwargs, through,
    )


def build_feature_matrix(
    seasons,
    include=GAME_GROUPS,
    cadence: str = "weekly",
    mode: str = "game",
    shrink_games: float = 4.0,
    min_games: int = DEFAULT_MIN_GAMES,
    elo_kwargs: dict | None = None,
    through=None,
) -> dict:
    """Load from the database, then assemble. Adds `facts` to the returned dict. `through`: ignore games after this date."""
    raw = load_raw(seasons)
    out = build_from_raw(raw, include, cadence, mode, shrink_games, min_games, elo_kwargs, through=through)
    out["facts"] = raw["facts"]
    return out
