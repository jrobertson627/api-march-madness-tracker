"""Point-in-time (as-of) snapshots for tournament prediction.

Everything here answers one question: "what was knowable on date D?"

Why this module exists: `team_seasons` (srs, sos, off_rtg, def_rtg, wins, losses)
and `team_season_stats` are full-season values that INCLUDE NCAA tournament
games, and `player_season_stats` likewise. Training on them would leak the
outcomes we are trying to predict. Instead we rebuild team and player state from
per-game box scores, using only games with game_date <= as_of.

Layout: loaders hit the database (and push the as-of filter into SQL); the
`compute_*` functions are pure DataFrame transforms that filter again, so they
can be unit tested without a database and cannot leak even if handed
too much data.

The cutoff for tournament prediction is Selection Sunday: the Sunday before the
first NCAA tournament game. Conference tournament finals are played that day and
seeds are announced that evening, so a snapshot at that cutoff matches what a
bracket-time prediction could actually see.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from data.db import query

STAT_COLS = ["fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"]
PLAYER_COUNT_COLS = ["minutes", "pts", *STAT_COLS]

# game_log_rows.game_type values for NCAA tournament games -> round number.
NCAA_ROUNDS = {
    "FIRST-FOUR": 0,
    "ROUND-64": 1,
    "ROUND-32": 2,
    "ROUND-16": 3,
    "ROUND-8": 4,
    "NATIONAL-SEMI": 5,
    "NATIONAL-FINAL": 6,
}

FTA_POSS = 0.44  # free-throw possession weight (same constant as data/teams.py)


# --------------------------------------------------------------------------- #
# dates / cutoffs
# --------------------------------------------------------------------------- #

def season_start(season: int) -> dt.date:
    """Games dated on/after this belong to `season` (season = ending year)."""
    return dt.date(season - 1, 8, 1)


def selection_sunday(first_tournament_date) -> pd.Timestamp:
    """The most recent Sunday strictly before the first tournament game."""
    d = pd.Timestamp(first_tournament_date).normalize()
    back = (d.weekday() + 1) % 7 or 7  # Mon=0..Sun=6 -> days since last Sunday
    return d - pd.Timedelta(days=back)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_tournament_games(seasons) -> pd.DataFrame:
    """One row per NCAA tournament game (First Four included), labels only.

    Built from game_log_rows rather than `games` because ~3% of tournament
    games have no box score in `games` but still have final scores here.
    Teams are put in a canonical order (a = lexicographically smaller path) so
    class balance does not depend on seed or home/away ordering.
    """
    rows = query(
        """
        SELECT ss.ending_year AS season,
               s.canonical_source_path AS team,
               glr.opponent_school_path AS opp,
               glr.game_date, glr.game_type,
               glr.team_score, glr.opponent_score, glr.overtimes,
               EXISTS (SELECT 1 FROM games g
                       WHERE g.canonical_box_score_path = glr.canonical_box_score_path) AS has_box_score
        FROM game_log_rows glr
        JOIN school_seasons ss ON ss.id = glr.school_season_id
        JOIN schools s ON s.id = ss.school_id
        WHERE glr.game_type = ANY(%s) AND glr.game_status = 'final' AND ss.ending_year = ANY(%s)
        """,
        (list(NCAA_ROUNDS), list(seasons)),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    flip = df["team"] > df["opp"]
    out = pd.DataFrame(
        {
            "season": df["season"],
            "game_date": df["game_date"],
            "round": df["game_type"].map(NCAA_ROUNDS),
            "round_name": df["game_type"],
            "team_a": np.where(flip, df["opp"], df["team"]),
            "team_b": np.where(flip, df["team"], df["opp"]),
            "score_a": np.where(flip, df["opponent_score"], df["team_score"]),
            "score_b": np.where(flip, df["team_score"], df["opponent_score"]),
            "overtimes": df["overtimes"].fillna(0).astype(int),
            "has_box_score": df["has_box_score"],
        }
    )
    # Both teams' game-log rows describe the same game; keep one.
    out = out.drop_duplicates(["season", "game_date", "team_a", "team_b"])
    out["a_win"] = (out["score_a"] > out["score_b"]).astype(int)
    out["margin_a"] = out["score_a"] - out["score_b"]
    out["total"] = out["score_a"] + out["score_b"]
    return out.sort_values(["season", "game_date", "round", "team_a"]).reset_index(drop=True)


def cutoffs_from_games(tournament_games: pd.DataFrame) -> dict[int, pd.Timestamp]:
    first = tournament_games.groupby("season")["game_date"].min()
    return {int(season): selection_sunday(d) for season, d in first.items()}


def load_team_games(season: int, through) -> pd.DataFrame:
    """One row per (D1 team, game) with own and opponent box score, through `through`."""
    own = ", ".join(f"t.{c}" for c in STAT_COLS)
    opp = ", ".join(f"o.{c} AS o_{c}" for c in STAT_COLS)
    rows = query(
        f"""
        SELECT g.id AS game_id, g.game_date, g.neutral_site, g.overtimes,
               gt.team_source_path AS team, gt.team_name, gt.side,
               gt.final_score AS pts_for, op.team_source_path AS opp, op.final_score AS pts_against,
               {own}, {opp}
        FROM games g
        JOIN game_teams gt ON gt.game_id = g.id
        JOIN game_teams op ON op.game_id = g.id AND op.id <> gt.id
        JOIN team_game_stats t ON t.game_team_id = gt.id
        JOIN team_game_stats o ON o.game_team_id = op.id
        WHERE g.game_status = 'final'
          AND gt.team_source_path IS NOT NULL
          AND g.game_date >= %s AND g.game_date <= %s
        """,
        (season_start(season), pd.Timestamp(through).date()),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = season
    num = ["pts_for", "pts_against", *STAT_COLS, *[f"o_{c}" for c in STAT_COLS]]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


def load_player_games(season: int, through) -> pd.DataFrame:
    cols = ", ".join(f"p.{c}" for c in PLAYER_COUNT_COLS)
    rows = query(
        f"""
        SELECT g.game_date, gt.team_source_path AS team,
               p.player_id, p.player_name, p.starter, {cols}
        FROM player_game_stats p
        JOIN game_teams gt ON gt.id = p.game_team_id
        JOIN games g ON g.id = gt.game_id
        WHERE g.game_status = 'final'
          AND gt.team_source_path IS NOT NULL
          AND g.game_date >= %s AND g.game_date <= %s
        """,
        (season_start(season), pd.Timestamp(through).date()),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = season
    df[PLAYER_COUNT_COLS] = df[PLAYER_COUNT_COLS].apply(pd.to_numeric, errors="coerce")
    df["starter"] = df["starter"].fillna(False).astype(int)
    return df


def load_player_seasons(season: int) -> pd.DataFrame:
    """Final player season stats for a COMPLETED season (use only as prior-season data)."""
    rows = query(
        """
        SELECT s.canonical_source_path AS team, p.player_id, p.player_name,
               p.games, p.games_started, p.minutes, p.pts, p.trb, p.ast, p.advanced
        FROM player_season_stats p
        JOIN school_seasons ss ON ss.id = p.school_season_id
        JOIN schools s ON s.id = ss.school_id
        WHERE ss.ending_year = %s
        """,
        (season,),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for k in ("bpm", "ws", "per", "usg_pct"):
        df[k] = [((a or {}).get(k) or {}).get("value") for a in df["advanced"]]
    df = df.drop(columns="advanced")
    num = ["games", "games_started", "minutes", "pts", "trb", "ast", "bpm", "ws", "per", "usg_pct"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


def load_team_meta(seasons) -> pd.DataFrame:
    rows = query(
        """
        SELECT s.canonical_source_path AS team, ss.ending_year AS season,
               s.display_name AS team_name,
               regexp_replace(ts.conference_name, ' MBB$', '') AS conference,
               ts.coach_name, ts.ncaa_seed, ts.ncaa_region
        FROM team_seasons ts
        JOIN school_seasons ss ON ss.id = ts.school_season_id
        JOIN schools s ON s.id = ss.school_id
        WHERE ss.ending_year = ANY(%s)
        """,
        (list(seasons),),
    )
    return pd.DataFrame(rows)


def load_schedule(season: int, through) -> pd.DataFrame:
    """Results for every final game (game_log_rows), through `through`.

    More complete than `games`: ~5% of D1 games have no box score row but do
    have scores here, so record/margin/form/SOS are computed from this table.
    """
    rows = query(
        """
        SELECT s.canonical_source_path AS team, glr.opponent_school_path AS opp, glr.game_date,
               glr.team_score AS pts_for, glr.opponent_score AS pts_against
        FROM game_log_rows glr
        JOIN school_seasons ss ON ss.id = glr.school_season_id
        JOIN schools s ON s.id = ss.school_id
        WHERE ss.ending_year = %s AND glr.game_status = 'final' AND glr.game_date <= %s
        """,
        (season, pd.Timestamp(through).date()),
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df[["pts_for", "pts_against"]] = df[["pts_for", "pts_against"]].apply(pd.to_numeric, errors="coerce")
        df["season"] = season
    return df


# --------------------------------------------------------------------------- #
# pure computations
# --------------------------------------------------------------------------- #

def _ratio(num, den):
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _schedule_features(sched: pd.DataFrame) -> pd.DataFrame:
    """Record / margin / form / naive SOS from one row per (team, game)."""
    sched = sched.dropna(subset=["pts_for", "pts_against"]).sort_values(["team", "game_date"], kind="stable").copy()
    sched["margin"] = sched["pts_for"] - sched["pts_against"]
    sched["win"] = (sched["margin"] > 0).astype(int)
    sched["opp_is_d1"] = sched["opp"].notna().astype(int)
    g = sched.groupby("team")
    n = g.size()
    out = pd.DataFrame(index=n.index)
    out["games"] = n
    out["d1_games"] = g["opp_is_d1"].sum()
    out["wins"] = g["win"].sum()
    out["losses"] = n - out["wins"]
    out["win_pct"] = out["wins"] / n
    out["avg_pts_for"] = g["pts_for"].mean()
    out["avg_pts_against"] = g["pts_against"].mean()
    out["avg_margin"] = out["avg_pts_for"] - out["avg_pts_against"]
    out["last10_margin"] = g.tail(10).groupby("team")["margin"].mean()
    # Mean of opponents' own season-to-date average margin. It includes the game
    # against the team itself, so treat it as a rough signal (adjusted SOS: Phase 1).
    out["sos_naive"] = sched.assign(opp_avg_margin=sched["opp"].map(out["avg_margin"])).groupby("team")["opp_avg_margin"].mean()
    out["last_game_date"] = g["game_date"].max()
    return out


def compute_team_snapshots(team_games: pd.DataFrame, season: int, as_of, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Team state using only games with game_date <= as_of (one row per team).

    Record, margin, form and SOS come from `schedule` (game_log_rows: complete).
    Efficiency, tempo and four factors come from box scores in `team_games`,
    which can miss ~5% of games; `box_games` says how many were used. If no
    schedule is given, one is derived from the box scores. Opponent-adjusted
    ratings are Phase 1.
    """
    as_of = pd.Timestamp(as_of)

    def window(frame):
        return frame[(frame["season"] == season) & (frame["game_date"] <= as_of)]

    df = window(team_games).copy()
    box_sched = team_games[["season", "team", "opp", "game_date", "pts_for", "pts_against"]]
    schedule_teams_with_log = set() if schedule is None else set(schedule["team"])
    if schedule is None:
        schedule = box_sched
    else:
        # A team whose game log hasn't been crawled yet falls back to box-score results.
        absent = ~box_sched["team"].isin(schedule_teams_with_log)
        schedule = pd.concat([schedule, box_sched[absent]], ignore_index=True)
    sched = window(schedule)
    if df.empty or sched.empty:
        return pd.DataFrame()

    df = df.sort_values(["team", "game_date", "game_id"])
    g = df.groupby("team")
    sum_cols = ["pts_for", "pts_against", *STAT_COLS, *[f"o_{c}" for c in STAT_COLS]]
    s = g[sum_cols].sum()
    n = g.size()

    def o(c):
        return s[f"o_{c}"]

    poss_own = s["fga"] - s["orb"] + s["tov"] + FTA_POSS * s["fta"]
    poss_opp = o("fga") - o("orb") + o("tov") + FTA_POSS * o("fta")
    poss = (poss_own + poss_opp) / 2

    sched_feats = _schedule_features(sched)
    out = pd.DataFrame(index=sched_feats.index)
    out["team_name"] = g["team_name"].last()
    for c in sched_feats.columns:
        out[c] = sched_feats[c]
    out["box_games"] = n
    out["box_coverage"] = n / sched_feats["games"]
    out["tempo"] = poss / n  # est. possessions per box-scored game
    out["off_eff"] = 100 * _ratio(s["pts_for"], poss)
    out["def_eff"] = 100 * _ratio(s["pts_against"], poss)
    out["eff_margin"] = out["off_eff"] - out["def_eff"]
    # four factors, offense then defense (opponent's numbers against this team)
    out["efg"] = _ratio(s["fg"] + 0.5 * s["fg3"], s["fga"])
    out["tov_pct"] = _ratio(s["tov"], s["fga"] + FTA_POSS * s["fta"] + s["tov"])
    out["orb_pct"] = _ratio(s["orb"], s["orb"] + o("drb"))
    out["ftr"] = _ratio(s["fta"], s["fga"])
    out["d_efg"] = _ratio(o("fg") + 0.5 * o("fg3"), o("fga"))
    out["d_tov_pct"] = _ratio(o("tov"), o("fga") + FTA_POSS * o("fta") + o("tov"))
    out["d_orb_pct"] = _ratio(o("orb"), o("orb") + s["drb"])
    out["d_ftr"] = _ratio(o("fta"), o("fga"))
    out["fg3_rate"] = _ratio(s["fg3a"], s["fga"])
    out["fg3_pct"] = _ratio(s["fg3"], s["fg3a"])
    out["ft_pct"] = _ratio(s["ft"], s["fta"])

    out["schedule_from_box"] = (~out.index.isin(set(schedule_teams_with_log))).astype(int)
    out["last_game_date"] = pd.concat([sched_feats["last_game_date"], g["game_date"].max()], axis=1).max(axis=1)
    out["season"] = season
    out["as_of"] = as_of
    return out.reset_index()


def compute_player_snapshots(player_games: pd.DataFrame, season: int, as_of, prior: pd.DataFrame | None) -> pd.DataFrame:
    """Per-player season-to-date line using games <= as_of, plus prior-season carry-over.

    `prior` is the previous (completed) season's player_season_stats frame, or
    None/empty when that season isn't in the database. Players without a
    player_id (~4% of box score rows) are keyed by name.
    """
    as_of = pd.Timestamp(as_of)
    df = player_games[(player_games["season"] == season) & (player_games["game_date"] <= as_of)].copy()
    if df.empty:
        return pd.DataFrame()

    df["player_key"] = df["player_id"].astype("Int64").astype("string").fillna("name:" + df["player_name"].astype("string"))
    g = df.groupby(["team", "player_key"])
    p = g[PLAYER_COUNT_COLS].sum()
    p["player_id"] = g["player_id"].first()
    p["player_name"] = g["player_name"].last()
    p["games"] = g.size()
    p["starts"] = g["starter"].sum()
    p = p.reset_index()

    team_minutes = p.groupby("team")["minutes"].transform("sum")
    p["min_share"] = _ratio(p["minutes"], team_minutes)  # shares sum to 1 per team
    p["mpg"] = _ratio(p["minutes"], p["games"])
    for c in ("pts", "trb", "ast", "stl", "blk", "tov"):
        p[f"{c}_per40"] = 40 * _ratio(p[c], p["minutes"])
    p["ts_pct"] = _ratio(p["pts"], 2 * (p["fga"] + FTA_POSS * p["fta"]))

    prior_cols = ["prior_team", "prior_games", "prior_minutes", "prior_bpm", "prior_ws", "prior_per", "prior_usg_pct"]
    if prior is not None and not prior.empty:
        pr = (
            prior.dropna(subset=["player_id"])
            .sort_values("minutes", ascending=False)
            .drop_duplicates("player_id")  # keep the stop where they played most
            .rename(
                columns={
                    "team": "prior_team", "games": "prior_games", "minutes": "prior_minutes",
                    "bpm": "prior_bpm", "ws": "prior_ws", "per": "prior_per", "usg_pct": "prior_usg_pct",
                }
            )[["player_id", *prior_cols]]
        )
        p = p.merge(pr, on="player_id", how="left")
        p["returning_same_team"] = (p["prior_team"] == p["team"]).astype(int)
    else:
        for c in prior_cols:
            p[c] = np.nan
        p["returning_same_team"] = 0

    p["season"] = season
    p["as_of"] = as_of
    return p


def compute_roster_features(players: pd.DataFrame, prior: pd.DataFrame | None) -> pd.DataFrame:
    """Team-level roster summary from a player snapshot (one row per team).

    returning_min_pct: share of last season's team minutes played by players who
    are back on the same team. Needs a prior season; NaN otherwise.
    """
    if players.empty:
        return pd.DataFrame()
    g = players.groupby("team")
    top = players.sort_values("min_share", ascending=False).groupby("team")
    out = pd.DataFrame(index=g.size().index)
    out["n_players"] = g.size()
    out["top5_min_share"] = top["min_share"].apply(lambda s: s.head(5).sum())
    out["top_player_min_share"] = top["min_share"].first()

    if prior is not None and not prior.empty:
        prev_minutes = prior.groupby("team")["minutes"].sum()
        back = players[players["returning_same_team"] == 1]
        out["returning_min_pct"] = _ratio(back.groupby("team")["prior_minutes"].sum(), prev_minutes.reindex(out.index)).fillna(0.0)
        incoming = players[(players["prior_team"].notna()) & (players["returning_same_team"] == 0)]
        out["incoming_prior_min"] = incoming.groupby("team")["prior_minutes"].sum().reindex(out.index).fillna(0.0)
        w = back.assign(w=back["prior_bpm"] * back["prior_minutes"]).groupby("team")
        out["returning_bpm_wavg"] = _ratio(w["w"].sum(), w["prior_minutes"].sum())
        out["prior_season_available"] = 1
    else:
        out["returning_min_pct"] = np.nan
        out["incoming_prior_min"] = np.nan
        out["returning_bpm_wavg"] = np.nan
        out["prior_season_available"] = 0
    return out.reset_index()


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def build_tournament_dataset(seasons) -> dict[str, pd.DataFrame]:
    """Labels + as-of-Selection-Sunday snapshots for each season.

    Returns dict with: games (labels), cutoffs, teams, players.
    Raises AssertionError if any snapshot uses a game after its cutoff.
    """
    seasons = sorted(seasons)
    games = load_tournament_games(seasons)
    cutoffs = cutoffs_from_games(games)
    meta = load_team_meta(seasons)

    teams, players = [], []
    prior_by_season = {}
    for season in seasons:
        cutoff = cutoffs[season]
        tg = load_team_games(season, cutoff)
        pg = load_player_games(season, cutoff)
        sched = load_schedule(season, cutoff)
        if season - 1 not in prior_by_season:
            prior_by_season[season - 1] = load_player_seasons(season - 1)
        prior = prior_by_season[season - 1]

        t = compute_team_snapshots(tg, season, cutoff, schedule=sched)
        p = compute_player_snapshots(pg, season, cutoff, prior)
        r = compute_roster_features(p, prior)
        t = t.merge(r, on="team", how="left")

        # Seeds/conference/coach are known at bracket time; ncaa_games results are NOT attached.
        t = t.merge(meta[meta["season"] == season].drop(columns=["team_name", "season"]), on="team", how="left")

        assert (t["last_game_date"] <= cutoff).all(), f"leak: {season} snapshot uses games after {cutoff.date()}"
        assert (p["as_of"] <= cutoff).all()
        teams.append(t)
        players.append(p)

    teams = pd.concat(teams, ignore_index=True)
    players = pd.concat(players, ignore_index=True)
    seeds = teams[["season", "team", "ncaa_seed", "ncaa_region"]]
    for side in ("a", "b"):
        games = games.merge(
            seeds.rename(columns={"team": f"team_{side}", "ncaa_seed": f"seed_{side}", "ncaa_region": f"region_{side}"}),
            on=["season", f"team_{side}"],
            how="left",
        )
    games["both_have_snapshot"] = games["seed_a"].notna() & games["seed_b"].notna()
    return {
        "games": games,
        "cutoffs": pd.DataFrame({"season": list(cutoffs), "cutoff": list(cutoffs.values())}),
        "teams": teams,
        "players": players,
    }
