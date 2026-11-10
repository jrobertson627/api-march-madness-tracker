"""Opponent-adjusted ratings, Elo and strength of schedule.

Pure functions of a long game table, so they can be tested without the database
and cannot leak: every function that takes an `as_of` filters to
game_date <= as_of itself.

Expected `facts` columns (one row per team per game, see data.features):
    season, team, opp, game_date, loc, pts_for, poss
`opp` is NaN for non-D1 opponents and `poss` is NaN without a box score; both
kinds of rows are ignored by the efficiency ratings. `loc` is +1 at home,
-1 away, 0 neutral, from `team`'s side.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from data.snapshots import season_start

ELO_BASE = 1500.0

RATING_COLS = ["team", "adj_off", "adj_def", "adj_em", "adj_tempo", "rated_games", "hca", "lg_eff", "lg_tempo", "season", "as_of"]


# --------------------------------------------------------------------------- #
# adjusted efficiency and tempo
# --------------------------------------------------------------------------- #

def adjusted_ratings(
    facts: pd.DataFrame,
    season: int,
    as_of,
    shrink_games: float = 4.0,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> pd.DataFrame:
    """Opponent- and venue-adjusted efficiency and tempo as of a date.

    Each box-scored D1 game contributes one observation per side:
        pts per 100 possessions = mu + off[team] + def[opp] + h * loc
        possessions             = mu_t + tp[team] + tp[opp]
    Solved by alternating updates, which converge to the ridge regression
    solution. A team's rating is its summed residuals divided by
    (games + shrink_games), so early-season ratings shrink toward the league
    average. Output per team:
        adj_off    points scored per 100 possessions against an average defense
        adj_def    points allowed per 100 possessions against an average offense
        adj_em     adj_off - adj_def (higher is better)
        adj_tempo  possessions per game against an average-tempo opponent
        rated_games, hca (home-court points per 100), lg_eff, lg_tempo
    """
    as_of = pd.Timestamp(as_of)
    g = facts[
        (facts["season"] == season) & (facts["game_date"] <= as_of) & facts["poss"].notna() & facts["opp"].notna()
    ]
    g = g[g["opp"].isin(set(g["team"]))]  # both sides must be rated teams
    if g.empty:
        return pd.DataFrame(columns=RATING_COLS)

    codes, teams = pd.factorize(g["team"], sort=True)
    t = codes
    o = teams.get_indexer(g["opp"])
    n = len(teams)
    y = 100.0 * g["pts_for"].to_numpy(float) / g["poss"].to_numpy(float)
    poss = g["poss"].to_numpy(float)
    loc = g["loc"].fillna(0).to_numpy(float)
    n_team = np.bincount(t, minlength=n).astype(float)
    n_opp = np.bincount(o, minlength=n).astype(float)

    off = np.zeros(n)
    dfn = np.zeros(n)
    tp = np.zeros(n)
    mu, h, mu_t = float(y.mean()), 0.0, float(poss.mean())
    loc_var = loc.var()
    for _ in range(max_iter):
        before = np.concatenate([off, dfn, tp, [mu, h, mu_t]])
        off = np.bincount(t, y - mu - dfn[o] - h * loc, n) / (n_team + shrink_games)
        dfn = np.bincount(o, y - mu - off[t] - h * loc, n) / (n_opp + shrink_games)
        r = y - off[t] - dfn[o]
        h = float(np.cov(loc, r, bias=True)[0, 1] / loc_var) if loc_var > 0 else 0.0
        mu = float(r.mean() - h * loc.mean())
        tp = np.bincount(t, poss - mu_t - tp[o], n) / (n_team + shrink_games)
        mu_t = float((poss - tp[t] - tp[o]).mean())
        if np.abs(np.concatenate([off, dfn, tp, [mu, h, mu_t]]) - before).max() < tol:
            break

    # Centre so an average team has off = def = 0; the shift moves into mu.
    mu_c = mu + off.mean() + dfn.mean()
    mu_t_c = mu_t + 2 * tp.mean()
    out = pd.DataFrame(
        {
            "team": teams,
            "adj_off": mu_c + (off - off.mean()),
            "adj_def": mu_c + (dfn - dfn.mean()),
            "adj_tempo": mu_t_c + (tp - tp.mean()),
            "rated_games": n_team.astype(int),
            "hca": h,
            "lg_eff": mu_c,
            "lg_tempo": mu_t_c,
        }
    )
    out["adj_em"] = out["adj_off"] - out["adj_def"]
    out["season"] = season
    out["as_of"] = as_of
    return out[RATING_COLS]


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #

def elo_history(
    games: pd.DataFrame,
    k: float = 24.0,
    home_adv: float = 90.0,
    carry: float = 0.8,
    new_team_start: float = 1400.0,
) -> pd.DataFrame:
    """Elo for every team after every game, across all seasons in `games`.

    `games`: one row per D1-vs-D1 game with season, game_date, team_a, team_b,
    score_a, score_b, loc_a (+1 when a is home). All games on one date use the
    ratings from the start of that date, so order within a day does not matter.
    Between seasons a team keeps `carry` of its distance from 1500; a team
    with no prior season starts at `new_team_start` (1500 in the first season).

    Output has one row per team per game plus one start-of-season row per team:
        team, season, game_date, elo_pre, elo_post, opp, opp_elo_pre, is_start
    """
    cols = ["team", "season", "game_date", "elo_pre", "elo_post", "opp", "opp_elo_pre", "is_start"]
    if games.empty:
        return pd.DataFrame(columns=cols)
    games = games.sort_values(["season", "game_date"], kind="stable")
    ratings: dict[str, float] = {}
    rows = []
    first_season = games["season"].min()
    for season, sg in games.groupby("season", sort=True):
        teams_now = set(sg["team_a"]) | set(sg["team_b"])
        for team in teams_now:
            if team in ratings:
                ratings[team] = ELO_BASE + carry * (ratings[team] - ELO_BASE)
            else:
                ratings[team] = ELO_BASE if season == first_season else new_team_start
            rows.append((team, season, pd.Timestamp(season_start(season)), np.nan, ratings[team], None, np.nan, True))

        for date, dg in sg.groupby("game_date", sort=True):
            delta: dict[str, float] = defaultdict(float)
            seen = []
            for a, b, sa, sb, loc in zip(dg["team_a"], dg["team_b"], dg["score_a"], dg["score_b"], dg["loc_a"]):
                ra, rb = ratings[a], ratings[b]
                diff = ra - rb + home_adv * (0.0 if pd.isna(loc) else loc)
                exp_a = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
                res = 1.0 if sa > sb else 0.0 if sa < sb else 0.5
                winner_edge = diff if res >= 0.5 else -diff
                mult = np.log(abs(sa - sb) + 1.0) * 2.2 / (0.001 * winner_edge + 2.2)
                d = k * mult * (res - exp_a)
                delta[a] += d
                delta[b] -= d
                seen.append((a, b, ra, rb))
            for team, d in delta.items():
                ratings[team] += d
            for a, b, ra, rb in seen:
                rows.append((a, season, date, ra, ratings[a], b, rb, False))
                rows.append((b, season, date, rb, ratings[b], a, ra, False))
    out = pd.DataFrame(rows, columns=cols)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out


def elo_asof(history: pd.DataFrame, season: int, as_of) -> pd.DataFrame:
    """Latest Elo per team and games played, using rows dated on or before `as_of`."""
    as_of = pd.Timestamp(as_of)
    h = history[(history["season"] == season) & (history["game_date"] <= as_of)]
    if h.empty:
        return pd.DataFrame(columns=["team", "elo", "elo_games"])
    h = h.sort_values(["team", "game_date"], kind="stable")
    last = h.groupby("team").tail(1).set_index("team")["elo_post"].rename("elo")
    games = h[~h["is_start"]].groupby("team").size().rename("elo_games")
    return pd.concat([last, games], axis=1).fillna({"elo_games": 0}).astype({"elo_games": int}).reset_index()


# --------------------------------------------------------------------------- #
# strength of schedule
# --------------------------------------------------------------------------- #

def sos_asof(history: pd.DataFrame, ratings: pd.DataFrame, season: int, as_of) -> pd.DataFrame:
    """Strength of schedule from games played on or before `as_of`.

    sos_elo: mean opponent Elo BEFORE each game, so a team's own result never
             feeds its schedule strength.
    sos_adj: mean opponent adj_em as of the date. It still carries a small
             effect from the games against the team itself.
    """
    as_of = pd.Timestamp(as_of)
    h = history[(history["season"] == season) & (history["game_date"] <= as_of) & ~history["is_start"]]
    if h.empty:
        return pd.DataFrame(columns=["team", "sos_elo", "sos_adj"])
    em = ratings.set_index("team")["adj_em"] if not ratings.empty else pd.Series(dtype=float)
    h = h.assign(opp_adj_em=h["opp"].map(em))
    g = h.groupby("team")
    return pd.DataFrame({"sos_elo": g["opp_elo_pre"].mean(), "sos_adj": g["opp_adj_em"].mean()}).reset_index()
