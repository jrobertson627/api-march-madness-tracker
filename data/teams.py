"""Team/season data backed by the cbb-web-scraper Supabase (Postgres) database.

Replaces the earlier hard-coded seed module; the function names and response
shapes are unchanged so the API routes didn't need to change.

Tables used: schools, school_seasons, team_seasons, team_season_stats
(side = 'team' | 'opponent'), game_teams, games.

Team ids are the Sports-Reference school slug (e.g. "michigan-state"), taken
from schools.canonical_source_path.
"""

from data.db import query

SEASON = 2026  # season ending year

_PATH_PREFIX = "www.sports-reference.com/cbb/schools/"
_PATH_SUFFIX = "/men"

_TEAM_COLUMNS = """
    split_part(s.canonical_source_path, '/', 4) AS id,
    s.display_name AS name,
    ts.ncaa_seed AS seed,
    regexp_replace(ts.conference_name, ' MBB$', '') AS conference,
    ts.wins,
    ts.losses
"""


def _f(value, ndigits=1):
    return None if value is None else round(float(value), ndigits)


def _team_header(row):
    record = None
    if row["wins"] is not None and row["losses"] is not None:
        record = f"{row['wins']}-{row['losses']}"
    return {
        "id": row["id"],
        "name": row["name"],
        "seed": row["seed"],
        "conference": row["conference"],
        "record": record,
    }


def list_teams(season=SEASON):
    """NCAA tournament teams (those with a seed) for the season, best seed first."""
    rows = query(
        f"""
        SELECT {_TEAM_COLUMNS}
        FROM schools s
        JOIN school_seasons ss ON ss.school_id = s.id AND ss.ending_year = %s
        JOIN team_seasons ts ON ts.school_season_id = ss.id
        WHERE ts.ncaa_seed IS NOT NULL
        ORDER BY ts.ncaa_seed, s.display_name
        """,
        (season,),
    )
    return [_team_header(r) for r in rows]


def get_team_stats(team_id, season=SEASON):
    rows = query(
        f"""
        SELECT {_TEAM_COLUMNS},
            ts.sos, ts.off_rtg, ts.def_rtg,
            t.fg, t.fga, t.fg3, t.fta, t.tov, t.trb,
            o.trb AS opp_trb,
            (
                SELECT avg(m.margin) FROM (
                    SELECT gt.final_score - opp.final_score AS margin
                    FROM game_teams gt
                    JOIN game_teams opp ON opp.game_id = gt.game_id AND opp.id <> gt.id
                    JOIN games g ON g.id = gt.game_id
                    WHERE gt.team_source_path = s.canonical_source_path
                      AND g.game_status = 'final'
                      AND g.game_date BETWEEN make_date(ss.ending_year - 1, 10, 1)
                                          AND make_date(ss.ending_year, 7, 1)
                    ORDER BY g.game_date DESC
                    LIMIT 10
                ) m
            ) AS recent_form
        FROM schools s
        JOIN school_seasons ss ON ss.school_id = s.id AND ss.ending_year = %s
        JOIN team_seasons ts ON ts.school_season_id = ss.id
        LEFT JOIN team_season_stats t ON t.school_season_id = ss.id AND t.side = 'team'
        LEFT JOIN team_season_stats o ON o.school_season_id = ss.id AND o.side = 'opponent'
        WHERE s.canonical_source_path = %s
        """,
        (season, f"{_PATH_PREFIX}{team_id}{_PATH_SUFFIX}"),
    )
    if not rows:
        return None
    r = rows[0]

    efficiency_margin = None
    if r["off_rtg"] is not None and r["def_rtg"] is not None:
        efficiency_margin = _f(r["off_rtg"] - r["def_rtg"])

    turnover_rate = None
    if None not in (r["fga"], r["fta"], r["tov"]):
        possessions = r["fga"] + 0.44 * r["fta"] + r["tov"]
        turnover_rate = _f(100 * r["tov"] / possessions) if possessions else None

    rebound_rate = None
    if r["trb"] is not None and r["opp_trb"] is not None and (r["trb"] + r["opp_trb"]):
        rebound_rate = _f(100 * r["trb"] / (r["trb"] + r["opp_trb"]))

    efg_pct = None
    if None not in (r["fg"], r["fg3"]) and r["fga"]:
        efg_pct = _f(100 * (r["fg"] + 0.5 * r["fg3"]) / r["fga"])

    return {
        **_team_header(r),
        "season": season,
        "stats": {
            "efficiency_margin": efficiency_margin,  # off_rtg - def_rtg
            "sos": _f(r["sos"]),
            "turnover_rate": turnover_rate,  # TOV / (FGA + 0.44*FTA + TOV)
            "rebound_rate": rebound_rate,  # TRB / (TRB + opp TRB)
            "efg_pct": efg_pct,  # (FG + 0.5*3P) / FGA
            "recent_form": _f(r["recent_form"]),  # avg margin, last 10 games
        },
    }
