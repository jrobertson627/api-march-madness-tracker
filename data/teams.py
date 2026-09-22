"""Seed team/season data shaped like the web-scraper's Postgres schema.

Field names mirror what the scraper's `schools` / `school_seasons` /
`team_game_stats` tables will eventually hold, so a future repository
implementation backed by that real database can replace this module
without changing the API routes that call it.

This is placeholder data for one bracket region (2025 seeds) until the
scraper has a working Sports-Reference adapter and permission to crawl.
"""

SEASON = 2026

_TEAMS = [
    {
        "id": "auburn", "name": "Auburn", "seed": 1, "conference": "SEC", "record": "28-5",
        "stats": {"efficiency_margin": 25.8, "sos": 10.9, "turnover_rate": 14.8, "rebound_rate": 54.3, "efg_pct": 56.2, "recent_form": 13.1},
    },
    {
        "id": "alabama-st", "name": "Alabama St.", "seed": 16, "conference": "SWAC", "record": "20-13",
        "stats": {"efficiency_margin": -8.2, "sos": -6.4, "turnover_rate": 19.5, "rebound_rate": 47.8, "efg_pct": 46.9, "recent_form": -3.5},
    },
    {
        "id": "louisville", "name": "Louisville", "seed": 8, "conference": "ACC", "record": "24-9",
        "stats": {"efficiency_margin": 15.4, "sos": 8.1, "turnover_rate": 16.9, "rebound_rate": 51.2, "efg_pct": 52.8, "recent_form": 6.7},
    },
    {
        "id": "creighton", "name": "Creighton", "seed": 9, "conference": "Big East", "record": "22-11",
        "stats": {"efficiency_margin": 13.9, "sos": 7.6, "turnover_rate": 15.7, "rebound_rate": 50.4, "efg_pct": 53.6, "recent_form": 9.2},
    },
    {
        "id": "michigan", "name": "Michigan", "seed": 5, "conference": "Big Ten", "record": "24-10",
        "stats": {"efficiency_margin": 18.6, "sos": 9.3, "turnover_rate": 15.1, "rebound_rate": 52.7, "efg_pct": 51.4, "recent_form": 7.8},
    },
    {
        "id": "uc-san-diego", "name": "UC San Diego", "seed": 12, "conference": "Big West", "record": "27-8",
        "stats": {"efficiency_margin": 12.1, "sos": 1.8, "turnover_rate": 13.9, "rebound_rate": 49.6, "efg_pct": 54.1, "recent_form": 10.4},
    },
    {
        "id": "texas-am", "name": "Texas A&M", "seed": 4, "conference": "SEC", "record": "22-10",
        "stats": {"efficiency_margin": 19.7, "sos": 9.8, "turnover_rate": 17.2, "rebound_rate": 53.9, "efg_pct": 50.7, "recent_form": 5.9},
    },
    {
        "id": "yale", "name": "Yale", "seed": 13, "conference": "Ivy", "record": "23-6",
        "stats": {"efficiency_margin": 9.8, "sos": 0.4, "turnover_rate": 14.2, "rebound_rate": 48.9, "efg_pct": 52.3, "recent_form": 8.6},
    },
    {
        "id": "ole-miss", "name": "Ole Miss", "seed": 6, "conference": "SEC", "record": "20-11",
        "stats": {"efficiency_margin": 16.3, "sos": 8.9, "turnover_rate": 16.4, "rebound_rate": 51.8, "efg_pct": 51.9, "recent_form": 4.3},
    },
    {
        "id": "north-carolina", "name": "North Carolina", "seed": 11, "conference": "ACC", "record": "21-13",
        "stats": {"efficiency_margin": 14.7, "sos": 8.4, "turnover_rate": 17.8, "rebound_rate": 50.9, "efg_pct": 52.1, "recent_form": 3.1},
    },
    {
        "id": "iowa-state", "name": "Iowa State", "seed": 3, "conference": "Big 12", "record": "23-9",
        "stats": {"efficiency_margin": 21.4, "sos": 9.6, "turnover_rate": 13.5, "rebound_rate": 51.5, "efg_pct": 53.4, "recent_form": 11.8},
    },
    {
        "id": "lipscomb", "name": "Lipscomb", "seed": 14, "conference": "ASUN", "record": "26-7",
        "stats": {"efficiency_margin": 6.5, "sos": -3.1, "turnover_rate": 15.9, "rebound_rate": 49.1, "efg_pct": 51.5, "recent_form": 6.9},
    },
    {
        "id": "marquette", "name": "Marquette", "seed": 7, "conference": "Big East", "record": "22-10",
        "stats": {"efficiency_margin": 15.9, "sos": 8.7, "turnover_rate": 15.3, "rebound_rate": 50.2, "efg_pct": 52.6, "recent_form": 5.4},
    },
    {
        "id": "new-mexico", "name": "New Mexico", "seed": 10, "conference": "Mountain West", "record": "25-9",
        "stats": {"efficiency_margin": 13.2, "sos": 3.5, "turnover_rate": 16.1, "rebound_rate": 52.4, "efg_pct": 50.9, "recent_form": 8.9},
    },
    {
        "id": "michigan-st", "name": "Michigan St.", "seed": 2, "conference": "Big Ten", "record": "27-7",
        "stats": {"efficiency_margin": 22.6, "sos": 10.2, "turnover_rate": 14.6, "rebound_rate": 55.1, "efg_pct": 52.9, "recent_form": 10.7},
    },
    {
        "id": "bryant", "name": "Bryant", "seed": 15, "conference": "America East", "record": "21-11",
        "stats": {"efficiency_margin": -4.8, "sos": -5.9, "turnover_rate": 18.3, "rebound_rate": 47.5, "efg_pct": 47.8, "recent_form": -1.2},
    },
]

_TEAMS_BY_ID = {team["id"]: team for team in _TEAMS}


def list_teams():
    return [
        {"id": t["id"], "name": t["name"], "seed": t["seed"], "conference": t["conference"], "record": t["record"]}
        for t in _TEAMS
    ]


def get_team_stats(team_id, season=SEASON):
    team = _TEAMS_BY_ID.get(team_id)
    if team is None:
        return None
    return {
        "id": team["id"],
        "name": team["name"],
        "seed": team["seed"],
        "conference": team["conference"],
        "record": team["record"],
        "season": season,
        "stats": team["stats"],
    }
