"""Build point-in-time tournament datasets and write them to artifacts/snapshots/.

    python scripts/build_snapshots.py                     # all seasons in the DB
    python scripts/build_snapshots.py --seasons 2024 2025

Outputs (CSV): tournament_games (labels: winner + both scores), team_snapshots,
player_snapshots, cutoffs. Prints a coverage/verification summary.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.db import get_pool, query  # noqa: E402
from data.snapshots import build_tournament_dataset  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "snapshots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", help="season ending years (default: all with tournament data)")
    args = ap.parse_args()

    seasons = args.seasons or [
        r["ending_year"]
        for r in query(
            """SELECT DISTINCT ss.ending_year FROM team_seasons ts
               JOIN school_seasons ss ON ss.id = ts.school_season_id
               WHERE ts.ncaa_seed IS NOT NULL ORDER BY 1"""
        )
    ]
    print(f"building snapshots for seasons {seasons}")
    ds = build_tournament_dataset(seasons)

    os.makedirs(OUT_DIR, exist_ok=True)
    ds["games"].to_csv(os.path.join(OUT_DIR, "tournament_games.csv"), index=False)
    ds["teams"].to_csv(os.path.join(OUT_DIR, "team_snapshots.csv"), index=False)
    ds["players"].to_csv(os.path.join(OUT_DIR, "player_snapshots.csv"), index=False)
    ds["cutoffs"].to_csv(os.path.join(OUT_DIR, "cutoffs.csv"), index=False)

    games, teams = ds["games"], ds["teams"]
    print("\ncutoffs (Selection Sunday):")
    print(ds["cutoffs"].to_string(index=False))
    print("\ntournament games per season (usable = both teams have a seeded snapshot):")
    print(
        games.groupby("season")
        .agg(games=("a_win", "size"), usable=("both_have_snapshot", "sum"), with_box=("has_box_score", "sum"))
        .to_string()
    )
    print("\nteam snapshots per season (tournament teams = seeded):")
    print(teams.groupby("season").agg(teams=("team", "size"), seeded=("ncaa_seed", "count"), prior_season=("prior_season_available", "max")).to_string())
    print("\nseeded teams: share of scheduled games that have a box score (feeds efficiency/four-factor stats):")
    seeded = teams[teams["ncaa_seed"].notna()].assign(missing=lambda d: d["games"] - d["box_games"])
    print(
        seeded.groupby("season")
        .agg(teams=("team", "size"), full_coverage=("missing", lambda s: (s == 0).sum()),
             mean_coverage=("box_coverage", "mean"), worst_coverage=("box_coverage", "min"),
             max_games_missing=("missing", "max"))
        .round(3)
        .to_string()
    )
    print(f"\nwrote CSVs to {os.path.normpath(OUT_DIR)}")
    get_pool().close()


if __name__ == "__main__":
    main()
