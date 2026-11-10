"""Leakage and math tests for data/snapshots.py. Pure pandas, no database needed.

    python -m unittest discover -s tests
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# data.db imports psycopg at import time; snapshots imports data.db, which is fine
# as long as DATABASE_URL isn't touched (the pool is created lazily).
from data.snapshots import (  # noqa: E402
    STAT_COLS,
    compute_player_snapshots,
    compute_roster_features,
    compute_team_snapshots,
    selection_sunday,
)


def _game(game_id, date, team, opp, pts_for, pts_against, **box):
    row = {
        "game_id": game_id, "game_date": pd.Timestamp(date), "season": 2024,
        "team": team, "team_name": team.upper(), "opp": opp,
        "pts_for": pts_for, "pts_against": pts_against,
    }
    for c in STAT_COLS:
        row[c] = box.get(c, 10)
        row[f"o_{c}"] = box.get(f"o_{c}", 10)
    return row


class SelectionSundayTest(unittest.TestCase):
    def test_first_four_tuesday(self):
        self.assertEqual(selection_sunday("2024-03-19"), pd.Timestamp("2024-03-17"))

    def test_thursday_start_without_first_four(self):
        self.assertEqual(selection_sunday("2021-03-18"), pd.Timestamp("2021-03-14"))

    def test_always_a_sunday_strictly_before(self):
        for d in pd.date_range("2024-03-10", "2024-03-24"):
            s = selection_sunday(d)
            self.assertEqual(s.weekday(), 6)
            self.assertLess(s, d)


class TeamSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame(
            [
                _game(1, "2024-01-01", "a", "b", 80, 70),
                _game(2, "2024-02-01", "a", "c", 60, 65),
                _game(3, "2024-03-20", "a", "d", 99, 40),  # after cutoff
            ]
        )

    def test_games_after_as_of_are_excluded(self):
        snap = compute_team_snapshots(self.games, 2024, "2024-03-17").set_index("team")
        self.assertEqual(snap.loc["a", "games"], 2)
        self.assertEqual(snap.loc["a", "wins"], 1)
        self.assertLessEqual(snap.loc["a", "last_game_date"], pd.Timestamp("2024-03-17"))

    def test_changing_post_cutoff_results_changes_nothing(self):
        before = compute_team_snapshots(self.games, 2024, "2024-03-17")
        poisoned = self.games.copy()
        post = poisoned["game_date"] > pd.Timestamp("2024-03-17")
        poisoned.loc[post, ["pts_for", "pts_against"]] = [0, 200]
        poisoned.loc[post, STAT_COLS] = 999
        after = compute_team_snapshots(poisoned, 2024, "2024-03-17")
        pd.testing.assert_frame_equal(before, after)

    def test_as_of_is_inclusive(self):
        snap = compute_team_snapshots(self.games, 2024, "2024-03-20").set_index("team")
        self.assertEqual(snap.loc["a", "games"], 3)

    def test_efficiency_math(self):
        g = pd.DataFrame([_game(1, "2024-01-01", "a", "b", 100, 80, fga=60, orb=10, tov=10, fta=0, o_fga=60, o_orb=10, o_tov=10, o_fta=0)])
        snap = compute_team_snapshots(g, 2024, "2024-02-01").iloc[0]
        poss = 60 - 10 + 10  # fga - orb + tov (+0.44*fta, fta=0)
        self.assertAlmostEqual(snap["tempo"], poss)
        self.assertAlmostEqual(snap["off_eff"], 100 * 100 / poss)
        self.assertAlmostEqual(snap["def_eff"], 100 * 80 / poss)
        self.assertAlmostEqual(snap["eff_margin"], snap["off_eff"] - snap["def_eff"])

    def test_other_seasons_ignored(self):
        g = self.games.copy()
        g["season"] = 2023
        self.assertTrue(compute_team_snapshots(g, 2024, "2024-03-17").empty)


class PlayerSnapshotTest(unittest.TestCase):
    def _pg(self, date, team, pid, name, minutes, pts):
        row = {"game_date": pd.Timestamp(date), "season": 2024, "team": team, "player_id": pid,
               "player_name": name, "starter": 1, "minutes": minutes, "pts": pts}
        for c in STAT_COLS:
            row[c] = 1
        return row

    def test_excludes_after_cutoff_and_handles_missing_id(self):
        pg = pd.DataFrame(
            [
                self._pg("2024-01-01", "a", 1, "P1", 30, 10),
                self._pg("2024-01-01", "a", None, "NoId", 10, 4),
                self._pg("2024-03-20", "a", 1, "P1", 40, 50),  # after cutoff
            ]
        )
        p = compute_player_snapshots(pg, 2024, "2024-03-17", prior=None).set_index("player_name")
        self.assertEqual(p.loc["P1", "pts"], 10)
        self.assertEqual(p.loc["NoId", "games"], 1)
        self.assertAlmostEqual(p["min_share"].sum(), 1.0)

    def test_prior_carryover_and_returning_minutes(self):
        pg = pd.DataFrame(
            [self._pg("2024-01-01", "a", 1, "Stay", 30, 10), self._pg("2024-01-01", "a", 2, "Transfer", 10, 4)]
        )
        prior = pd.DataFrame(
            [
                {"team": "a", "player_id": 1, "player_name": "Stay", "games": 30, "minutes": 600.0, "bpm": 4.0, "ws": 3.0, "per": 20.0, "usg_pct": 20.0},
                {"team": "a", "player_id": 9, "player_name": "Gone", "games": 30, "minutes": 400.0, "bpm": 0.0, "ws": 1.0, "per": 10.0, "usg_pct": 15.0},
                {"team": "z", "player_id": 2, "player_name": "Transfer", "games": 30, "minutes": 300.0, "bpm": 2.0, "ws": 2.0, "per": 15.0, "usg_pct": 18.0},
            ]
        )
        p = compute_player_snapshots(pg, 2024, "2024-03-17", prior)
        r = compute_roster_features(p, prior).set_index("team").loc["a"]
        self.assertAlmostEqual(r["returning_min_pct"], 600 / 1000)  # only "Stay" returns
        self.assertAlmostEqual(r["incoming_prior_min"], 300)
        self.assertAlmostEqual(r["returning_bpm_wavg"], 4.0)
        self.assertEqual(r["prior_season_available"], 1)

    def test_no_prior_season_gives_nan_not_zero(self):
        pg = pd.DataFrame([self._pg("2024-01-01", "a", 1, "P", 30, 10)])
        p = compute_player_snapshots(pg, 2024, "2024-03-17", prior=None)
        r = compute_roster_features(p, None).iloc[0]
        self.assertTrue(np.isnan(r["returning_min_pct"]))
        self.assertEqual(r["prior_season_available"], 0)


if __name__ == "__main__":
    unittest.main()
