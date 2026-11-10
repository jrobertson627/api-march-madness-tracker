import unittest

import numpy as np
import pandas as pd

from ml.lines import BETIQ_ALIASES, betiq_to_games, devig, match_to_matrix, ml_to_prob, read_lines, season_of
from ml.team_names import match_names

SR = "www.sports-reference.com/cbb/schools/{}/men"
TABLE = pd.DataFrame(
    {
        "team": [SR.format(s) for s in ("connecticut", "michigan", "duke", "north-carolina-at", "arkansas-little-rock", "appalachian-state")],
        "display_name": ["Connecticut Huskies", "Michigan Wolverines", "Duke Blue Devils", "North Carolina A&T Aggies",
                         "Arkansas-Little Rock Trojans", "Appalachian State Mountaineers"],
    }
)
UC, MI, DU = SR.format("connecticut"), SR.format("michigan"), SR.format("duke")

TWO_ROWS = pd.DataFrame(
    {
        "Date": ["2026-04-06", "2026-04-06"],
        "Team": ["Michigan", "UConn"],
        "Opponent": ["UConn", "Michigan"],
        "Location": ["Neutral", "Neutral"],
        "Spread": ["-6.5", "6.5"],
        "Total (O/U)": ["146.5", "146.5"],
        "Money Line": ["-300", "240"],
        "Score": ["69-63", "63-69"],
        "MOV": ["6", "-6"],
    }
)


class ConversionTests(unittest.TestCase):
    def test_moneyline_probabilities(self):
        self.assertAlmostEqual(float(ml_to_prob(-300)), 0.75)
        self.assertAlmostEqual(float(ml_to_prob(240)), 100 / 340)
        self.assertAlmostEqual(float(ml_to_prob(100)), 0.5)
        p = devig(ml_to_prob(240), ml_to_prob(-300))
        self.assertAlmostEqual(float(p), (100 / 340) / (100 / 340 + 0.75))
        self.assertAlmostEqual(float(p + devig(ml_to_prob(-300), ml_to_prob(240))), 1.0)

    def test_season_from_date(self):
        s = season_of(pd.Series(["2025-11-04", "2026-03-20", "2026-08-01", "2026-07-31"]))
        self.assertEqual(s.tolist(), [2026, 2026, 2027, 2026])


class AdapterTests(unittest.TestCase):
    def test_two_team_rows_become_one_game_in_canonical_order(self):
        g, rep = betiq_to_games(TWO_ROWS, TABLE, sigma=11.0)
        self.assertEqual(len(g), 1)
        r = g.iloc[0]
        self.assertEqual((r["team_a"], r["team_b"]), (UC, MI))  # connecticut sorts before michigan
        self.assertEqual(r["season"], 2026)
        self.assertEqual(r["spread_a"], -6.5)  # UConn is the underdog by 6.5, so A is "favored by -6.5"
        self.assertEqual(r["total"], 146.5)
        self.assertEqual((r["ml_a"], r["ml_b"]), (240.0, -300.0))
        self.assertEqual((r["score_a"], r["score_b"], r["margin_a"]), (63.0, 69.0, -6.0))
        self.assertAlmostEqual(r["p_ml"], (100 / 340) / (100 / 340 + 0.75))
        self.assertLess(r["p_spread"], 0.5)
        self.assertEqual(r["p_a_win"], r["p_ml"])
        self.assertEqual(rep["games"], 1)
        self.assertEqual(rep["rows_dropped_unmatched"], 0)

    def test_a_game_with_only_team_b_row_is_flipped_correctly(self):
        g, _ = betiq_to_games(TWO_ROWS.iloc[[0]], TABLE, sigma=11.0)  # only Michigan's row (Michigan is team B)
        r = g.iloc[0]
        self.assertEqual(r["spread_a"], -6.5)  # Michigan -6.5 means A (UConn) is the underdog by 6.5
        self.assertTrue(np.isnan(r["ml_a"]) and r["ml_b"] == -300)
        self.assertTrue(np.isnan(r["p_ml"]))
        self.assertEqual(r["p_a_win"], r["p_spread"])  # falls back to the spread
        self.assertEqual(r["margin_a"], -6.0)

    def test_blank_lines_stay_missing(self):
        df = TWO_ROWS.copy()
        df["Spread"], df["Money Line"] = ["-", ""], ["NA", "pk"]
        g, rep = betiq_to_games(df, TABLE, sigma=11.0)
        self.assertTrue(np.isnan(g.iloc[0]["p_a_win"]))
        self.assertEqual(rep["games_without_any_line"], 1)

    def test_unmatched_teams_are_reported_and_their_rows_dropped(self):
        df = pd.concat([TWO_ROWS, TWO_ROWS.assign(Team=["Gonzaga", "Michigan"], Opponent=["Michigan", "Gonzaga"])], ignore_index=True)
        g, rep = betiq_to_games(df, TABLE, sigma=11.0)
        self.assertEqual(len(g), 1)
        self.assertEqual(rep["rows_dropped_unmatched"], 2)
        self.assertEqual(rep["unmatched_names"], ["Gonzaga"])

    def test_betiq_short_names_resolve(self):
        names = ["NC A&T", "AR Lit Rock", "App State", "UConn", "Duke"]
        mapping, unmatched = match_names(names, TABLE, BETIQ_ALIASES)
        self.assertEqual(unmatched, [])
        self.assertEqual(mapping["NC A&T"], SR.format("north-carolina-at"))
        self.assertEqual(mapping["App State"], SR.format("appalachian-state"))

    def test_missing_required_columns(self):
        with self.assertRaises(ValueError):
            read_lines(pd.DataFrame({"Date": ["2026-01-01"], "Team": ["Duke"]}))

    def test_sigma_is_estimated_outside_the_evaluated_season(self):
        rng = np.random.default_rng(0)
        n = 300
        rows = []
        for start, count in ((pd.Timestamp("2024-09-01"), 230), (pd.Timestamp("2025-09-01"), 60)):  # seasons 2025 and 2026
            for i in range(count):
                day = (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                margin = int(rng.normal(0, 9))
                rows += [
                    {"Date": day, "Team": "Duke", "Opponent": "UConn", "Spread": "-3.5", "Money Line": "", "Score": f"{70 + margin}-70"},
                    {"Date": day, "Team": "UConn", "Opponent": "Duke", "Spread": "3.5", "Money Line": "", "Score": f"70-{70 + margin}"},
                ]
        g, rep = betiq_to_games(pd.DataFrame(rows), TABLE, exclude_sigma_seasons=[2026])
        self.assertIn("estimated", rep["sigma_source"])
        self.assertAlmostEqual(rep["sigma"], 9.0, delta=1.5)
        g2, rep2 = betiq_to_games(pd.DataFrame(rows[:20]), TABLE)
        self.assertIn("fallback", rep2["sigma_source"])
        self.assertEqual(rep2["sigma"], 11.0)


class MatchToMatrixTests(unittest.TestCase):
    def setUp(self):
        self.lines, _ = betiq_to_games(TWO_ROWS, TABLE, sigma=11.0)

    def matrix(self, date):
        return pd.DataFrame({"game_id": ["g1"], "season": [2026], "team_a": [UC], "team_b": [MI], "game_date": [pd.Timestamp(date)]})

    def test_exact_and_one_day_off_dates_match(self):
        self.assertEqual(len(match_to_matrix(self.lines, self.matrix("2026-04-06"))), 1)
        self.assertEqual(len(match_to_matrix(self.lines, self.matrix("2026-04-07"))), 1)  # late tip, next UTC day
        self.assertEqual(len(match_to_matrix(self.lines, self.matrix("2026-04-09"))), 0)

    def test_a_pair_that_meets_twice_takes_the_nearest_line(self):
        second = self.lines.assign(game_date=pd.Timestamp("2026-04-08"), p_a_win=0.9)
        both = pd.concat([self.lines, second], ignore_index=True)
        m = match_to_matrix(both, self.matrix("2026-04-08"))
        self.assertEqual(len(m), 1)
        self.assertEqual(m.iloc[0]["p_a_win"], 0.9)


if __name__ == "__main__":
    unittest.main()
