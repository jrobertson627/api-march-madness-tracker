import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bracket import pick, upsets
from bracket.outputs import bracket_json, picks_table, write_outputs
from bracket.simulate import exact_levels
from bracket.structure import Bracket
from tests.test_bracket_core import full_field, strength_matrix


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.b = Bracket.from_teams(full_field())
        self.P = strength_matrix(len(self.b.teams), 7, scale=1.2)
        self.lv = exact_levels(self.b, self.P)
        self.picks = {"best expected points": pick.best_expected_bracket(self.b, self.lv), "model favorite": pick.favorite_bracket(self.b, self.P)}

    def test_picks_table_has_one_row_per_game_and_consistent_picks(self):
        t = picks_table(self.b, self.picks["best expected points"], self.lv)
        self.assertEqual(len(t), 63)
        self.assertEqual(t.groupby("round").size().to_dict(), {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1})
        self.assertTrue(((t.pick == t.team_a) | (t.pick == t.team_b)).all())
        self.assertTrue(((t.p_pick_wins_game > 0) & (t.p_pick_wins_game <= 1)).all())

    def test_json_contains_every_team_and_each_pick_set(self):
        doc = bracket_json(self.b, self.lv, self.picks, {"season": 2030})
        self.assertEqual(len(doc["teams"]), 68)
        self.assertEqual(set(doc["picks"]), set(self.picks))
        self.assertAlmostEqual(sum(t["p_champion"] for t in doc["teams"]), 1.0)
        json.dumps(doc)  # serializable

    def test_write_outputs_creates_the_files(self):
        flags = upsets.first_round_table(self.b, self.lv, {p: 0.2 for p in upsets.SEED_PAIRS})
        with tempfile.TemporaryDirectory() as d:
            paths = write_outputs(Path(d), 2030, self.b, self.lv, self.picks, flags, {"season": 2030}, "report text")
            names = sorted(p.name for p in paths)
            self.assertIn("advancement_2030.csv", names)
            self.assertIn("first_round_2030.csv", names)
            self.assertIn("picks_2030_best_expected_points.csv", names)
            self.assertIn("bracket_2030.json", names)
            self.assertIn("report_2030.txt", names)
            adv = pd.read_csv(Path(d) / "advancement_2030.csv")
            self.assertEqual(len(adv), 68)


if __name__ == "__main__":
    unittest.main()
