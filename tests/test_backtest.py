import unittest

import numpy as np
import pandas as pd

from api import backtest as bt
from api.sample import sample_backtest


def scores():
    rows = []
    for season, events in ((2022, 100), (2023, 300)):
        for method, ll in (("coin", 0.33), ("seed", 0.28), ("eff", 0.27)):
            rows.append((season, method, 0.0, ll, 0.1, events))
        for tau, ll in ((0.0, 0.25), (0.1, 0.20), (0.3, 0.10)):
            rows.append((season, "model", tau, ll + (0.02 if season == 2023 else 0), 0.09, events))
    return pd.DataFrame(rows, columns=["season", "method", "tau", "log_loss", "brier", "events"])


class AdvancementTests(unittest.TestCase):
    def test_only_the_chosen_tau_is_used_and_methods_get_plain_names(self):
        rows = bt.advancement_rows(scores(), tau=0.0)
        self.assertEqual({r["method"] for r in rows}, {"coin flip", "seed", "efficiency gap", "model"})
        model = [r for r in rows if r["method"] == "model" and r["season"] != "all"]
        self.assertEqual(sorted(r["log_loss"] for r in model), [0.25, 0.27])  # the tau 0.1 and 0.3 rows are not in the report

    def test_the_pooled_row_weights_each_season_by_its_events(self):
        pooled = next(r for r in bt.advancement_rows(scores()) if r["season"] == "all" and r["method"] == "model")
        self.assertAlmostEqual(pooled["log_loss"], (0.25 * 100 + 0.27 * 300) / 400, places=4)
        self.assertEqual(pooled["events"], 400)

    def test_the_locked_scores_accept_the_model_name_with_its_tau(self):
        locked = {"season": 2026, "scores": [{"method": "coin", "log_loss": 0.33, "brier": 0.1, "events": 408}, {"method": "model (tau 0.0)", "log_loss": 0.19, "brier": 0.06, "events": 408},
                                              {"method": "mystery", "log_loss": 1.0, "brier": 1.0, "events": 1}]}
        rows = bt.locked_advancement_rows(locked)
        self.assertEqual([(r["season"], r["method"]) for r in rows], [(2026, "coin flip"), (2026, "model")])


class GameRowTests(unittest.TestCase):
    EXP = pd.DataFrame([("A", "loso", "game", "2022", 67, 0.6, 0.2, 0.7), ("A", "loso", "game", "all", 134, 0.58, 0.19, 0.71), ("A", "loso", "bracket", "2022", 67, 0.9, 0.9, 0.1),
                        ("A", "locked", "game", "2026", 67, 0.5, 0.17, 0.72), ("B", "loso", "game", "2022", 67, 0.7, 0.25, 0.5)],
                       columns=["name", "protocol", "mode", "season", "n", "log_loss", "brier", "accuracy"])

    def test_only_per_game_rows_of_the_named_runs_are_used(self):
        rows = bt.game_rows(self.EXP, [("model", "A")], protocol="loso")
        self.assertEqual([(r["season"], r["n"]) for r in rows], [(2022, 67), ("all", 134)])  # no bracket-mode row, no locked row, no run B

    def test_without_a_protocol_every_protocol_is_read(self):
        self.assertEqual(len(bt.game_rows(self.EXP, [("model", "A")])), 3)


class PairedTests(unittest.TestCase):
    def test_the_sign_and_the_zero_flag(self):
        inside = bt.paired_row({"games": 267, "mean_diff": -0.012, "lo": -0.026, "hi": 0.003, "season_wins": 4, "seasons": 4}, "model", "efficiency gap")
        self.assertFalse(inside["interval_excludes_zero"])
        self.assertEqual((inside["diff"], inside["seasons_ahead"]), (-0.012, 4))
        better = bt.paired_row({"games": 10, "mean_diff": -0.05, "lo": -0.08, "hi": -0.01, "season_wins": 1, "seasons": 1}, "m", "b")
        worse = bt.paired_row({"games": 10, "mean_diff": 0.05, "lo": 0.01, "hi": 0.08, "season_wins": 0, "seasons": 1}, "m", "b")
        self.assertTrue(better["interval_excludes_zero"] and worse["interval_excludes_zero"])


class UpsetTests(unittest.TestCase):
    FLAGS = pd.DataFrame([(2022, "East", "www.sports-reference.com/cbb/schools/a/men", "www.sports-reference.com/cbb/schools/b/men", 1, 16, 0.05, False, False, False, False),
                          (2022, "East", "www.sports-reference.com/cbb/schools/c/men", "www.sports-reference.com/cbb/schools/d/men", 8, 9, 0.55, True, False, True, True),
                          (2023, "West", "www.sports-reference.com/cbb/schools/e/men", "www.sports-reference.com/cbb/schools/f/men", 5, 12, 0.30, False, True, True, False),
                          (2023, "West", "www.sports-reference.com/cbb/schools/g/men", "www.sports-reference.com/cbb/schools/h/men", 6, 11, 0.30, False, False, False, True)],
                         columns=["season", "region", "favorite", "underdog", "favorite_seed", "underdog_seed", "p_underdog_model", "model_picks_upset", "model_above_history", "flagged", "upset"])

    def test_slugs_names_and_counts(self):
        rows, s = bt.upset_flags(self.FLAGS, {"www.sports-reference.com/cbb/schools/c/men": "Team C"})
        self.assertEqual((rows[1]["favorite"], rows[1]["favorite_name"], rows[1]["underdog_name"]), ("c", "Team C", None))
        self.assertEqual(s["all_games"], {"games": 4, "upsets": 2})
        self.assertEqual(s["model_picks_upset"], {"games": 1, "upsets": 1})
        self.assertEqual(s["model_above_history"], {"games": 1, "upsets": 0})
        self.assertEqual(s["no_flag"], {"games": 2, "upsets": 1})  # games where neither flag was raised

    def test_the_limits_text_uses_the_counts_it_was_given(self):
        report = sample_backtest()["report"]
        text = " ".join(report["limits"])
        s = report["selection"]["upset_summary"]
        self.assertIn(f"{s['model_picks_upset']['upsets']} of {s['model_picks_upset']['games']}", text)
        self.assertIn("includes zero", text)


class ReportTests(unittest.TestCase):
    def test_wording_changes_when_an_interval_excludes_zero(self):
        base = sample_backtest()["report"]
        sel = dict(base["selection"], paired=[dict(base["selection"]["paired"][0], interval_excludes_zero=True)])
        lines = bt.limits_text(sel, base["locked"], sel["upset_summary"])
        self.assertTrue(any("others exclude zero" in l for l in lines))
        self.assertFalse(any(l.startswith("Every paired") for l in lines))

    def test_a_report_without_a_locked_season_still_builds(self):
        base = sample_backtest()["report"]["selection"]
        report = bt.build_report(seasons=base["seasons"], advancement=base["advancement"], games_scored=base["games_scored"], picks=base["picks"], flags=base["upset_flags"],
                                 flag_summary=base["upset_summary"], paired=base["paired"], locked=None)
        self.assertIsNone(report["locked"])
        self.assertFalse(any("scored once" in l for l in report["limits"]))

    def test_no_nan_reaches_the_report(self):
        rows = bt.advancement_rows(pd.DataFrame([(2022, "coin", 0.0, np.nan, 0.1, 10)], columns=["season", "method", "tau", "log_loss", "brier", "events"]).fillna(0.0))
        self.assertEqual(rows[0]["log_loss"], 0.0)
        self.assertIsNone(bt._num(float("nan")))


if __name__ == "__main__":
    unittest.main()
