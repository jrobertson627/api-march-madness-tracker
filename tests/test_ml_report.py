import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.experiment import Config, run_experiment, save_results
from ml.model_card import model_card, write_model_card
from ml.predictor import fit_final
from ml.report import load_results, paired, per_season, reliability_plot, summary
from ml.synthetic import make_synthetic
from ml.team_names import ALIASES, map_team_columns, match_names, normalize_name

DF = make_synthetic(seasons=(2022, 2023, 2024, 2025), n_regular=300, n_conf=50, n_ncaa=40)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.configs = [Config("B0 coin", "coin", "rating_core"), Config("M5 good", "logistic", "ratings_plus", grid={"C": [0.1, 1.0]})]
        for cfg in cls.configs:
            save_results(run_experiment(cfg, DF), Path(cls.dir.name))
        cls.exp, cls.oof = load_results(Path(cls.dir.name))
        cls.names = cls.exp.drop_duplicates("name").set_index("name")["run_id"].to_dict()

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_summary_has_one_pooled_row_per_run_sorted_by_log_loss(self):
        s = summary(self.exp, "game")
        self.assertEqual(len(s), 2)
        self.assertEqual(s.iloc[0]["name"], "M5 good")
        self.assertTrue(s["log_loss"].is_monotonic_increasing)

    def test_per_season_table_has_a_column_per_season(self):
        t = per_season(self.exp, "game")
        self.assertEqual(sorted(map(str, t.columns)), ["2022", "2023", "2024", "2025"])
        self.assertEqual(set(t.index), {"B0 coin", "M5 good"})

    def test_paired_comparison_favors_the_real_model(self):
        r = paired(self.oof, "B0 coin", "M5 good", self.names, n=500)
        self.assertEqual(r["games"], 4 * 40)
        self.assertLess(r["mean_diff"], 0)
        self.assertTrue(r["keep"])

    def test_reliability_plot_is_written(self):
        path = reliability_plot(self.oof, self.names["M5 good"], Path(self.dir.name) / "plots" / "r.png")
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1000)


class ModelCardTests(unittest.TestCase):
    def test_card_states_data_model_results_and_limits(self):
        cfg = Config("card", "logistic", "ratings_plus", grid={"C": [0.1, 1.0]}, calibration="platt")
        cv = [{"mode": "game", "protocol": "loso", "n": 160, "log_loss": 0.55, "brier": 0.19, "accuracy": 0.7, "ece": 0.05}]
        pred = fit_final(cfg, DF, stamp={"hash": "abc123"}, cv_metrics=cv)
        text = model_card(pred.manifest, "plots/reliability.png")
        for needle in ("Model card: card", "2022 to 2025", "abc123", "ratings_plus", "platt", "0.5500", "Known limits", "plots/reliability.png", "scikit-learn"):
            self.assertIn(needle, text)
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(write_model_card(d, pred.manifest).exists())

    def test_card_without_cross_validation_says_so(self):
        pred = fit_final(Config("c", "logistic", "rating_core"), DF)
        self.assertIn("No cross-validation rows", model_card(pred.manifest))


class TeamNameTests(unittest.TestCase):
    TABLE = pd.DataFrame(
        {
            "team": ["x/duke", "x/connecticut", "x/michigan-state", "x/saint-johns-ny", "x/miami-fl", "x/miami-oh", "x/texas-am", "x/north-carolina-state"],
            "display_name": ["Duke", "Connecticut", "Michigan State", "Saint John's (NY)", "Miami (FL)", "Miami (OH)", "Texas A&M", "North Carolina State"],
        }
    )

    def test_normalizer(self):
        self.assertEqual(normalize_name("Michigan St."), "michigan state")
        self.assertEqual(normalize_name("St. John's"), "saint johns")
        self.assertEqual(normalize_name("Texas A&M"), "texas am")
        self.assertEqual(normalize_name("The Duke University"), "duke")

    def test_matches_aliases_and_reports_the_rest(self):
        names = ["Duke", "UConn", "Michigan St.", "St. John's", "Miami (FL)", "Miami Ohio", "Texas A&M", "NC State", "Gonzaga"]
        mapping, unmatched = match_names(names, self.TABLE)
        self.assertEqual(mapping["UConn"], "x/connecticut")
        self.assertEqual(mapping["Michigan St."], "x/michigan-state")
        self.assertEqual(mapping["Miami Ohio"], "x/miami-oh")
        self.assertEqual(mapping["NC State"], "x/north-carolina-state")
        self.assertEqual(mapping["Texas A&M"], "x/texas-am")
        self.assertEqual(mapping["St. John's"], "x/saint-johns-ny")
        self.assertEqual(unmatched, ["Gonzaga"])

    def test_real_id_format_and_mascot_display_names(self):
        sr = "www.sports-reference.com/cbb/schools/{}/men"
        table = pd.DataFrame(
            {
                "team": [sr.format(s) for s in ("duke", "michigan-state", "loyola-il", "texas-am")],
                "display_name": ["Duke Blue Devils", "Michigan State Spartans", "Loyola (IL) Ramblers", "Texas A&M Aggies"],
            }
        )
        mapping, unmatched = match_names(["Duke", "Michigan St.", "Loyola Chicago", "Texas A&M", "Gonzaga"], table)
        self.assertEqual(mapping["Duke"], sr.format("duke"))
        self.assertEqual(mapping["Loyola Chicago"], sr.format("loyola-il"))
        self.assertEqual(mapping["Texas A&M"], sr.format("texas-am"))
        self.assertEqual(unmatched, ["Gonzaga"])

    def test_an_ambiguous_name_is_not_guessed(self):
        table = pd.concat([self.TABLE, pd.DataFrame({"team": ["x/duke2"], "display_name": ["Duke"]})], ignore_index=True)
        mapping, unmatched = match_names(["Duke"], table)
        self.assertEqual(unmatched, ["Duke"])
        self.assertNotIn("Duke", mapping)

    def test_unmatched_rows_are_dropped_and_listed(self):
        lines = pd.DataFrame({"team_a": ["Duke", "Gonzaga"], "team_b": ["UConn", "Duke"], "spread_a": [3.0, 5.0]})
        out, unmatched = map_team_columns(lines, self.TABLE)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["team_b"], "x/connecticut")
        self.assertEqual(unmatched, ["Gonzaga"])
        self.assertIn("uconn", ALIASES)


if __name__ == "__main__":
    unittest.main()
