import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.benchmarks import compare_to_model, estimate_sigma, load_benchmark, score_benchmark, spread_to_prob
from ml.dataset import swap_features
from ml.experiment import Config
from ml.predictor import JsonPredictor, Predictor, fit_final
from ml.synthetic import make_synthetic

DF = make_synthetic(seasons=(2022, 2023, 2024, 2025), n_regular=300, n_conf=50, n_ncaa=40)
CFG = Config("p", "logistic", "ratings_plus", grid={"C": [0.1, 1.0]}, calibration="platt", score_model="ridge")


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = fit_final(CFG, DF, stamp={"hash": "abc"})
        cls.ncaa = DF[DF["game_group"] == "ncaa"]

    def test_predicts_probability_and_scores(self):
        out = self.pred.predict(self.ncaa)
        self.assertEqual(len(out), len(self.ncaa))
        self.assertTrue(((out["p_a_win"] > 0) & (out["p_a_win"] < 1)).all())
        self.assertTrue(np.allclose(out["score_a_hat"] + out["score_b_hat"], out["total_reg_hat"]))
        self.assertTrue(np.allclose(out["score_a_hat"] - out["score_b_hat"], out["margin_hat"]))

    def test_swapping_teams_flips_the_prediction(self):
        cols = self.pred.required_columns
        a = self.pred.predict(self.ncaa)
        swapped = swap_features(self.ncaa[cols]).assign(game_group=self.ncaa["game_group"])
        b = self.pred.predict(swapped)
        self.assertTrue(np.allclose(a["p_a_win"] + b["p_a_win"], 1.0))
        self.assertTrue(np.allclose(a["margin_hat"], -b["margin_hat"]))

    def test_missing_columns_raise_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.pred.predict(self.ncaa.drop(columns=["diff_adj_em"]))

    def test_manifest_records_what_the_model_was_trained_on(self):
        m = self.pred.manifest
        self.assertEqual(m["training_seasons"], [2022, 2023, 2024, 2025])
        self.assertEqual(m["data_stamp"], {"hash": "abc"})
        for key in ("win_features", "packages", "chosen_hyperparameters", "calibration", "sigma", "created", "limits"):
            self.assertIn(key, m)
        self.assertIn("scikit-learn", m["packages"])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            self.pred.save(d)
            self.assertTrue((Path(d) / "manifest.json").exists())
            again = Predictor.load(d)
            pd.testing.assert_frame_equal(self.pred.predict(self.ncaa), again.predict(self.ncaa))
            self.assertEqual(again.manifest["run_id"], self.pred.manifest["run_id"])

    def test_json_export_matches_the_full_predictor(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.pred.export_json(Path(d) / "model.json")
            doc = json.loads(path.read_text())
            self.assertIn("win", doc)
            lean = JsonPredictor.load(path).predict(self.ncaa)
        full = self.pred.predict(self.ncaa)
        for c in ("p_a_win", "p_raw", "margin_hat", "total_reg_hat", "score_a_hat", "score_b_hat", "p_from_margin"):
            self.assertTrue(np.allclose(lean[c], full[c], atol=1e-8), c)

    def test_json_export_refuses_models_it_cannot_express(self):
        hgb = fit_final(Config("h", "hgb", "ratings_plus", params={"max_iter": 20}), DF)
        with self.assertRaises(NotImplementedError):
            hgb.export_json("unused.json")


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.ncaa = DF[DF["game_group"] == "ncaa"].reset_index(drop=True)
        self.ncaa["game_date"] = pd.to_datetime(self.ncaa["game_date"])
        rng = np.random.default_rng(0)
        self.spread = self.ncaa["proj_margin"] + rng.normal(0, 1, len(self.ncaa))

    def lines(self, flip=False):
        df = pd.DataFrame({"season": self.ncaa["season"], "game_date": self.ncaa["game_date"],
                           "team_a": self.ncaa["team_a"], "team_b": self.ncaa["team_b"], "spread_a": self.spread})
        if flip:
            df = df.rename(columns={"team_a": "team_b", "team_b": "team_a"}).assign(spread_a=-df["spread_a"])
        return df

    def test_spread_to_probability(self):
        p = spread_to_prob([-10, 0, 10], sigma=11.0)
        self.assertAlmostEqual(p[1], 0.5)
        self.assertTrue((np.diff(p) > 0).all())
        self.assertAlmostEqual(p[0] + p[2], 1.0)
        self.assertAlmostEqual(estimate_sigma([3, -2], [5, -2]), 1.0)

    def test_flipped_rows_are_put_in_canonical_order(self):
        same = load_benchmark(self.lines())
        flipped = load_benchmark(self.lines(flip=True))
        a = same.set_index(["season", "game_date", "team_a", "team_b"])["p_bench"]
        b = flipped.set_index(["season", "game_date", "team_a", "team_b"])["p_bench"]
        self.assertTrue(((flipped["team_a"] < flipped["team_b"]).all()))
        # ids in the fake file are not alphabetical, so compare per game where the order matches
        both = a.index.intersection(b.index)
        self.assertTrue(np.allclose(a.loc[both], b.loc[both]))

    def test_scoring_and_coverage(self):
        bench = load_benchmark(self.lines().iloc[:100])
        r = score_benchmark(bench, self.ncaa)
        self.assertEqual(r["games"], 100)
        self.assertAlmostEqual(r["coverage"], 100 / len(self.ncaa))
        self.assertLess(r["log_loss"], np.log(2))

    def test_unusable_file_is_rejected(self):
        with self.assertRaises(ValueError):
            load_benchmark(pd.DataFrame({"season": [2022], "game_date": [pd.Timestamp("2022-03-17")], "team_a": ["a"], "team_b": ["b"]}))

    def test_paired_comparison_against_a_model(self):
        bench = load_benchmark(self.lines())
        oof = self.ncaa.assign(p=0.5)  # a coin flip against the lines
        r = compare_to_model(bench, oof, n=500)
        self.assertEqual(r["games"], len(self.ncaa))
        self.assertGreater(r["model_log_loss"], r["bench_log_loss"])
        self.assertGreater(r["mean_diff"], 0)


if __name__ == "__main__":
    unittest.main()
