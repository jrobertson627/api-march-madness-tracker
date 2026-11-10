import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.experiment import Config, fit_predict_split, run_experiment, run_locked, save_results
from ml.metrics import log_loss
from ml.splits import loso
from ml.synthetic import make_synthetic

DF = make_synthetic(seasons=(2022, 2023, 2024, 2025, 2026), n_regular=300, n_conf=50, n_ncaa=40)
CFG = Config("t", "logistic", "ratings_plus", grid={"C": [0.1, 1.0], "mix": ["equal", "tourney_x5"]}, calibration="platt")


class RunTests(unittest.TestCase):
    def test_one_oof_row_per_ncaa_game_per_season(self):
        res = run_experiment(CFG, DF)
        ncaa = DF[DF["game_group"] == "ncaa"]
        self.assertEqual(len(res.oof), len(ncaa))
        self.assertEqual(set(res.oof["game_id"]), set(ncaa["game_id"]))
        m = res.metrics
        self.assertEqual(set(m["season"].astype(str)), {"2022", "2023", "2024", "2025", "2026", "all"})
        self.assertTrue((m["log_loss"] < np.log(2)).all())

    def test_deterministic(self):
        a = run_experiment(CFG, DF).oof["p"].to_numpy()
        b = run_experiment(CFG, DF).oof["p"].to_numpy()
        self.assertTrue(np.array_equal(a, b))

    def test_excluded_season_is_absent(self):
        cfg = Config("t", "logistic", "rating_core", exclude_seasons=(2026,))
        res = run_experiment(cfg, DF)
        self.assertNotIn(2026, set(res.oof["season"]))

    def test_predictions_do_not_depend_on_test_labels(self):
        """Poison every label of the test season: the predictions for it cannot change."""
        sp = next(s for s in loso(DF) if s.season == 2024)
        test = DF.iloc[sp.test]
        base = fit_predict_split(CFG, DF, sp.train, {"game": test})["game"]
        poisoned = DF.copy()
        rows = poisoned["season"] == 2024
        poisoned.loc[rows, "a_win"] = 1 - poisoned.loc[rows, "a_win"]
        poisoned.loc[rows, "margin_a"] = -poisoned.loc[rows, "margin_a"]
        again = fit_predict_split(CFG, poisoned, sp.train, {"game": poisoned.iloc[sp.test]})["game"]
        self.assertTrue(np.array_equal(base["p"].to_numpy(), again["p"].to_numpy()))

    def test_bracket_mode_uses_the_same_fitted_model_on_other_features(self):
        bracket = DF[DF["game_group"] == "ncaa"].copy()
        bracket["diff_adj_em"] = bracket["diff_adj_em"] * 0.5
        res = run_experiment(CFG, DF, bracket=bracket)
        g = res.oof[res.oof["mode"] == "game"].set_index("game_id")
        b = res.oof[res.oof["mode"] == "bracket"].set_index("game_id").loc[g.index]
        self.assertEqual(len(g), len(b))
        self.assertFalse(np.allclose(g["p"], b["p"]))

    def test_score_model_adds_score_metrics(self):
        cfg = Config("s", "logistic", "rating_core", score_model="ridge")
        res = run_experiment(cfg, DF)
        row = res.metrics[res.metrics["season"].astype(str) == "all"].iloc[0]
        for c in ("margin_mae", "total_mae_reg", "total_mae_raw", "score_mae_raw", "margin_prob_log_loss", "margin_agrees_share"):
            self.assertTrue(np.isfinite(row[c]), c)

    def test_calibrated_predictions_are_symmetric_probabilities(self):
        res = run_experiment(CFG, DF)
        self.assertTrue(((res.oof["p"] > 0) & (res.oof["p"] < 1)).all())

    def test_run_id_changes_with_config(self):
        self.assertNotEqual(CFG.run_id(), Config("t", "logistic", "full").run_id())
        self.assertEqual(CFG.run_id(), Config("t", "logistic", "ratings_plus", grid={"C": [0.1, 1.0], "mix": ["equal", "tourney_x5"]}, calibration="platt").run_id())


class LockedBoxTests(unittest.TestCase):
    def test_second_locked_run_is_refused(self):
        cfg = Config("l", "logistic", "rating_core")
        with tempfile.TemporaryDirectory() as d:
            res = run_locked(cfg, DF, 2026, state_dir=Path(d))
            self.assertEqual(set(res.oof["season"]), {2026})
            with self.assertRaises(RuntimeError):
                run_locked(cfg, DF, 2026, state_dir=Path(d))
            run_locked(cfg, DF, 2026, state_dir=Path(d), force=True)

    def test_locked_run_does_not_train_on_the_locked_season(self):
        from ml.splits import locked_split

        sp = locked_split(DF, 2026)
        self.assertNotIn(2026, set(DF["season"].iloc[sp.train]))


class SaveTests(unittest.TestCase):
    def test_saving_twice_replaces_rows_but_logs_both_runs(self):
        cfg = Config("p", "logistic", "rating_core")
        res = run_experiment(cfg, DF)
        with tempfile.TemporaryDirectory() as d:
            save_results(res, Path(d))
            n = len(pd.read_csv(Path(d) / "experiments.csv"))
            save_results(res, Path(d))
            self.assertEqual(len(pd.read_csv(Path(d) / "experiments.csv")), n)
            self.assertEqual(len(pd.read_csv(Path(d) / "runs.csv")), 2)


if __name__ == "__main__":
    unittest.main()
