import unittest

import numpy as np
import pandas as pd

from ml.dataset import feature_columns, swap_features
from ml.metrics import log_loss
from ml.models import ScoreModel, make_win_model, normal_cdf
from ml.synthetic import make_synthetic

DF = make_synthetic(seasons=(2022, 2023, 2024, 2025), n_regular=500, n_conf=60, n_ncaa=40)
TRAIN = DF[DF["season"] != 2025]
TEST = DF[(DF["season"] == 2025) & (DF["game_group"] == "ncaa")]


def fit_predict(name, feature_set, train=TRAIN, test=TEST, **params):
    cols = feature_columns(feature_set, DF.columns)
    m = make_win_model(name, **params).fit(train[cols], train["a_win"])
    return m, m.predict_proba(test[cols]), cols


class WinModelTests(unittest.TestCase):
    def test_every_model_is_symmetric(self):
        cases = [
            ("coin", "rating_core", {}),
            ("elo_prob", "elo_prob", {}),
            ("logistic", "rating_core", {}),
            ("logistic", "full", {}),
            ("hgb", "ratings_plus", {"max_iter": 40}),
            ("hgb", "full", {"max_iter": 40, "monotone": True}),
        ]
        for name, fs, params in cases:
            with self.subTest(model=name, features=fs):
                m, p, cols = fit_predict(name, fs, **params)
                q = m.predict_proba(swap_features(TEST[cols]))
                self.assertTrue(np.allclose(p + q, 1.0, atol=1e-9), (name, fs))

    def test_seed_pair_is_symmetric_and_orders_by_seed(self):
        ncaa_train = TRAIN[TRAIN["game_group"] == "ncaa"]
        cols = ["seed_a", "seed_b"]
        m = make_win_model("seed_pair").fit(ncaa_train[cols], ncaa_train["a_win"])
        X = pd.DataFrame({"seed_a": [1.0, 16.0, 8.0], "seed_b": [16.0, 1.0, 8.0]})
        p = m.predict_proba(X)
        self.assertGreater(p[0], 0.5)
        self.assertAlmostEqual(p[0] + p[1], 1.0)
        self.assertEqual(p[2], 0.5)

    def test_logistic_learns_the_known_relationship(self):
        m, p, cols = fit_predict("logistic", "rating_core")
        self.assertGreater(m.params()["coef"]["diff_adj_em"], 0)
        self.assertLess(log_loss(p, TEST["a_win"]), np.log(2) - 0.05)

    def test_logistic_ignores_columns_that_do_not_flip_sign(self):
        cols = feature_columns("rating_core", DF.columns)
        base = make_win_model("logistic").fit(TRAIN[cols], TRAIN["a_win"])
        with_flag = make_win_model("logistic").fit(TRAIN[[*cols, "is_ncaa"]], TRAIN["a_win"])
        self.assertEqual(with_flag.cols_, base.cols_)
        self.assertTrue(np.allclose(base.predict_proba(TEST[cols]), with_flag.predict_proba(TEST[[*cols, "is_ncaa"]])))

    def test_missing_differences_count_as_no_information(self):
        cols = feature_columns("rating_core", DF.columns)
        m = make_win_model("logistic").fit(TRAIN[cols], TRAIN["a_win"])
        blank = pd.DataFrame({"diff_adj_em": [np.nan], "loc_a": [0.0]})
        self.assertAlmostEqual(float(m.predict_proba(blank)[0]), 0.5)

    def test_hgb_beats_a_coin_and_is_repeatable(self):
        _, p1, _ = fit_predict("hgb", "ratings_plus", max_iter=60)
        _, p2, _ = fit_predict("hgb", "ratings_plus", max_iter=60)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertLess(log_loss(p1, TEST["a_win"]), np.log(2))

    def test_hgb_copes_with_a_column_that_is_entirely_missing(self):
        cols = feature_columns("ratings_plus", DF.columns)
        train = TRAIN[cols].copy()
        train["diff_form_trend"] = np.nan  # e.g. roster features when only the first season is in training
        m = make_win_model("hgb", max_iter=30).fit(train, TRAIN["a_win"])
        self.assertNotIn("diff_form_trend", m.cols_)
        p = m.predict_proba(TEST[cols])  # the column is present at prediction time and ignored
        q = m.predict_proba(swap_features(TEST[cols]))
        self.assertTrue(np.allclose(p + q, 1.0))

    def test_zero_weight_rows_are_ignored(self):
        cols = feature_columns("rating_core", DF.columns)
        w = np.where(TRAIN["game_group"] == "ncaa", 1.0, 0.0)
        a = make_win_model("logistic").fit(TRAIN[cols], TRAIN["a_win"], w)
        ncaa = TRAIN[TRAIN["game_group"] == "ncaa"]
        b = make_win_model("logistic").fit(ncaa[cols], ncaa["a_win"])
        self.assertTrue(np.allclose(a.predict_proba(TEST[cols]), b.predict_proba(TEST[cols])))

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            make_win_model("nope")


class ScoreModelTests(unittest.TestCase):
    def fit(self, kind="ridge"):
        cols = feature_columns("full", DF.columns) + ["sum_adj_off", "sum_adj_def", "sum_adj_tempo", "sum_tempo"]
        m = ScoreModel(kind=kind).fit(
            TRAIN[cols], TRAIN["margin_a"], TRAIN["total_reg"], group=TRAIN["game_group"]
        )
        return m, cols

    def test_margin_flips_and_total_does_not_when_teams_swap(self):
        for kind in ("ridge", "hgb"):
            with self.subTest(kind=kind):
                m, cols = self.fit(kind)
                a = m.predict(TEST[cols])
                b = m.predict(swap_features(TEST[cols]))
                self.assertTrue(np.allclose(a["margin_hat"], -b["margin_hat"]))
                self.assertTrue(np.allclose(a["total_reg_hat"], b["total_reg_hat"]))
                self.assertTrue(np.allclose(a["p_from_margin"] + b["p_from_margin"], 1.0))

    def test_scores_agree_with_margin_and_total(self):
        m, cols = self.fit()
        p = m.predict(TEST[cols])
        self.assertTrue(np.allclose(p["score_a_hat"] + p["score_b_hat"], p["total_reg_hat"]))
        self.assertTrue(np.allclose(p["score_a_hat"] - p["score_b_hat"], p["margin_hat"]))

    def test_recovers_margin_signal_and_noise_level(self):
        m, cols = self.fit()
        p = m.predict(TEST[cols])
        self.assertGreater(np.corrcoef(p["margin_hat"], TEST["margin_a"])[0, 1], 0.4)
        self.assertAlmostEqual(m.sigma_["all"], 11.0, delta=2.5)  # the simulation's noise
        self.assertIn("ncaa", m.sigma_)

    def test_margin_implied_probability_agrees_with_the_margin_sign(self):
        m, cols = self.fit()
        p = m.predict(TEST[cols])
        self.assertTrue(((p["p_from_margin"] > 0.5) == (p["margin_hat"] > 0)).all())

    def test_total_is_trained_on_regulation_scoring(self):
        ot = DF[DF["overtimes"] > 0]
        self.assertGreater(len(ot), 0)
        self.assertTrue((ot["total_reg"] < ot["total"]).all())
        reg = DF[DF["overtimes"] == 0]
        self.assertTrue(np.allclose(reg["total_reg"], reg["total"]))

    def test_normal_cdf(self):
        z = np.array([-3.0, 0.0, 1.0, 3.0])
        out = normal_cdf(z)
        self.assertAlmostEqual(out[1], 0.5)
        self.assertTrue((np.diff(out) > 0).all())
        self.assertAlmostEqual(out[2], 0.8413447, places=6)


if __name__ == "__main__":
    unittest.main()
