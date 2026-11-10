import unittest

import numpy as np
import pandas as pd

from ml.dataset import MIXES, augment, feature_columns, ot_factor, swap_features, training_weights
from ml.splits import forward_chain, inner, locked_split, loso
from ml.synthetic import make_synthetic


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic(seasons=(2022, 2023, 2024, 2025, 2026))

    def seasons(self, idx):
        return set(self.df["season"].iloc[idx])

    def test_loso_is_pure(self):
        splits = list(loso(self.df))
        self.assertEqual([s.season for s in splits], [2022, 2023, 2024, 2025, 2026])
        for s in splits:
            self.assertEqual(len(np.intersect1d(s.train, s.test)), 0)
            self.assertEqual(self.seasons(s.test), {s.season})
            self.assertTrue((self.df["game_group"].iloc[s.test] == "ncaa").all())
            self.assertNotIn(s.season, self.seasons(s.train))  # no row of the held-out season, any type
            self.assertEqual(self.seasons(s.train), {2022, 2023, 2024, 2025, 2026} - {s.season})

    def test_excluded_season_is_never_used(self):
        for s in loso(self.df, exclude_seasons=[2026]):
            self.assertNotIn(2026, self.seasons(s.train) | self.seasons(s.test))
            self.assertNotEqual(s.season, 2026)

    def test_forward_chain_only_trains_on_the_past(self):
        splits = list(forward_chain(self.df, min_train_seasons=2))
        self.assertEqual([s.season for s in splits], [2024, 2025, 2026])
        for s in splits:
            self.assertTrue(all(t < s.season for t in self.seasons(s.train)))

    def test_inner_never_touches_the_outer_test_season(self):
        outer = next(s for s in loso(self.df) if s.season == 2024)
        inners = list(inner(self.df, outer.train))
        self.assertEqual(len(inners), 4)
        for s in inners:
            used = self.seasons(s.train) | self.seasons(s.test)
            self.assertNotIn(2024, used)
            self.assertNotIn(s.season, self.seasons(s.train))
            self.assertTrue(self.df["game_group"].iloc[s.test].isin(["ncaa", "conf_tourney"]).all())

    def test_locked_split_trains_on_every_other_season(self):
        s = locked_split(self.df, 2026)
        self.assertEqual(self.seasons(s.train), {2022, 2023, 2024, 2025})
        self.assertEqual(self.seasons(s.test), {2026})


class SymmetryHelpersTests(unittest.TestCase):
    def test_swap_flips_signed_columns_and_is_an_involution(self):
        df = make_synthetic(seasons=(2022,))
        cols = ["diff_adj_em", "loc_a", "proj_margin", "proj_total", "elo_expected_a", "is_ncaa"]
        X = df[cols]
        S = swap_features(X)
        pd.testing.assert_series_equal(S["diff_adj_em"], -X["diff_adj_em"])
        pd.testing.assert_series_equal(S["elo_expected_a"], 1 - X["elo_expected_a"])
        pd.testing.assert_series_equal(S["proj_total"], X["proj_total"])  # invariant
        pd.testing.assert_series_equal(S["is_ncaa"], X["is_ncaa"])
        pd.testing.assert_frame_equal(swap_features(S), X)

    def test_augment_doubles_rows_and_flips_labels(self):
        df = make_synthetic(seasons=(2022,))
        X, y, w = augment(df[["diff_adj_em"]], df["a_win"], np.ones(len(df)) * 2)
        self.assertEqual(len(X), 2 * len(df))
        n = len(df)
        self.assertTrue(np.allclose(y[n:], 1 - y[:n]))
        self.assertTrue(np.allclose(X["diff_adj_em"].to_numpy()[n:], -X["diff_adj_em"].to_numpy()[:n]))
        self.assertTrue((w == 2).all())

    def test_overtime_factor(self):
        self.assertTrue(np.allclose(ot_factor([0, 1, 2]), [1.0, 40 / 45, 40 / 50]))

    def test_weights_by_game_group(self):
        df = make_synthetic(seasons=(2022,))
        w = training_weights(df, "tourney_x5")
        self.assertTrue((w[df["game_group"] == "regular"] == 1).all())
        self.assertTrue((w[df["game_group"] == "ncaa"] == 5).all())
        self.assertTrue((training_weights(df, "ncaa_only")[df["game_group"] != "ncaa"] == 0).all())
        with self.assertRaises(KeyError):
            training_weights(df, "nope")
        self.assertIn("equal", MIXES)

    def test_feature_set_errors(self):
        df = make_synthetic(seasons=(2022,))
        self.assertIn("diff_adj_em", feature_columns("rating_core", df.columns))
        self.assertNotIn("diff_seed", feature_columns("full", df.columns))
        self.assertIn("diff_seed", feature_columns("full_with_seed", df.columns))
        with self.assertRaises(KeyError):
            feature_columns("nope", df.columns)
        with self.assertRaises(KeyError):
            feature_columns("rating_core", ["diff_elo"])


if __name__ == "__main__":
    unittest.main()
