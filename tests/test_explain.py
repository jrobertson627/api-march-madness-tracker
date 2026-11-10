import unittest

import numpy as np

from data.features import DIFF_FEATURES
from ml.dataset import feature_columns, swap_features
from ml.experiment import Config
from ml.explain import GROUPS, HEADLINE, check_complete, factor_effects, group_columns
from ml.predictor import fit_final
from ml.synthetic import make_synthetic

DF = make_synthetic(seasons=(2022, 2023, 2024), n_regular=300, n_conf=50, n_ncaa=40)
LOGIT = fit_final(Config("l", "logistic", "ratings_plus"), DF)
HGB = fit_final(Config("h", "hgb", "full", params={"max_iter": 40}), DF)
ROWS = DF[DF["game_group"] == "ncaa"].reset_index(drop=True)


class GroupTests(unittest.TestCase):
    def test_every_model_input_belongs_to_a_group(self):
        real_inputs = [f"diff_{c}" for c in DIFF_FEATURES] + ["proj_margin", "loc_a"]
        self.assertEqual(check_complete(real_inputs), [])

    def test_unmapped_inputs_are_reported(self):
        self.assertEqual(check_complete(["diff_adj_em", "diff_mystery"]), ["diff_mystery"])

    def test_groups_only_list_columns_the_model_has(self):
        g = group_columns(["diff_adj_em", "diff_elo", "loc_a", "proj_total"])
        self.assertEqual(g, {"efficiency": ["diff_adj_em"], "elo": ["diff_elo"]})
        self.assertTrue(set(HEADLINE) >= set(GROUPS))


class EffectTests(unittest.TestCase):
    def test_effects_plus_interaction_equal_the_move_from_a_coin_flip(self):
        for model in (LOGIT, HGB):
            fx = factor_effects(model, ROWS)
            groups = [c for c in fx.columns if c not in ("p_a_win", "interaction")]
            self.assertTrue(groups)
            self.assertTrue(np.allclose(fx[groups].sum(axis=1) + fx["interaction"], fx["p_a_win"] - 0.5, atol=1e-9))

    def test_zeroing_every_signed_input_gives_exactly_half(self):
        for model in (LOGIT, HGB):
            rows = ROWS.copy()
            cols = [c for c in model.fitted.cols if c.startswith("diff_") or c in ("proj_margin", "loc_a")]
            rows[cols] = 0.0
            p = model.fitted.model.predict_proba(rows[model.fitted.cols])
            self.assertTrue(np.allclose(p, 0.5, atol=1e-9))

    def test_effects_flip_when_the_teams_swap(self):
        fx = factor_effects(LOGIT, ROWS)
        swapped = swap_features(ROWS[LOGIT.fitted.cols]).assign(game_group=ROWS["game_group"])
        fs = factor_effects(LOGIT, swapped)
        groups = [c for c in fx.columns if c != "p_a_win"]
        self.assertTrue(np.allclose(fx[groups].to_numpy(), -fs[groups].to_numpy(), atol=1e-9))

    def test_a_better_team_on_efficiency_gets_a_positive_efficiency_effect(self):
        fx = factor_effects(LOGIT, ROWS)
        better = ROWS["diff_adj_em"] > 5
        self.assertGreater(len(ROWS[better]), 5)
        self.assertTrue((fx.loc[better, "efficiency"] > 0).all())


if __name__ == "__main__":
    unittest.main()
