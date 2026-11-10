import unittest

import numpy as np

from ml.calibration import IsotonicCalibrator, PlattCalibrator, logit, sigmoid
from ml.metrics import (
    accuracy,
    brier,
    ece,
    game_log_loss,
    keep_complex,
    log_loss,
    mae,
    paired_bootstrap,
    reliability_table,
    rmse,
    season_wins,
)


class MetricTests(unittest.TestCase):
    def test_coin_flip_log_loss_is_ln2(self):
        y = np.array([0, 1, 1, 0, 1])
        self.assertAlmostEqual(log_loss(np.full(5, 0.5), y), np.log(2))

    def test_hand_computed_cases(self):
        self.assertAlmostEqual(log_loss([0.9], [1]), -np.log(0.9))
        self.assertAlmostEqual(brier([0.8, 0.2], [1, 1]), (0.04 + 0.64) / 2)
        self.assertEqual(accuracy([0.6, 0.4, 0.5], [1, 1, 1]), (1 + 0 + 0.5) / 3)
        self.assertEqual(mae([1, 2, 3], [2, 2, 5]), 1.0)
        self.assertAlmostEqual(rmse([0, 0], [3, 4]), np.sqrt(12.5))

    def test_extreme_probabilities_do_not_blow_up(self):
        self.assertTrue(np.isfinite(game_log_loss([0.0, 1.0], [1, 0])).all())

    def test_calibrated_simulation_has_small_ece(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.random(20_000) < p).astype(int)
        self.assertLess(ece(p, y), 0.02)
        overconfident = sigmoid(3 * logit(p))
        self.assertGreater(ece(overconfident, y), 0.08)

    def test_reliability_table_covers_every_game(self):
        rng = np.random.default_rng(1)
        p = rng.random(333)
        t = reliability_table(p, (rng.random(333) < p).astype(int), bins=10)
        self.assertEqual(t["games"].sum(), 333)
        self.assertEqual(len(t), 10)


class BootstrapTests(unittest.TestCase):
    def test_same_seed_repeats_exactly(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(300), rng.random(300)
        self.assertEqual(paired_bootstrap(a, b, n=500, seed=3), paired_bootstrap(a, b, n=500, seed=3))

    def test_clearly_better_candidate_has_interval_below_zero(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.4, 0.9, 334)
        r = paired_bootstrap(base - 0.1 + rng.normal(0, 0.02, 334), base, n=2000)
        self.assertLess(r["hi"], 0)
        self.assertGreater(r["share_better"], 0.99)

    def test_identical_models_are_inconclusive(self):
        base = np.random.default_rng(1).uniform(0.4, 0.9, 334)
        r = paired_bootstrap(base, base, n=500)
        self.assertEqual(r["mean_diff"], 0.0)
        self.assertEqual((r["lo"], r["hi"]), (0.0, 0.0))

    def test_decision_rule(self):
        rng = np.random.default_rng(2)
        seasons = np.repeat([2022, 2023, 2024, 2025, 2026], 67)
        base = rng.uniform(0.4, 0.9, 335)
        better = keep_complex(base - 0.08, base, seasons, n=1000)
        self.assertTrue(better["keep"])
        self.assertEqual(better["season_wins"], 5)
        same = keep_complex(base + rng.normal(0, 0.3, 335) * 0, base, seasons, n=1000)
        self.assertFalse(same["keep"])
        self.assertEqual(season_wins(base, base, seasons), 0)

    def test_four_season_wins_keep_even_when_the_interval_spans_zero(self):
        seasons = np.repeat([2022, 2023, 2024, 2025, 2026], 67)
        base = np.full(335, 0.6)
        cand = base.copy()
        cand[seasons == 2026] += 0.4  # one bad season drags the pooled interval across zero
        cand[seasons != 2026] -= 0.02
        r = keep_complex(cand, base, seasons, n=2000)
        self.assertEqual(r["season_wins"], 4)
        self.assertTrue(r["keep"])


class CalibrationTests(unittest.TestCase):
    def test_platt_recovers_overconfidence(self):
        rng = np.random.default_rng(0)
        true_p = rng.uniform(0.1, 0.9, 20_000)
        y = (rng.random(20_000) < true_p).astype(int)
        model_p = sigmoid(2 * logit(true_p))  # twice too confident
        cal = PlattCalibrator().fit(model_p, y)
        self.assertAlmostEqual(cal.slope, 0.5, delta=0.05)
        self.assertEqual(cal.intercept, 0.0)

    def test_platt_is_symmetric(self):
        cal = PlattCalibrator().fit(np.array([0.2, 0.7, 0.9, 0.4, 0.6, 0.1]), np.array([0, 1, 1, 0, 1, 0]))
        p = np.linspace(0.05, 0.95, 19)
        self.assertTrue(np.allclose(cal.transform(p) + cal.transform(1 - p), 1.0))

    def test_isotonic_is_monotone_and_symmetric(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 3000)
        y = (rng.random(3000) < p**1.5 / (p**1.5 + (1 - p) ** 1.5)).astype(int)
        cal = IsotonicCalibrator().fit(p, y)
        grid = np.linspace(0.05, 0.95, 50)
        out = cal.transform(grid)
        self.assertTrue((np.diff(out) >= -1e-12).all())
        self.assertTrue(np.allclose(out + cal.transform(1 - grid), 1.0))


if __name__ == "__main__":
    unittest.main()
