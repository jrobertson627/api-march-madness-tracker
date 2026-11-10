import unittest

import numpy as np
import pandas as pd

from bracket import pick, upsets
from bracket.backtest import POOL, build_tournament, choose_tau, event_scores, flag_outcomes, flag_summary, levels_for, strategy_scores
from bracket.simulate import exact_levels, simulate
from bracket.structure import Bracket
from tests.test_bracket_core import REGIONS, ReplayTests, full_field, strength_matrix


def play(bracket, P, rng):
    """Play one tournament from probabilities P; return the results table."""
    rows, level = [], []
    for cand in bracket.slots:
        if len(cand) == 1:
            level.append(cand[0])
        else:
            a, b = cand
            w = a if rng.random() < P[a, b] else b
            rows.append((bracket.teams[a], bracket.teams[b], bracket.teams[w], 0))
            level.append(w)
    for rnd in range(1, bracket.n_rounds + 1):
        nxt = []
        for j in range(len(level) // 2):
            a, b = level[2 * j], level[2 * j + 1]
            w = a if rng.random() < P[a, b] else b
            rows.append((bracket.teams[a], bracket.teams[b], bracket.teams[w], rnd))
            nxt.append(w)
        level = nxt
    return pd.DataFrame(rows, columns=["team_a", "team_b", "winner", "round"])


def fake_tournament(seed, P=None):
    field = full_field()
    b = Bracket.from_teams(field)
    P = strength_matrix(len(b.teams), 100 + seed, scale=1.3) if P is None else P
    res = play(b, P, np.random.default_rng(seed))
    return build_tournament(2022 + seed, field, res), P


class EventTests(unittest.TestCase):
    def test_true_probabilities_beat_a_coin_flip_on_average(self):
        true_loss, coin_loss = [], []
        for seed in range(12):
            t, P = fake_tournament(seed)
            true_loss.append(event_scores(exact_levels(t.bracket, P).reach(), t.realized)["log_loss"])
            coin_loss.append(event_scores(exact_levels(t.bracket, np.full_like(P, 0.5)).reach(), t.realized)["log_loss"])
        self.assertLess(np.mean(true_loss), np.mean(coin_loss))

    def test_events_cover_rounds_two_onward(self):
        t, P = fake_tournament(0)
        s = event_scores(exact_levels(t.bracket, P).reach(), t.realized)
        self.assertEqual(s["events"], 68 * 6)

    def test_a_tournament_that_does_not_match_the_structure_is_rejected(self):
        field = full_field()
        b = Bracket.from_teams(field)
        res = play(b, strength_matrix(68, 0), np.random.default_rng(0))
        res.loc[5, "team_b"] = "W9"
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_tournament(2022, field, res)


class TauTests(unittest.TestCase):
    def test_tau_zero_is_kept_unless_noise_wins_in_most_seasons(self):
        loss = pd.DataFrame({0.0: [0.30, 0.31, 0.32, 0.33], 0.1: [0.31, 0.30, 0.33, 0.34], 0.2: [0.32, 0.33, 0.34, 0.35]}, index=list(POOL))
        r = choose_tau(loss)
        self.assertEqual(r["tau"], 0.0)

    def test_a_clearly_better_tau_is_chosen_and_checked_nested(self):
        loss = pd.DataFrame({0.0: [0.40, 0.41, 0.42, 0.43], 0.1: [0.39, 0.40, 0.41, 0.42], 0.2: [0.38, 0.39, 0.40, 0.41], 0.3: [0.39, 0.40, 0.41, 0.42]}, index=list(POOL))
        r = choose_tau(loss)
        self.assertEqual(r["tau"], 0.2)
        self.assertEqual(r["seasons_better"], 4)
        self.assertAlmostEqual(r["nested_mean_loss"], loss[0.2].mean())
        self.assertLess(r["mean_loss_chosen"], r["mean_loss_tau0"])

    def test_a_tiny_gain_that_fails_the_nested_check_keeps_tau_at_zero(self):
        # The real 2022 to 2025 backtest: tau 0.3 has the lowest mean and wins in 3 of 4 seasons, by about 0.0002,
        # but choosing tau on the other seasons and scoring the held-out one is worse than tau 0.
        loss = pd.DataFrame(
            {0.0: [0.2862, 0.2819, 0.2326, 0.1648], 0.1: [0.2858, 0.2819, 0.2323, 0.1650],
             0.2: [0.2853, 0.2812, 0.2323, 0.1661], 0.3: [0.2846, 0.2801, 0.2320, 0.1678]}, index=list(POOL))
        r = choose_tau(loss)
        self.assertEqual((r["best_by_mean"], r["seasons_better"]), (0.3, 3))
        self.assertGreater(r["nested_mean_loss"], r["mean_loss_tau0"])
        self.assertEqual(r["tau"], 0.0)

    def test_levels_for_uses_exact_odds_at_tau_zero(self):
        t, P = fake_tournament(1)
        a = levels_for(t.bracket, P, 0.0, 1000, 0)
        self.assertIsNone(a.n_sims)
        b = levels_for(t.bracket, P, 0.2, 2000, 0)
        self.assertEqual(b.n_sims, 2000)


class StrategyAndFlagTests(unittest.TestCase):
    def test_strategy_table_has_three_rows_and_best_has_most_expected_points(self):
        t, P = fake_tournament(2)
        lv = exact_levels(t.bracket, P)
        table = strategy_scores(t, P, lv)
        self.assertEqual(len(table), 3)
        best = table[table.strategy == "best expected points"]["expected_points"].iloc[0]
        self.assertGreaterEqual(best + 1e-9, table["expected_points"].max())
        self.assertTrue((table["realized_points"] >= 0).all())

    def test_seed_history_and_flags(self):
        games = pd.DataFrame({"round": [1] * 6, "seed_a": [1, 1, 8, 9, 5, 12], "seed_b": [16, 16, 9, 8, 12, 5], "a_win": [1, 1, 0, 0, 0, 1]})
        h = upsets.seed_history(games, smoothing=1.0)
        self.assertAlmostEqual(h[(1, 16)], (0 + 1) / (2 + 2))   # no 16-seed won
        self.assertAlmostEqual(h[(8, 9)], (1 + 1) / (2 + 2))    # one of two went to the 9 seed
        self.assertAlmostEqual(h[(5, 12)], (2 + 1) / (2 + 2))   # both 12-seeds won
        self.assertAlmostEqual(h[(4, 13)], 0.5)                 # no games: smoothed to 0.5

    def test_first_round_flags_and_realized_upsets(self):
        t, P = fake_tournament(3)
        lv = exact_levels(t.bracket, P)
        hist = {pair: 0.2 for pair in upsets.SEED_PAIRS}
        flags = upsets.first_round_table(t.bracket, lv, hist)
        self.assertEqual(len(flags), 32)
        self.assertTrue(((flags.p_underdog_model >= 0) & (flags.p_underdog_model <= 1)).all())
        self.assertTrue((flags.model_picks_upset == (flags.p_underdog_model >= 0.5)).all())
        out = flag_outcomes(t, flags)
        self.assertEqual(int(out.upset.sum()), int(upsets.realized_upsets(t.bracket, t.actual).sum()))
        s = flag_summary(out)
        self.assertEqual(int(s[s.group == "all games"].games.iloc[0]), 32)
        self.assertEqual(int(s[s.group == "either flag"].games.iloc[0] + s[s.group == "not flagged"].games.iloc[0]), 32)

    def test_a_simulation_scores_close_to_the_exact_odds(self):
        t, P = fake_tournament(4)
        ex = event_scores(exact_levels(t.bracket, P).reach(), t.realized)["log_loss"]
        sm = event_scores(simulate(t.bracket, P, n_sims=100_000, seed=0).reach(), t.realized)["log_loss"]
        self.assertAlmostEqual(ex, sm, delta=0.01)


if __name__ == "__main__":
    unittest.main()
