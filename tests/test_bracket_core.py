import itertools
import unittest

import numpy as np
import pandas as pd

from bracket.pick import Picks, Scoring, best_expected_bracket, expected_points, favorite_bracket, score, seed_bracket
from bracket.simulate import Levels, check_matrix, exact_levels, fixed_matrix, reach_table, simulate
from bracket.structure import SLOT_SEEDS, Bracket

REGIONS = ("East", "West", "South", "Midwest")


def full_field(first_four=((0, 16), (2, 11))):
    """68 teams: four regions, seeds 1 to 16, play-ins at region 0 seed 16 and region 2 seed 11."""
    rows = []
    for ri, region in enumerate(REGIONS):
        for s in range(1, 17):
            rows.append((f"{region[0]}{s}", s, region))
    for ri, seed in first_four:
        rows.append((f"{REGIONS[ri][0]}{seed}b", seed, REGIONS[ri]))
    rows += [("Xa", 16, "West"), ("Ya", 11, "Midwest")]  # two more play-in teams, in other regions
    return pd.DataFrame(rows, columns=["team", "seed", "region"])


def strength_matrix(n, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, scale, n)
    d = s[:, None] - s[None, :]
    return 1 / (1 + np.exp(-d))


def small_bracket(n_slots=8, playin=()):
    """A generic bracket of n_slots slots; slots listed in `playin` get a second team."""
    teams = [f"t{i}" for i in range(n_slots + len(playin))]
    slots = [(i,) for i in range(n_slots)]
    for k, extra in enumerate(playin):
        slots[extra] = (extra, n_slots + k)
    return Bracket.from_slots(teams, slots)


class StructureTests(unittest.TestCase):
    def test_full_bracket_has_64_slots_and_six_rounds(self):
        t = full_field()
        b = Bracket.from_teams(t.iloc[:68], final_four=(("East", "West"), ("South", "Midwest")))
        self.assertEqual((b.n_slots, b.n_rounds), (64, 6))
        self.assertEqual(len(b.teams), 68)
        self.assertEqual(b.regions, ("East", "West", "South", "Midwest"))

    def test_first_four_are_found_from_shared_slots(self):
        b = Bracket.from_teams(full_field())
        playins = [tuple(b.teams[i] for i in b.slots[k]) for k in b.playin_slots]
        self.assertEqual(len(playins), 4)
        self.assertTrue(all(len(p) == 2 for p in playins))

    def test_region_slot_order_follows_the_standard_bracket(self):
        b = Bracket.from_teams(full_field(), final_four=(("East", "West"), ("South", "Midwest")))
        seeds_first_region = [b.seed[b.slots[k][0]] for k in range(16)]
        self.assertEqual(seeds_first_region, list(SLOT_SEEDS))
        self.assertEqual({b.region[t] for k in range(16) for t in b.slots[k]}, {"East"})

    def test_bad_inputs_raise_clear_errors(self):
        t = full_field()
        with self.assertRaisesRegex(ValueError, "four regions"):
            Bracket.from_teams(t[t.region != "East"])
        with self.assertRaisesRegex(ValueError, "do not have one or two teams"):
            Bracket.from_teams(t[t.team != "E7"])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            Bracket.from_teams(t, final_four=(("East", "East"), ("South", "Midwest")))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            Bracket.from_teams(t.drop(columns="seed"))
        with self.assertRaisesRegex(ValueError, "power of two"):
            Bracket.from_slots(["a", "b", "c"], [(0,), (1,), (2,)])

    def test_unseeded_first_four_team_takes_its_opponents_slot(self):
        t = full_field()
        t = t[t.team != "Xa"]  # Xa is a West 16 play-in team that has no seed row
        results = pd.DataFrame({"team_a": ["Xa"], "team_b": ["W16"], "round": [0], "winner": ["W16"]})
        b = Bracket.from_results(t, results)
        self.assertIn("Xa", b.teams)
        xa = b.index["Xa"]
        self.assertEqual((b.seed[xa], b.region[xa]), (16, "West"))
        self.assertEqual(len([k for k in b.playin_slots if xa in b.slots[k]]), 1)


class ReplayTests(unittest.TestCase):
    def play_all(self, b, rng):
        """Play a whole bracket with coin flips and return its results table."""
        rows, level = [], []
        for cand in b.slots:
            if len(cand) == 1:
                level.append(cand[0])
            else:
                w = cand[rng.integers(2)]
                rows.append((b.teams[cand[0]], b.teams[cand[1]], b.teams[w], 0))
                level.append(w)
        for rnd in range(1, b.n_rounds + 1):
            nxt = []
            for j in range(len(level) // 2):
                a, c = level[2 * j], level[2 * j + 1]
                w = a if rng.random() < 0.5 else c
                rows.append((b.teams[a], b.teams[c], b.teams[w], rnd))
                nxt.append(w)
            level = nxt
        return pd.DataFrame(rows, columns=["team_a", "team_b", "winner", "round"])

    def test_a_played_out_tournament_replays_with_no_problems(self):
        b = Bracket.from_teams(full_field())
        res = self.play_all(b, np.random.default_rng(1))
        self.assertEqual(len(res), 67)
        actual, problems = b.replay(res)
        self.assertEqual(problems, [])
        self.assertTrue((actual[6] >= 0).all())
        reach = b.realized_reach(actual)
        self.assertEqual(reach[:, -1].sum(), 1.0)  # exactly one champion
        self.assertEqual(reach[:, 1].sum(), 32)  # 32 teams in round 2

    def test_a_game_that_cannot_happen_is_reported(self):
        b = Bracket.from_teams(full_field())
        res = self.play_all(b, np.random.default_rng(2))
        bad = res.copy()
        bad.loc[bad.index[10], "team_b"] = "W9"  # a team that is not in that slot
        _, problems = b.replay(bad)
        self.assertTrue(problems)

    def test_a_missing_result_is_reported(self):
        b = Bracket.from_teams(full_field())
        res = self.play_all(b, np.random.default_rng(3))
        _, problems = b.replay(res.iloc[:-1])
        self.assertTrue(any("no result" in p for p in problems))


class ExactAndSimulatedTests(unittest.TestCase):
    def test_two_team_bracket(self):
        b = small_bracket(2)
        P = np.array([[0.5, 0.7], [0.3, 0.5]])
        lv = exact_levels(b, P)
        self.assertTrue(np.allclose(lv.reach()[:, -1], [0.7, 0.3]))

    def test_four_team_bracket_by_hand(self):
        b = small_bracket(4)
        P = strength_matrix(4, 0)
        lv = exact_levels(b, P)
        # team 0 wins the title: beat 1, then beat the winner of 2 v 3
        expected0 = P[0, 1] * (P[2, 3] * P[0, 2] + P[3, 2] * P[0, 3])
        self.assertAlmostEqual(lv.reach()[0, -1], expected0)

    def test_conservation_and_symmetry(self):
        b = Bracket.from_teams(full_field())
        P = strength_matrix(len(b.teams), 4)
        lv = exact_levels(b, P)
        reach = lv.reach()
        self.assertAlmostEqual(reach[:, -1].sum(), 1.0)
        for r, n in enumerate([64, 32, 16, 8, 4, 2]):
            self.assertAlmostEqual(reach[:, r].sum(), n)
        self.assertTrue(np.allclose(reach[:, 0][[i for k in b.playin_slots for i in b.slots[k]]].reshape(-1, 2).sum(axis=1), 1.0))

    def test_simulation_matches_exact_within_sampling_error(self):
        for n_slots, playin in ((2, ()), (4, ()), (8, (0,)), (8, ())):
            with self.subTest(slots=n_slots, playin=playin):
                b = small_bracket(n_slots, playin)
                P = strength_matrix(len(b.teams), 5, scale=1.2)
                ex, sim = exact_levels(b, P), simulate(b, P, n_sims=200_000, seed=1)
                err = np.abs(ex.reach() - sim.reach())
                bound = 4 * np.sqrt(0.25 / 200_000)
                self.assertLess(err.max(), bound)

    def test_same_seed_same_result(self):
        b = small_bracket(8)
        P = strength_matrix(8, 6)
        a = simulate(b, P, n_sims=5000, seed=7).reach()
        c = simulate(b, P, n_sims=5000, seed=7).reach()
        self.assertTrue(np.array_equal(a, c))

    def test_strength_noise_helps_underdogs_and_keeps_the_totals(self):
        b = small_bracket(8)
        P = strength_matrix(8, 8, scale=1.5)
        base = exact_levels(b, P).reach()[:, -1]
        noisy = simulate(b, P, n_sims=300_000, tau=0.8, seed=2).reach()[:, -1]
        self.assertAlmostEqual(noisy.sum(), 1.0)
        weakest = np.argmin(base)
        self.assertGreater(noisy[weakest], base[weakest])  # longshots win more once their strength is uncertain
        zero = simulate(b, P, n_sims=300_000, tau=0.0, seed=2).reach()[:, -1]
        self.assertLess(np.abs(zero - base).max(), 4 * np.sqrt(0.25 / 300_000))

    def test_fixed_results_are_respected(self):
        b = small_bracket(4)
        P = strength_matrix(4, 9)
        fixed = [("t0", "t1", "t1")]
        ex = exact_levels(b, P, fixed).reach()
        self.assertEqual(ex[0, 1], 0.0)  # t0 never reaches the final
        sim = simulate(b, P, n_sims=20_000, fixed=fixed, seed=3).reach()
        self.assertEqual(sim[0, 1], 0.0)
        self.assertAlmostEqual(sim[1, 1], 1.0)  # t1 is certainly in the final
        self.assertTrue(np.allclose(ex, sim, atol=0.02))

    def test_a_fixed_winner_who_was_not_in_the_game_is_rejected(self):
        with self.assertRaises(ValueError):
            fixed_matrix(small_bracket(4), [("t0", "t1", "t2")])

    def test_a_bad_matrix_is_rejected(self):
        b = small_bracket(4)
        P = strength_matrix(4, 0)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            check_matrix(P + 0.1 * np.triu(np.ones((4, 4)), 1), 4)
        with self.assertRaisesRegex(ValueError, "missing"):
            check_matrix(np.where(np.eye(4) == 1, 0.5, np.nan), 4)
        with self.assertRaisesRegex(ValueError, "expected"):
            exact_levels(b, np.zeros((3, 3)))

    def test_reach_table_columns(self):
        b = Bracket.from_teams(full_field())
        t = reach_table(b, exact_levels(b, strength_matrix(len(b.teams), 1)))
        self.assertEqual(len(t), 68)
        self.assertIn("p_champion", t.columns)
        self.assertAlmostEqual(t["p_champion"].sum(), 1.0)


class PickTests(unittest.TestCase):
    def brute_force_best(self, b, lv, scoring):
        """Try every consistent bracket on a small bracket and return the best expected points."""

        def rec(levels):
            prev = levels[-1]
            if len(prev) == 1:
                yield levels
                return
            n = len(prev) // 2
            for combo in itertools.product(*[(prev[2 * j], prev[2 * j + 1]) for j in range(n)]):
                yield from rec(levels + [np.array(combo)])

        level0 = np.array([c[0] for c in b.slots])
        return max(expected_points(Picks(levels), lv, scoring) for levels in rec([level0]))

    def test_dp_equals_brute_force(self):
        scoring = Scoring((1, 2, 4))
        for seed in range(4):
            b = small_bracket(8)
            P = strength_matrix(8, seed, scale=1.3)
            lv = exact_levels(b, P)
            picks = best_expected_bracket(b, lv, scoring)
            self.assertAlmostEqual(expected_points(picks, lv, scoring), self.brute_force_best(b, lv, scoring), places=6)

    def test_picks_are_consistent(self):
        b = Bracket.from_teams(full_field())
        P = strength_matrix(len(b.teams), 3, scale=1.2)
        lv = exact_levels(b, P)
        for picks in (best_expected_bracket(b, lv), favorite_bracket(b, P), seed_bracket(b)):
            for r in range(1, 7):
                prev = picks.winners[r - 1]
                for j, w in enumerate(picks.winners[r]):
                    self.assertIn(w, (prev[2 * j], prev[2 * j + 1]), (picks.name, r, j))

    def test_best_expected_beats_or_ties_the_other_strategies(self):
        b = Bracket.from_teams(full_field())
        P = strength_matrix(len(b.teams), 11, scale=1.5)
        lv = exact_levels(b, P)
        best = expected_points(best_expected_bracket(b, lv), lv)
        self.assertGreaterEqual(best + 1e-9, expected_points(favorite_bracket(b, P), lv))
        self.assertGreaterEqual(best + 1e-9, expected_points(seed_bracket(b), lv))

    def test_favorite_bracket_picks_the_higher_probability(self):
        b = small_bracket(4)
        P = np.array([[0.5, 0.2, 0.6, 0.6], [0.8, 0.5, 0.6, 0.6], [0.4, 0.4, 0.5, 0.9], [0.4, 0.4, 0.1, 0.5]])
        picks = favorite_bracket(b, P)
        self.assertEqual(picks.winners[1].tolist(), [1, 2])
        self.assertEqual(picks.winners[2].tolist(), [1])

    def test_score_counts_only_correct_games(self):
        b = small_bracket(4)
        picks = Picks([np.arange(4), np.array([0, 2]), np.array([0])])
        actual = [np.arange(4), np.array([0, 3]), np.array([0])]
        self.assertEqual(score(picks, actual, Scoring((10, 20))), 10 + 20)
        actual2 = [np.arange(4), np.array([1, 2]), np.array([2])]
        self.assertEqual(score(picks, actual2, Scoring((10, 20))), 10)
        self.assertEqual(score(picks, [np.arange(4), np.array([-1, -1]), np.array([-1])], Scoring((10, 20))), 0)

    def test_expected_points_equal_a_direct_calculation(self):
        b = small_bracket(4)
        P = strength_matrix(4, 5)
        lv = exact_levels(b, P)
        picks = favorite_bracket(b, P)
        direct = sum(10 * lv.arrays[1][j, picks.winners[1][j]] for j in range(2)) + 20 * lv.arrays[2][0, picks.winners[2][0]]
        self.assertAlmostEqual(expected_points(picks, lv, Scoring((10, 20))), direct)


if __name__ == "__main__":
    unittest.main()
