import unittest

import numpy as np
import pandas as pd

from data.ratings import adjusted_ratings, elo_asof, elo_history, sos_asof

T = pd.Timestamp


def simulate_league(n_teams=30, games_each=24, true_h=3.0, noise=5.0, seed=0):
    """Round-robin-ish league with known offense/defense strengths and home court."""
    rng = np.random.default_rng(seed)
    names = [f"t{i:02d}" for i in range(n_teams)]
    off = rng.normal(0, 4, n_teams)
    dfn = rng.normal(0, 4, n_teams)
    mu, mu_t = 105.0, 68.0
    rows = []
    day = T("2025-11-10")
    for gi in range(games_each * n_teams // 2):
        a, b = rng.choice(n_teams, 2, replace=False)
        loc = rng.choice([1, -1, 0], p=[0.45, 0.45, 0.1])  # +1: a at home
        poss = mu_t + rng.normal(0, 3)
        date = day + pd.Timedelta(days=gi // 15)
        for t, o, lc in ((a, b, loc), (b, a, -loc)):
            eff = mu + off[t] + dfn[o] + true_h * lc + rng.normal(0, noise)
            rows.append(
                dict(season=2026, team=names[t], opp=names[o], game_date=date, loc=lc, pts_for=eff * poss / 100, poss=poss)
            )
    return pd.DataFrame(rows), pd.Series(off - dfn, index=names)


class AdjustedRatingsTests(unittest.TestCase):
    def setUp(self):
        self.facts, self.true_em = simulate_league()
        self.end = self.facts["game_date"].max()

    def test_recovers_known_strengths_and_home_court(self):
        r = adjusted_ratings(self.facts, 2026, self.end).set_index("team")
        corr = np.corrcoef(r["adj_em"], self.true_em.reindex(r.index))[0, 1]
        self.assertGreater(corr, 0.95)
        self.assertAlmostEqual(r["hca"].iloc[0], 3.0, delta=1.0)
        self.assertAlmostEqual(r["adj_off"].mean() + 0, r["lg_eff"].iloc[0], places=6)  # centred

    def test_more_shrinkage_pulls_toward_zero(self):
        loose = adjusted_ratings(self.facts, 2026, self.end, shrink_games=1)["adj_em"].std()
        tight = adjusted_ratings(self.facts, 2026, self.end, shrink_games=500)["adj_em"].std()
        self.assertLess(tight, loose * 0.3)

    def test_ignores_games_after_as_of(self):
        mid = self.facts["game_date"].quantile(0.5)
        base = adjusted_ratings(self.facts, 2026, mid)
        poisoned = self.facts.copy()
        later = poisoned["game_date"] > mid
        poisoned.loc[later, "pts_for"] = 999.0
        pd.testing.assert_frame_equal(base, adjusted_ratings(poisoned, 2026, mid))

    def test_skips_non_d1_and_missing_box_rows(self):
        f = self.facts.copy()
        f.loc[f.index[:50], "opp"] = np.nan
        f.loc[f.index[50:100], "poss"] = np.nan
        r = adjusted_ratings(f, 2026, self.end)
        self.assertFalse(r.empty)
        self.assertLessEqual(r["rated_games"].sum(), len(f) - 100)

    def test_empty_when_no_games(self):
        self.assertTrue(adjusted_ratings(self.facts, 2026, "2020-01-01").empty)


def games_frame(rows):
    return pd.DataFrame(rows, columns=["season", "game_date", "team_a", "team_b", "score_a", "score_b", "loc_a"]).assign(
        game_date=lambda d: pd.to_datetime(d["game_date"])
    )


class EloTests(unittest.TestCase):
    def test_changes_sum_to_zero_each_date(self):
        g = games_frame(
            [
                (2025, "2024-11-05", "a", "b", 80, 70, 1),
                (2025, "2024-11-05", "c", "d", 60, 75, 0),
                (2025, "2024-11-09", "a", "c", 70, 71, -1),
                (2025, "2024-11-09", "b", "d", 90, 60, 0),
            ]
        )
        h = elo_history(g)
        h = h[~h["is_start"]]
        change = (h["elo_post"] - h["elo_pre"]).groupby(h["game_date"]).sum()
        self.assertTrue(np.allclose(change, 0.0))

    def test_order_within_a_date_does_not_matter(self):
        rows = [
            (2025, "2024-11-05", "a", "b", 80, 70, 1),
            (2025, "2024-11-05", "a", "c", 60, 75, 0),  # a plays twice in one day
            (2025, "2024-11-09", "b", "c", 70, 71, -1),
        ]
        h1 = elo_history(games_frame(rows))
        h2 = elo_history(games_frame([rows[1], rows[0], rows[2]]))
        e1 = elo_asof(h1, 2025, "2024-12-01").set_index("team")["elo"]
        e2 = elo_asof(h2, 2025, "2024-12-01").set_index("team")["elo"]
        pd.testing.assert_series_equal(e1.sort_index(), e2.sort_index())

    def test_season_carry_over_and_new_teams(self):
        g = games_frame(
            [
                (2025, "2024-11-05", "a", "b", 90, 60, 0),
                (2026, "2025-11-05", "a", "c", 70, 69, 0),
            ]
        )
        h = elo_history(g, carry=0.5, new_team_start=1400)
        end_2025 = elo_asof(h, 2025, "2025-06-01").set_index("team")["elo"]
        start_2026 = h[(h["season"] == 2026) & h["is_start"]].set_index("team")["elo_post"]
        self.assertAlmostEqual(start_2026["a"], 1500 + 0.5 * (end_2025["a"] - 1500))
        self.assertAlmostEqual(start_2026["c"], 1400)
        first_year = h[(h["season"] == 2025) & h["is_start"]]["elo_post"]
        self.assertTrue((first_year == 1500).all())

    def test_asof_ignores_later_games(self):
        g = games_frame(
            [
                (2025, "2024-11-05", "a", "b", 80, 70, 0),
                (2025, "2024-12-05", "a", "b", 10, 100, 0),
            ]
        )
        early = elo_asof(elo_history(g), 2025, "2024-11-20").set_index("team")
        only_first = elo_asof(elo_history(g.iloc[:1]), 2025, "2024-11-20").set_index("team")
        pd.testing.assert_frame_equal(early.sort_index(), only_first.sort_index())
        self.assertEqual(early.loc["a", "elo_games"], 1)

    def test_home_team_gains_less_for_the_same_win(self):
        home = elo_history(games_frame([(2025, "2024-11-05", "a", "b", 80, 70, 1)]))
        neutral = elo_history(games_frame([(2025, "2024-11-05", "a", "b", 80, 70, 0)]))
        gain = lambda h: elo_asof(h, 2025, "2024-12-01").set_index("team").loc["a", "elo"] - 1500
        self.assertLess(gain(home), gain(neutral))


class SosTests(unittest.TestCase):
    def test_sos_elo_uses_opponent_rating_before_the_game(self):
        h = elo_history(games_frame([(2025, "2024-11-05", "a", "b", 99, 50, 0)]))
        s = sos_asof(h, pd.DataFrame(columns=["team", "adj_em"]), 2025, "2024-12-01").set_index("team")
        # Both teams were 1500 before the game, whatever the result did to them after.
        self.assertEqual(s.loc["a", "sos_elo"], 1500)
        self.assertEqual(s.loc["b", "sos_elo"], 1500)

    def test_sos_adj_averages_opponent_adjusted_margin(self):
        h = elo_history(
            games_frame([(2025, "2024-11-05", "a", "b", 70, 60, 0), (2025, "2024-11-08", "a", "c", 70, 60, 0)])
        )
        r = pd.DataFrame({"team": ["a", "b", "c"], "adj_em": [0.0, 4.0, 8.0]})
        s = sos_asof(h, r, 2025, "2024-12-01").set_index("team")
        self.assertAlmostEqual(s.loc["a", "sos_adj"], 6.0)


if __name__ == "__main__":
    unittest.main()
