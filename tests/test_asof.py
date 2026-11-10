import unittest

import pandas as pd

from data.asof import asof_for_games, asof_grid, assert_no_leak

T = pd.Timestamp


class AsOfGridTests(unittest.TestCase):
    def setUp(self):
        # 2026 season shape: Selection Sunday Mar 15, final Apr 6.
        self.cutoff = T("2026-03-15")
        self.grid = asof_grid("2025-11-03", self.cutoff, "2026-04-06")

    def test_weekly_then_daily(self):
        before = [d for d in self.grid if d <= self.cutoff]
        after = [d for d in self.grid if d > self.cutoff]
        self.assertTrue(all(d.weekday() == 6 for d in before))  # Sundays
        self.assertEqual(before[-1], self.cutoff)
        self.assertEqual(after[0], T("2026-03-16"))
        self.assertEqual(after[-1], T("2026-04-06"))
        self.assertEqual(len(after), 22)

    def test_cutoff_on_grid_even_if_not_sunday(self):
        grid = asof_grid("2025-11-03", "2026-03-14", "2026-03-20")
        self.assertIn(T("2026-03-14"), grid)

    def test_daily_cadence(self):
        grid = asof_grid("2025-11-03", self.cutoff, "2025-11-09", cadence="daily")
        self.assertEqual(len(grid), 7)

    def test_bad_cadence(self):
        with self.assertRaises(ValueError):
            asof_grid("2025-11-03", self.cutoff, "2026-04-06", cadence="hourly")


class AsOfForGamesTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = T("2026-03-15")
        self.grid = {2026: asof_grid("2025-11-02", self.cutoff, "2026-04-06")}
        self.cutoffs = {2026: self.cutoff}

    def games(self, rows):
        return pd.DataFrame(rows, columns=["season", "game_date", "game_group"]).assign(
            game_date=lambda d: pd.to_datetime(d["game_date"])
        )

    def test_strictly_before_the_game(self):
        # Game on a grid Sunday gets the previous Sunday, not itself.
        g = self.games([(2026, "2025-12-07", "regular"), (2026, "2025-12-10", "regular")])
        out = asof_for_games(g, self.grid, self.cutoffs)
        self.assertEqual(out.iloc[0], T("2025-11-30"))
        self.assertEqual(out.iloc[1], T("2025-12-07"))
        assert_no_leak(g.assign(as_of=out))

    def test_before_first_grid_date_is_nat(self):
        g = self.games([(2026, "2025-11-02", "regular"), (2026, "2025-11-01", "regular")])
        out = asof_for_games(g, self.grid, self.cutoffs)
        self.assertTrue(out.isna().all())

    def test_tournament_rounds_see_earlier_rounds(self):
        # Round of 64 Thursday and Sweet 16 Thursday get the day before in per-game mode.
        g = self.games([(2026, "2026-03-19", "ncaa"), (2026, "2026-03-26", "ncaa")])
        out = asof_for_games(g, self.grid, self.cutoffs, mode="game")
        self.assertEqual(out.iloc[0], T("2026-03-18"))
        self.assertEqual(out.iloc[1], T("2026-03-25"))

    def test_bracket_mode_pins_ncaa_to_selection_sunday(self):
        g = self.games(
            [(2026, "2026-03-19", "ncaa"), (2026, "2026-03-26", "ncaa"), (2026, "2026-03-10", "conf_tourney")]
        )
        out = asof_for_games(g, self.grid, self.cutoffs, mode="bracket")
        self.assertEqual(out.iloc[0], self.cutoff)
        self.assertEqual(out.iloc[1], self.cutoff)
        self.assertEqual(out.iloc[2], T("2026-03-08"))  # non-NCAA rows unchanged

    def test_leak_guard_raises(self):
        g = self.games([(2026, "2025-12-07", "regular")]).assign(as_of=T("2025-12-07"))
        with self.assertRaises(AssertionError):
            assert_no_leak(g)

    def test_bad_mode(self):
        g = self.games([(2026, "2025-12-07", "regular")])
        with self.assertRaises(ValueError):
            asof_for_games(g, self.grid, self.cutoffs, mode="nope")


if __name__ == "__main__":
    unittest.main()
