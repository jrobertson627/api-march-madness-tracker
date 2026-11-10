import unittest

import numpy as np
import pandas as pd

from data.features import (
    FEATURES,
    GLOBAL_FEATURES,
    assemble_feature_matrix,
    build_games,
    coach_features,
    column_dictionary,
    compute_form,
    matchup_for,
    possessions,
    with_both_orderings,
)
from data.snapshots import STAT_COLS

T = pd.Timestamp
CUTOFF = T("2026-03-15")
TEAMS = [f"t{i}" for i in range(8)]


def game_group(date):
    if date >= T("2026-03-17"):
        return "ncaa"
    if date >= T("2026-03-04"):
        return "conf_tourney"
    return "regular"


def game_type(group, date):
    if group == "ncaa":
        return "ROUND-64" if date < T("2026-03-21") else "ROUND-32"
    return "CTOURN" if group == "conf_tourney" else "REG (Conf)"


def make_world(seed=0, extra_ncaa=()):
    """Eight teams, one season, a game or two a day, with consistent facts and box scores."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 5, len(TEAMS))))
    facts, box = [], []
    n = 0
    dates = list(pd.date_range("2025-11-10", "2026-03-14", freq="D")) + [T(d) for d in ("2026-03-19", "2026-03-22")]
    pairs = [(d, *rng.choice(TEAMS, 2, replace=False)) for d in dates for _ in range(2)]
    pairs += [(T(d), a, b) for d, a, b in extra_ncaa]
    seen = set()
    for date, a, b in pairs:
        if (date, a) in seen or (date, b) in seen or a == b:
            continue
        seen |= {(date, a), (date, b)}
        grp = game_group(date)
        loc = 0 if grp == "ncaa" else int(rng.choice([1, -1, 0]))
        n += 1
        stats = {}
        for t in (a, b):
            fga = int(rng.integers(52, 66))
            fg3 = int(rng.integers(4, 11))
            fg = int(fga * (0.42 + 0.01 * strength[t] / 5 + rng.normal(0, 0.03)))
            ft = int(rng.integers(8, 22))
            stats[t] = dict(
                fg=fg, fga=fga, fg3=fg3, fg3a=fg3 + int(rng.integers(10, 20)), ft=ft, fta=ft + int(rng.integers(0, 6)),
                orb=int(rng.integers(5, 14)), drb=int(rng.integers(18, 28)), trb=30, ast=12, stl=6, blk=3,
                tov=int(rng.integers(8, 16)), pf=15,
            )
            stats[t]["pts"] = 2 * fg + fg3 + ft
        for t, o, lc in ((a, b, loc), (b, a, -loc)):
            row_box = dict(
                game_id=n, box_path=f"g{n}", game_date=date, team=t, team_name=t, opp=o, season=2026,
                pts_for=stats[t]["pts"], pts_against=stats[o]["pts"],
                **{c: stats[t][c] for c in STAT_COLS}, **{f"o_{c}": stats[o][c] for c in STAT_COLS},
            )
            box.append(row_box)
            facts.append(
                dict(
                    season=2026, team=t, opp=o, game_date=date, game_type=game_type(grp, date), game_group=grp,
                    overtimes=0, pts_for=stats[t]["pts"], pts_against=stats[o]["pts"], loc=lc, box_path=f"g{n}",
                )
            )
    box = pd.DataFrame(box)
    box["poss"] = possessions(box)
    facts = pd.DataFrame(facts).merge(box[["box_path", "team", "poss"]], on=["box_path", "team"], how="left")
    facts["has_box"] = facts["poss"].notna()
    meta = pd.DataFrame(
        {
            "season": 2026,
            "team": TEAMS,
            "conference": "Test",
            "coach_name": [f"coach{i}" for i in range(8)],
            "ncaa_seed": [1, 16, 8, 9, np.nan, np.nan, np.nan, np.nan],
            "ncaa_region": "East",
        }
    )
    return facts, box, meta


def run(facts, box, meta, **kw):
    kw.setdefault("min_games", 3)
    return assemble_feature_matrix(facts, box, {}, {}, meta, **kw)


def feature_cols(m):
    own = set(m.columns)
    cols = [c for c in m.columns if c.endswith(("_a", "_b")) and c[:-2] in FEATURES]
    cols += [c for c in own if c.startswith("diff_")]
    cols += [c for c in GLOBAL_FEATURES if c in own]
    cols += [c for c in ("proj_poss", "proj_pts_a", "proj_pts_b", "proj_margin", "proj_total", "elo_expected_a")]
    return sorted(cols)


class MatchupLeakageTests(unittest.TestCase):
    def test_features_ignore_games_on_or_after_the_row_date(self):
        facts, box, meta = make_world()
        base = run(facts, box, meta)["matchups"].set_index("game_id")

        cut = T("2026-02-01")
        pf, pb = facts.copy(), box.copy()
        pb[STAT_COLS + [f"o_{c}" for c in STAT_COLS]] = pb[STAT_COLS + [f"o_{c}" for c in STAT_COLS]].astype(float)
        for frame in (pf, pb):
            frame[["pts_for", "pts_against"]] = frame[["pts_for", "pts_against"]].astype(float)
            late = frame["game_date"] >= cut
            frame.loc[late, "pts_for"] = frame.loc[late, "pts_for"] * 3 + 40
            frame.loc[late, "pts_against"] = frame.loc[late, "pts_against"] * 0.2
        for c in STAT_COLS:
            pb.loc[pb["game_date"] >= cut, c] = pb.loc[pb["game_date"] >= cut, c] * 2
        pb["poss"] = possessions(pb)
        pf = pf.drop(columns="poss").merge(pb[["box_path", "team", "poss"]], on=["box_path", "team"], how="left")
        poisoned = run(pf, pb, meta)["matchups"].set_index("game_id")

        early = base.index[base["game_date"] <= cut]
        self.assertGreater(len(early), 20)
        cols = feature_cols(base)
        pd.testing.assert_frame_equal(base.loc[early, cols], poisoned.loc[early, cols])
        late_ids = base.index[base["game_date"] > cut]
        self.assertFalse(base.loc[late_ids, cols].equals(poisoned.loc[late_ids, cols]))  # the poison does reach later rows

    def test_every_row_uses_an_as_of_before_its_game(self):
        facts, box, meta = make_world()
        m = run(facts, box, meta)["matchups"]
        self.assertTrue((m["as_of"] < m["game_date"]).all())


class TournamentAsOfTests(unittest.TestCase):
    def setUp(self):
        self.facts, self.box, self.meta = make_world(extra_ncaa=[("2026-03-19", "t0", "t1"), ("2026-03-22", "t0", "t2")])

    def test_later_round_sees_earlier_round_in_game_mode(self):
        m = run(self.facts, self.box, self.meta, mode="game")["matchups"]
        row = m[(m["game_date"] == T("2026-03-22")) & (m["team_a"] == "t0") & (m["team_b"] == "t2")].iloc[0]
        self.assertEqual(row["as_of"], T("2026-03-21"))
        self.assertEqual(row["as_of_mode"], "game")
        first_round = m[(m["game_date"] == T("2026-03-19")) & (m["team_a"] == "t0") & (m["team_b"] == "t1")].iloc[0]
        self.assertEqual(row["games_a"], first_round["games_a"] + 1)  # saw the March 19 game

    def test_bracket_mode_uses_selection_sunday_for_every_round(self):
        m = run(self.facts, self.box, self.meta, mode="bracket")["matchups"]
        ncaa = m[m["game_group"] == "ncaa"]
        self.assertTrue((ncaa["as_of"] == CUTOFF).all())
        self.assertTrue((m[m["game_group"] != "ncaa"]["as_of"] < CUTOFF).all())

    def test_seed_only_known_from_selection_sunday(self):
        ta = run(self.facts, self.box, self.meta)["team_asof"]
        self.assertTrue(ta[ta["as_of"] < CUTOFF]["seed"].isna().all())
        seeded = ta[(ta["as_of"] >= CUTOFF) & (ta["team"] == "t0")]
        self.assertTrue((seeded["seed"] == 1).all())


class SymmetryTests(unittest.TestCase):
    def test_swapping_flips_everything_signed(self):
        facts, box, meta = make_world()
        m = run(facts, box, meta)["matchups"]
        both = with_both_orderings(m)
        orig, sw = both[both["swapped"] == 0].reset_index(drop=True), both[both["swapped"] == 1].reset_index(drop=True)
        self.assertEqual(len(orig), len(sw))
        pd.testing.assert_series_equal(sw["team_a"], orig["team_b"].rename("team_a"))
        pd.testing.assert_series_equal(sw["a_win"], 1 - orig["a_win"])
        pd.testing.assert_series_equal(sw["loc_a"], -orig["loc_a"])
        for c in [c for c in orig.columns if c.startswith("diff_")] + ["proj_margin", "margin_a"]:
            pd.testing.assert_series_equal(sw[c], -orig[c], check_names=False)
        pd.testing.assert_series_equal(sw["proj_pts_a"], orig["proj_pts_b"], check_names=False)
        pd.testing.assert_series_equal(sw["proj_total"], orig["proj_total"])
        pd.testing.assert_series_equal(sw["elo_expected_a"], 1 - orig["elo_expected_a"])

    def test_projected_score_is_symmetric_when_recomputed(self):
        # Swapping the teams by hand through the Phase 4 helper gives the mirror image.
        facts, box, meta = make_world()
        ta = run(facts, box, meta)["team_asof"]
        as_of = ta["as_of"].max()
        ab = matchup_for(ta, 2026, "t0", "t1", as_of, loc=1).iloc[0]
        ba = matchup_for(ta, 2026, "t1", "t0", as_of, loc=-1).iloc[0]  # same game, other labelling
        self.assertAlmostEqual(ab["proj_margin"], ba["proj_margin"], places=9)
        self.assertEqual(ab["team_a"], ba["team_a"])


class RegistryTests(unittest.TestCase):
    def test_every_output_column_is_documented(self):
        facts, box, meta = make_world()
        out = run(facts, box, meta)
        d = column_dictionary(out["team_asof"].columns, out["matchups"].columns)
        self.assertEqual(set(d["table"]), {"team_asof", "matchups"})

    def test_undocumented_column_fails(self):
        with self.assertRaises(KeyError):
            column_dictionary(["season", "team", "as_of", "mystery"], [])


class FeatureDetailTests(unittest.TestCase):
    def test_form_uses_recent_d1_games_only(self):
        rows = [
            ("a", "b", f"2025-12-{d:02d}", m, 0)
            for d, m in zip(range(1, 13), [10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0])
        ]
        f = pd.DataFrame(rows, columns=["team", "opp", "game_date", "pts_for", "pts_against"]).assign(
            game_date=lambda d: pd.to_datetime(d["game_date"]), season=2026
        )
        f.loc[len(f)] = ["a", np.nan, T("2025-12-13"), 100, 0, 2026]  # non-D1 blowout, ignored
        out = compute_form(f, 2026, "2025-12-31").set_index("team").loc["a"]
        self.assertEqual(out["last5_margin"], 0.0)
        self.assertEqual(out["last10_margin"], 5.0)  # games 3..12: five 10s and five 0s
        self.assertAlmostEqual(out["form_trend"], 0.0 - 70 / 12)  # last 5 minus the season mean of 12 D1 games

    def test_coach_tenure_and_prior_bids(self):
        meta = pd.DataFrame(
            {
                "team": ["x", "x", "x", "y"],
                "season": [2024, 2025, 2026, 2026],
                "coach_name": ["Ann", "Ann", "Bob", None],
                "ncaa_seed": [5, np.nan, np.nan, 3],
            }
        )
        c = coach_features(meta).set_index(["team", "season"])
        self.assertEqual(c.loc[("x", 2025), "coach_seasons_at_school"], 2)
        self.assertEqual(c.loc[("x", 2025), "coach_prior_ncaa_apps"], 1)
        self.assertEqual(c.loc[("x", 2026), "coach_seasons_at_school"], 1)
        self.assertEqual(c.loc[("x", 2026), "coach_prior_ncaa_apps"], 0)
        self.assertTrue(pd.isna(c.loc[("y", 2026), "coach_seasons_at_school"]))

    def test_trainable_requires_minimum_games(self):
        facts, box, meta = make_world()
        m = run(facts, box, meta, min_games=10_000)["matchups"]
        self.assertEqual(m["trainable"].sum(), 0)
        m = run(facts, box, meta, min_games=3)["matchups"]
        self.assertGreater(m["trainable"].sum(), 0)

    def test_build_games_one_row_per_game_with_canonical_order(self):
        facts, _, _ = make_world()
        g = build_games(facts)
        self.assertTrue((g["team_a"] < g["team_b"]).all())
        self.assertEqual(len(g), len(facts) // 2)
        self.assertEqual(g["game_id"].nunique(), len(g))


if __name__ == "__main__":
    unittest.main()
