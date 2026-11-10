import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from api import publish
from api.sample import sample_bundle
from api.store import FileStore, ModelVersion


def good_pairs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(n)]
    rows_a, rows_b, p = [], [], []
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            rows_a.append(a), rows_b.append(b), p.append(float(rng.uniform(0.1, 0.9)))
    p = np.array(p)
    pred = pd.DataFrame({"p_a_win": p, "margin_hat": (p - 0.5) * 30, "total_reg_hat": 140.0, "score_a_hat": 70 + (p - 0.5) * 15, "score_b_hat": 70 - (p - 0.5) * 15})
    eff = 0.7 * (p - 0.5)
    effects = pd.DataFrame({"p_a_win": p, "efficiency": eff, "form": 0.2 * (p - 0.5), "interaction": 0.1 * (p - 0.5)})
    return teams, publish.assemble_pairs(rows_a, rows_b, pred, effects)


class PairChecks(unittest.TestCase):
    def test_a_sound_set_of_pairs_passes(self):
        teams, pairs = good_pairs()
        self.assertEqual(publish.check_pairs(pairs, teams), [])
        self.assertEqual(len(pairs), 15)
        self.assertAlmostEqual(sum(pairs.iloc[0]["factors"].values()), pairs.iloc[0]["p_a_win"] - 0.5, places=5)

    def test_each_kind_of_problem_is_caught(self):
        teams, pairs = good_pairs()
        cases = {
            "missing pairs": pairs.iloc[:-1],
            "duplicates": pd.concat([pairs.iloc[:-1], pairs.iloc[[0]]], ignore_index=True),
            "order": pairs.assign(team_a=pairs["team_b"], team_b=pairs["team_a"]),
            "probability": pairs.assign(p_a_win=[1.0] + list(pairs["p_a_win"][1:])),
            "nan probability": pairs.assign(p_a_win=[np.nan] + list(pairs["p_a_win"][1:])),
            "strangers": pairs.assign(team_a=["zzz"] + list(pairs["team_a"][1:])),
            "factors": pairs.assign(factors=[{"efficiency": 0.4, "interaction": 0.0}] + list(pairs["factors"][1:])),
            "partial scores": pairs.assign(score_a=[np.nan] + list(pairs["score_a"][1:])),
        }
        for label, bad in cases.items():
            with self.subTest(label):
                self.assertTrue(publish.check_pairs(bad, teams), label)

    def test_scores_can_be_absent_for_every_pair(self):
        teams, pairs = good_pairs()
        none = pairs.assign(margin_a=np.nan, total=np.nan, score_a=np.nan, score_b=np.nan)
        self.assertEqual(publish.check_pairs(none, teams), [])


class BundleTests(unittest.TestCase):
    def model(self, seasons):
        return ModelVersion("m", "m", "r", "live", seasons, "h")

    def test_in_sample_comes_from_the_training_seasons(self):
        teams, pairs = good_pairs()
        tdf = pd.DataFrame({"team": teams, "name": teams, "values": [{}] * len(teams)})
        a = publish.make_bundle(self.model([2022, 2023]), 2024, "2024-03-17", "selection_sunday", pairs, tdf, None, (), {(1, 16): 0.02})
        b = publish.make_bundle(self.model([2022, 2023, 2024]), 2024, "2024-03-17", "selection_sunday", pairs, tdf, None, (), {})
        self.assertFalse(a.snapshot.in_sample)
        self.assertTrue(b.snapshot.in_sample)
        self.assertEqual(a.snapshot.meta["seed_history"], {"1-16": 0.02})
        self.assertEqual(a.snapshot.n_teams, len(teams))
        self.assertIn("pairs", publish.size_report(a))

    def test_field_table_uses_slugs_and_names(self):
        t = pd.DataFrame({"team": ["www.sports-reference.com/cbb/schools/duke/men"], "seed": [1], "region": ["East"]})
        f = publish.field_table(t, {"www.sports-reference.com/cbb/schools/duke/men": "Duke Blue Devils"})
        self.assertEqual(f.iloc[0].to_dict(), {"team": "duke", "name": "Duke Blue Devils", "seed": 1, "region": "East"})

    def test_team_rows_keep_only_headline_numbers(self):
        rows = pd.DataFrame([{"team": "www.sports-reference.com/cbb/schools/duke/men", "adj_em": 25.123456, "elo": 1900.0, "unrelated": 5, "efg": np.nan}])
        out = publish.team_rows(rows, {})
        self.assertEqual(out.iloc[0]["team"], "duke")
        self.assertEqual(out.iloc[0]["values"], {"adj_em": 25.123, "elo": 1900.0})

    def test_results_keep_only_games_between_teams_in_the_bracket(self):
        pre = "www.sports-reference.com/cbb/schools/"
        results = pd.DataFrame([(f"{pre}duke/men", f"{pre}houston/men", f"{pre}duke/men"), (f"{pre}duke/men", f"{pre}outsider/men", f"{pre}duke/men"),
                                (f"{pre}duke/men", f"{pre}houston/men", f"{pre}someone-else/men")], columns=["team_a", "team_b", "winner"])
        out = publish.results_json(results, [f"{pre}duke/men", f"{pre}houston/men"])
        self.assertEqual(out, [{"team_a": "duke", "team_b": "houston", "winner": "duke"}])  # not the outsider, and not a winner who did not play

    def test_a_bundle_carries_the_results(self):
        teams, pairs = good_pairs()
        tdf = pd.DataFrame({"team": teams, "name": teams, "values": [{}] * len(teams)})
        field = pd.DataFrame({"team": ["t0"], "name": ["t0"], "seed": [1], "region": ["East"]})
        b = publish.make_bundle(self.model([2022]), 2024, "2024-03-17", "selection_sunday", pairs, tdf, field, (("East", "West"), ("South", "Midwest")), {},
                                results=[{"team_a": "t0", "team_b": "t1", "winner": "t0"}])
        self.assertEqual(b.bracket["results"], [{"team_a": "t0", "team_b": "t1", "winner": "t0"}])
        self.assertEqual(publish.make_bundle(self.model([2022]), 2024, "2024-03-17", "selection_sunday", pairs, tdf, field, [], {}).bracket["results"], [])

    def model(self, seasons):
        return ModelVersion("m", "m", "r", "live", seasons, "h")

    def test_seed_history_keys(self):
        self.assertEqual(publish.seed_history_json({(8, 9): 0.5123456}), {"8-9": 0.5123})


class DryRunAndRepublish(unittest.TestCase):
    def test_a_check_failure_means_nothing_is_written(self):
        teams, pairs = good_pairs()
        self.assertTrue(publish.check_pairs(pairs.iloc[:-1], teams))
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(d)
            self.assertEqual(store.snapshots(), [])  # a failed check never reaches write_snapshot

    def test_publishing_twice_is_refused_then_replaces_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(Path(d))
            b = sample_bundle()
            store.write_snapshot(b)
            with self.assertRaises(FileExistsError):
                store.write_snapshot(b)
            info = store.write_snapshot(b, replace=True)
            self.assertEqual(info["pairs"], len(b.pairs))
            self.assertEqual(len(store.snapshots()), 1)


if __name__ == "__main__":
    unittest.main()
