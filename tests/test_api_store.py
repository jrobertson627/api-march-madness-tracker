import tempfile
import unittest
from pathlib import Path

import numpy as np

from api.ids import is_slug, slug_of, team_id_of
from api.sample import sample_bundle
from api.store import FileStore, _text


class IdTests(unittest.TestCase):
    def test_slugs(self):
        self.assertEqual(slug_of("www.sports-reference.com/cbb/schools/michigan-state/men"), "michigan-state")
        self.assertEqual(slug_of("duke"), "duke")
        self.assertEqual(team_id_of("duke"), "www.sports-reference.com/cbb/schools/duke/men")
        self.assertEqual(slug_of(team_id_of("texas-am")), "texas-am")
        for ok in ("duke", "texas-am", "n-c-state", "a1"):
            self.assertTrue(is_slug(ok), ok)
        for bad in ("", "Duke", "du ke", "-duke", "duke/men", "x" * 90, "du;ke"):
            self.assertFalse(is_slug(bad), bad)


class MissingNameTests(unittest.TestCase):
    def test_a_missing_name_is_none_not_the_text_nan(self):
        self.assertIsNone(_text(float("nan")))
        self.assertIsNone(_text(None))
        self.assertEqual(_text("Duke"), "Duke")


class FileStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = FileStore(self.dir.name)
        self.bundle = sample_bundle()

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip(self):
        info = self.store.write_snapshot(self.bundle, activate=True)
        n = self.bundle.snapshot.n_teams
        self.assertEqual(info["pairs"], n * (n - 1) // 2)
        self.assertEqual([m.id for m in self.store.models()], ["sample-model"])
        self.assertTrue(self.store.active_model().active)
        snaps = self.store.snapshots(2030)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(self.store.snapshots(1999), [])
        snap = snaps[0]
        row = self.store.pair(snap, "e1", "w1")
        want = self.bundle.pairs[(self.bundle.pairs.team_a == "e1") & (self.bundle.pairs.team_b == "w1")].iloc[0]
        self.assertAlmostEqual(row["p_a_win"], want.p_a_win)
        self.assertAlmostEqual(sum(v for k, v in row["factors"].items()), row["p_a_win"] - 0.5, places=9)
        self.assertIsNone(self.store.pair(snap, "e1", "nobody"))

    def test_pairs_among_and_teams_and_bracket(self):
        self.store.write_snapshot(self.bundle)
        snap = self.store.snapshots(2030)[0]
        field = list(self.bundle.bracket["teams"]["team"])
        among = self.store.pairs_among(snap, field)
        self.assertEqual(len(among), len(field) * (len(field) - 1) // 2)
        self.assertTrue(set(among.team_a) | set(among.team_b) <= set(field))
        self.assertEqual(len(self.store.teams(snap)), snap.n_teams)
        b = self.store.bracket(2030)
        self.assertEqual(len(b["teams"]), 68)
        self.assertEqual(b["final_four"], [["East", "West"], ["South", "Midwest"]])
        self.assertIsNone(self.store.bracket(1999))

    def test_a_published_snapshot_is_immutable_unless_replaced(self):
        self.store.write_snapshot(self.bundle)
        with self.assertRaises(FileExistsError):
            self.store.write_snapshot(self.bundle)
        self.store.write_snapshot(sample_bundle(seed=5), replace=True)
        snap = self.store.snapshots(2030)[0]
        self.assertAlmostEqual(self.store.pair(snap, "e1", "w1")["p_a_win"],
                               sample_bundle(seed=5).pairs.query("team_a == 'e1' and team_b == 'w1'").iloc[0].p_a_win)

    def test_activating_a_new_model_deactivates_the_old(self):
        self.store.write_snapshot(self.bundle, activate=True)
        self.store.write_snapshot(sample_bundle(model_id="second", season=2031), activate=True)
        active = [m.id for m in self.store.models() if m.active]
        self.assertEqual(active, ["second"])

    def test_results_are_stored_with_the_bracket_and_can_be_replaced(self):
        self.store.write_snapshot(self.bundle)
        self.assertEqual(len(self.store.bracket(2030)["results"]), 3)
        info = self.store.write_results(2030, [{"team_a": "s1", "team_b": "s16", "winner": "s16"}])
        self.assertEqual(info, {"season": 2030, "results": 1})
        self.assertEqual(self.store.bracket(2030)["results"], [{"team_a": "s1", "team_b": "s16", "winner": "s16"}])
        self.assertEqual(len(self.store.bracket(2030)["teams"]), 68)  # the rest of the bracket is untouched
        with self.assertRaises(FileNotFoundError):
            self.store.write_results(1999, [])

    def test_a_bracket_without_results_reads_as_an_empty_list(self):
        bundle = sample_bundle()
        del bundle.bracket["results"]
        self.store.write_snapshot(bundle)
        self.assertEqual(self.store.bracket(2030)["results"], [])

    def test_the_backtest_report_round_trips_and_the_newest_wins(self):
        self.assertIsNone(self.store.backtest())
        self.store.write_backtest({"limits": ["one"]}, "m1", "h1")
        first = self.store.backtest()
        self.assertEqual((first["model_id"], first["data_hash"], first["report"]), ("m1", "h1", {"limits": ["one"]}))
        self.assertTrue(first["created_at"])
        self.store.write_backtest({"limits": ["two"]}, "m1", "h2")
        self.assertEqual(self.store.backtest()["report"], {"limits": ["two"]})

    def test_empty_store(self):
        empty = FileStore(Path(self.dir.name) / "nothing")
        self.assertEqual(empty.models(), [])
        self.assertEqual(empty.snapshots(), [])
        self.assertIsNone(empty.active_model())
        self.assertFalse(empty.ping())


if __name__ == "__main__":
    unittest.main()
