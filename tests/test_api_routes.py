import tempfile
import unittest
from unittest import mock

import jsonschema
import numpy as np

from api.app import create_app
from api.config import Config
from api.sample import sample_bundle
from api.schemas import SCHEMAS
from api.store import FileStore, Store
from bracket.simulate import exact_levels
from bracket.structure import Bracket

SEASON = 2030


def validate(body, name):
    jsonschema.validate(body, SCHEMAS[name])


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.store = FileStore(cls.dir.name)
        cls.bundle = sample_bundle(SEASON, "out-of-sample", in_sample=False, seed=1)
        cls.store.write_snapshot(cls.bundle, activate=True)
        cls.store.write_snapshot(sample_bundle(SEASON, "in-sample", in_sample=True, seed=2))
        cls.app = create_app(Config(cors_origins=["http://localhost:3000"]), store=cls.store)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def get(self, url, **kw):
        return self.client.get(url, **kw)


class HealthAndListTests(Base):
    def test_health_ok(self):
        r = self.get("/health")
        self.assertEqual(r.status_code, 200)
        validate(r.get_json(), "health")
        self.assertEqual(r.get_json()["status"], "ok")
        self.assertEqual(r.get_json()["active_model"]["id"], "out-of-sample")

    def test_health_degraded_without_a_store(self):
        app = create_app(Config(store="none"), store=None)
        r = app.test_client().get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "degraded")
        self.assertEqual(app.test_client().get("/v1/models").status_code, 503)

    def test_models_and_seasons(self):
        m = self.get("/v1/models").get_json()
        validate(m, "models")
        self.assertEqual({x["id"] for x in m["models"]}, {"out-of-sample", "in-sample"})
        s = self.get("/v1/seasons").get_json()
        validate(s, "seasons")
        self.assertEqual(s["seasons"][0]["season"], SEASON)
        self.assertEqual(len(s["seasons"][0]["snapshots"]), 2)

    def test_request_id_and_cors(self):
        r = self.get("/health", headers={"Origin": "http://localhost:3000"})
        self.assertTrue(r.headers["X-Request-Id"])
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "http://localhost:3000")
        other = self.get("/health", headers={"Origin": "http://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", other.headers)


class MatchupTests(Base):
    URL = f"/v1/predict/matchup?season={SEASON}"

    def test_a_matchup_is_valid_and_consistent(self):
        r = self.get(f"{self.URL}&team_a=e1&team_b=w1")
        self.assertEqual(r.status_code, 200)
        b = r.get_json()
        validate(b, "matchup")
        self.assertAlmostEqual(b["p_a_win"] + b["p_b_win"], 1.0, places=3)
        p = b["projected"]
        self.assertAlmostEqual(p["score_a"] + p["score_b"], p["total"], delta=0.11)
        self.assertAlmostEqual(p["score_a"] - p["score_b"], p["margin_a"], delta=0.11)
        self.assertEqual(b["snapshot"]["model_id"], "out-of-sample")  # the out-of-sample model is preferred
        self.assertLessEqual(len(b["factors"]), 5)
        effects = [abs(f["effect"]) for f in b["factors"]]
        self.assertEqual(effects, sorted(effects, reverse=True))
        self.assertAlmostEqual(sum(f["effect"] for f in b["factors"]) + b["interaction"], b["p_a_win"] - 0.5, delta=0.002)
        stored = self.bundle.pairs.query("team_a == 'e1' and team_b == 'w1'").iloc[0]
        self.assertAlmostEqual(b["p_a_win"], stored.p_a_win, places=3)

    def test_swapping_the_teams_swaps_the_answer(self):
        a = self.get(f"{self.URL}&team_a=w1&team_b=e1").get_json()
        b = self.get(f"{self.URL}&team_a=e1&team_b=w1").get_json()
        self.assertAlmostEqual(a["p_a_win"], b["p_b_win"], places=3)
        self.assertAlmostEqual(a["projected"]["margin_a"], -b["projected"]["margin_a"], places=1)
        self.assertAlmostEqual(a["projected"]["score_a"], b["projected"]["score_b"], places=1)
        self.assertEqual(a["team_a"]["id"], b["team_b"]["id"])
        fa = {f["group"]: f["effect"] for f in a["factors"]}
        fb = {f["group"]: f["effect"] for f in b["factors"]}
        for g in set(fa) & set(fb):
            self.assertAlmostEqual(fa[g], -fb[g], places=3)

    def test_a_team_outside_the_field_works(self):
        r = self.get(f"{self.URL}&team_a=x0&team_b=e1")
        self.assertEqual(r.status_code, 200)
        validate(r.get_json(), "matchup")

    def test_choosing_the_in_sample_model_adds_a_note(self):
        b = self.get(f"{self.URL}&team_a=e1&team_b=w1&model=in-sample").get_json()
        self.assertTrue(b["snapshot"]["in_sample"])
        self.assertTrue(any("in-sample" in n for n in b["notes"]))

    def test_errors_have_one_shape(self):
        cases = [
            (f"/v1/predict/matchup?team_b=e1&season={SEASON}", 400, "bad_parameter"),
            (f"{self.URL}&team_a=E1&team_b=w1", 400, "bad_parameter"),
            (f"{self.URL}&team_a=e1&team_b=e1", 400, "same_team"),
            (f"{self.URL}&team_a=e1&team_b=nobody", 404, "unknown_team"),
            ("/v1/predict/matchup?team_a=e1&team_b=w1&season=1850", 400, "bad_parameter"),
            ("/v1/predict/matchup?team_a=e1&team_b=w1&season=abc", 400, "bad_parameter"),
            ("/v1/predict/matchup?team_a=e1&team_b=w1&season=1999", 404, "snapshot_not_found"),
            (f"{self.URL}&team_a=e1&team_b=w1&model=ghost", 404, "snapshot_not_found"),
            ("/v1/nothing", 404, "not_found"),
        ]
        for url, status, code in cases:
            with self.subTest(url=url):
                r = self.get(url)
                self.assertEqual(r.status_code, status)
                body = r.get_json()
                validate(body, "error")
                self.assertEqual(body["error"]["code"], code)

    def test_unknown_team_suggests_close_names(self):
        r = self.get(f"{self.URL}&team_a=e1&team_b=e11x")
        self.assertEqual(r.status_code, 404)
        self.assertIn("e11", r.get_json()["error"]["details"]["suggestions"])


class BracketTests(Base):
    def test_bracket_is_valid_and_sums(self):
        r = self.get(f"/v1/bracket/{SEASON}")
        self.assertEqual(r.status_code, 200)
        b = r.get_json()
        validate(b, "bracket")
        self.assertEqual(len(b["field"]), 68)
        self.assertAlmostEqual(sum(t["p_champion"] for t in b["field"]), 1.0, places=2)
        for name, rows in b["picks"].items():
            self.assertEqual(len(rows), 63, name)
        self.assertEqual(len(b["first_round"]), 32)
        self.assertGreaterEqual(b["expected_points"]["best_expected_points"] + 0.1, max(b["expected_points"].values()))
        self.assertEqual(b["scoring"], [10, 20, 40, 80, 160, 320])

    def test_bracket_equals_the_offline_calculation(self):
        b = self.get(f"/v1/bracket/{SEASON}").get_json()
        teams = self.bundle.bracket["teams"]
        bracket = Bracket.from_teams(teams[["team", "seed", "region"]], (("East", "West"), ("South", "Midwest")))
        n = len(bracket.teams)
        P = np.full((n, n), 0.5)
        sub = self.bundle.pairs[self.bundle.pairs.team_a.isin(teams.team) & self.bundle.pairs.team_b.isin(teams.team)]
        for r in sub.itertuples():
            i, j = bracket.index[r.team_a], bracket.index[r.team_b]
            P[i, j], P[j, i] = r.p_a_win, 1 - r.p_a_win
        want = exact_levels(bracket, P).reach()[:, -1]
        got = {t["team"]: t["p_champion"] for t in b["field"]}
        for team, p in zip(bracket.teams, want):
            self.assertAlmostEqual(got[team], p, places=3)

    def test_scoring_changes_the_expected_points(self):
        a = self.get(f"/v1/bracket/{SEASON}?scoring=1,2,4,8,16,32").get_json()
        b = self.get(f"/v1/bracket/{SEASON}").get_json()
        self.assertEqual(a["scoring"], [1, 2, 4, 8, 16, 32])
        self.assertLess(a["expected_points"]["best_expected_points"], b["expected_points"]["best_expected_points"])
        for bad in ("1,2,3", "a,b,c,d,e,f", "-1,2,3,4,5,6", "1,2,3,4,5,99999"):
            r = self.get(f"/v1/bracket/{SEASON}?scoring={bad}")
            self.assertEqual(r.status_code, 400, bad)

    def test_round_one_matchups(self):
        b = self.get(f"/v1/bracket/{SEASON}/matchups?round=1").get_json()
        validate(b, "round_matchups")
        self.assertEqual(len(b["games"]), 32)
        self.assertEqual(self.get(f"/v1/bracket/{SEASON}/matchups?round=2").status_code, 400)

    def test_unknown_season_has_no_bracket(self):
        self.assertEqual(self.get("/v1/bracket/1999").status_code, 404)


class WhatIfTests(Base):
    def post(self, body, season=SEASON):
        return self.client.post(f"/v1/bracket/{season}/odds", json=body)

    def test_a_locked_result_changes_the_odds(self):
        r = self.post({"results": [{"team_a": "s1", "team_b": "s16", "winner": "s16"}]})  # South has one 16 seed, so they meet in round one
        self.assertEqual(r.status_code, 200)
        b = r.get_json()
        validate(b, "bracket")
        by = {t["team"]: t for t in b["field"]}
        self.assertEqual(b["locked_results"], 1)
        self.assertEqual(by["s1"]["p_round_of_32"], 0.0)
        self.assertEqual(by["s1"]["p_champion"], 0.0)
        self.assertEqual(by["s16"]["p_round_of_32"], 1.0)
        self.assertAlmostEqual(sum(t["p_champion"] for t in b["field"]), 1.0, places=2)
        self.assertEqual(r.headers["Cache-Control"], "no-store")

    def test_no_results_equals_the_get_answer(self):
        post = self.post({"results": []}).get_json()
        get = self.get(f"/v1/bracket/{SEASON}").get_json()
        self.assertEqual(post["field"], get["field"])

    def test_bad_requests_are_rejected_with_a_reason(self):
        cases = [
            ({"results": [{"team_a": "e1", "team_b": "e16", "winner": "w1"}]}, 422, "bad_result"),
            ({"results": [{"team_a": "e1", "team_b": "nobody", "winner": "e1"}]}, 422, "bad_result"),
            ({"results": [{"team_a": "e1", "team_b": "e16"}]}, 422, "bad_result"),
            ({"results": [{"team_a": "e1", "team_b": "e16", "winner": "e1"}, {"team_a": "e16", "team_b": "e1", "winner": "e16"}]}, 422, "bad_result"),
            # e1 lost in round one, so it cannot be in a round two game
            ({"results": [{"team_a": "s1", "team_b": "s16", "winner": "s16"}, {"team_a": "s1", "team_b": "s8", "winner": "s1"}]}, 422, "impossible_result"),
            ({"results": "nope"}, 400, "bad_body"),
            ({"results": [], "scoring": [1, 2]}, 400, "bad_parameter"),
        ]
        for body, status, code in cases:
            with self.subTest(body=str(body)[:70]):
                r = self.post(body)
                self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (status, code))
                validate(r.get_json(), "error")
        self.assertEqual(self.client.post(f"/v1/bracket/{SEASON}/odds", data="not json", content_type="text/plain").status_code, 400)

    def test_an_oversized_body_is_refused(self):
        r = self.client.post(f"/v1/bracket/{SEASON}/odds", data="x" * 30_000, content_type="application/json")
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.get_json()["error"]["code"], "too_large")


class CachingTests(Base):
    def test_etag_and_not_modified(self):
        url = f"/v1/predict/matchup?season={SEASON}&team_a=e1&team_b=w1"
        r = self.get(url)
        etag = r.headers["ETag"]
        self.assertIn("max-age", r.headers["Cache-Control"])
        again = self.get(url, headers={"If-None-Match": etag})
        self.assertEqual(again.status_code, 304)
        self.assertEqual(again.data, b"")
        other = self.get(f"/v1/predict/matchup?season={SEASON}&team_a=e1&team_b=w2")
        self.assertNotEqual(other.headers["ETag"], etag)
        stale = self.get(url, headers={"If-None-Match": '"something-else"'})
        self.assertEqual(stale.status_code, 200)

    def test_a_repeated_bracket_request_hits_the_cache(self):
        svc = self.app.extensions["service"]
        self.get(f"/v1/bracket/{SEASON}")
        n = len(svc.results)
        self.get(f"/v1/bracket/{SEASON}")
        self.assertEqual(len(svc.results), n)


class FailureTests(unittest.TestCase):
    def test_a_broken_store_gives_503_not_a_crash(self):
        class Broken(Store):
            def snapshots(self, season=None):
                raise ConnectionError("db down")

            def models(self):
                raise ConnectionError("db down")

            def ping(self):
                return False

        app = create_app(Config(), store=Broken())
        c = app.test_client()
        for url in ("/v1/predict/matchup?team_a=a&team_b=b&season=2026", "/v1/models", "/v1/bracket/2026"):
            r = c.get(url)
            self.assertEqual(r.status_code, 503, url)
            self.assertEqual(r.get_json()["error"]["code"], "store_unavailable")
        self.assertEqual(c.get("/health").get_json()["status"], "degraded")

    def test_legacy_team_routes_still_exist(self):
        app = create_app(Config(), store=None)
        with mock.patch("data.teams.list_teams", return_value=[{"id": "duke"}]):
            r = app.test_client().get("/teams?season=2026")
        self.assertEqual(r.get_json(), [{"id": "duke"}])


class StaleServingTests(unittest.TestCase):
    def test_a_store_that_goes_down_still_serves_what_it_has_already_served(self):
        d = tempfile.TemporaryDirectory()
        real = FileStore(d.name)
        real.write_snapshot(sample_bundle(SEASON, "m", seed=4), activate=True)

        class Flaky(FileStore):
            down = False

            def _maybe(self):
                if Flaky.down:
                    raise ConnectionError("db down")

            def snapshots(self, season=None):
                self._maybe()
                return super().snapshots(season)

            def pair(self, snap, a, b):
                self._maybe()
                return super().pair(snap, a, b)

            def pairs_among(self, snap, teams):
                self._maybe()
                return super().pairs_among(snap, teams)

        store = Flaky(d.name)
        app = create_app(Config(), store=store)
        c = app.test_client()
        url = f"/v1/predict/matchup?season={SEASON}&team_a=e1&team_b=w1"
        warm = c.get(url).get_json()
        self.assertFalse(any("memory" in n for n in warm["notes"]))
        c.get(f"/v1/bracket/{SEASON}")
        Flaky.down = True
        try:
            # Within the ttl nothing needs the store: published data never changes, so it is served from memory.
            again = c.get(url)
            self.assertEqual((again.status_code, again.get_json()["p_a_win"]), (200, warm["p_a_win"]))
            # Once the cached lists expire the store is tried, fails, and the old list is used with a note.
            app.extensions["service"].ttl = 0
            later = c.get(url)
            self.assertEqual(later.status_code, 200)
            self.assertTrue(any("memory" in n for n in later.get_json()["notes"]))
            self.assertEqual(c.get(f"/v1/bracket/{SEASON}").status_code, 200)
            new_pair = c.get(f"/v1/predict/matchup?season={SEASON}&team_a=e2&team_b=w2")
            self.assertEqual(new_pair.status_code, 503)
            self.assertEqual(new_pair.get_json()["error"]["code"], "store_unavailable")
        finally:
            Flaky.down = False
            d.cleanup()


if __name__ == "__main__":
    unittest.main()
