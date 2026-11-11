"""The API's logic, separate from Flask so it can be tested directly.

Everything reads a Store. Bracket odds, picks and flags are computed here from the stored pair probabilities with the
bracket package (numpy only), so nothing here needs scikit-learn.
"""

from __future__ import annotations

import difflib
import json
import time

import numpy as np
import pandas as pd

from api.cache import LRU
from api.errors import ApiError
from api.ids import is_slug
from api.store import Snapshot, Store
from bracket import pick, upsets
from bracket.outputs import ROUND_NAMES
from bracket.simulate import exact_levels, reach_table
from bracket.structure import REACH_COLUMNS, Bracket
from ml.explain import HEADLINE

DEFAULT_SCORING = pick.DEFAULT_POINTS
TOP_FACTORS = 5
PLAYED_ROUND_NAMES = {0: "first four", **ROUND_NAMES}
REPORT_TTL = 300.0  # seconds the backtest report is reused


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def season_param(value) -> int:
    try:
        s = int(value)
    except (TypeError, ValueError):
        raise ApiError("bad_parameter", "season must be a four-digit year", 400, {"season": value})
    if not 1990 <= s <= 2100:
        raise ApiError("bad_parameter", "season must be between 1990 and 2100", 400, {"season": value})
    return s


def slug_param(name: str, value) -> str:
    if value is None or value == "":
        raise ApiError("bad_parameter", f"{name} is required", 400, {"parameter": name})
    if not is_slug(value):
        raise ApiError("bad_parameter", f"{name} must be a school slug such as 'michigan-state'", 400, {"parameter": name, "value": str(value)[:80]})
    return value


def scoring_param(text, n_rounds: int = 6) -> tuple:
    if text is None or text == "":
        return DEFAULT_SCORING
    try:
        pts = tuple(int(x) for x in str(text).split(","))
    except ValueError:
        raise ApiError("bad_parameter", "scoring must be comma-separated whole numbers", 400, {"scoring": str(text)[:80]})
    if len(pts) != n_rounds or any(p < 0 or p > 10_000 for p in pts):
        raise ApiError("bad_parameter", f"scoring needs {n_rounds} numbers between 0 and 10000, one per round", 400, {"scoring": str(text)[:80]})
    return pts


def _model_json(m) -> dict:
    return {"id": m.id, "name": m.name, "kind": m.kind, "trained_seasons": list(m.training_seasons), "data_hash": m.data_hash,
            "created_at": m.created_at, "metrics": m.metrics, "active": m.active}


def _snap_json(s: Snapshot) -> dict:
    return {"model_id": s.model_id, "season": s.season, "as_of": s.as_of, "kind": s.kind, "in_sample": s.in_sample, "n_teams": s.n_teams}


class Service:
    def __init__(self, store: Store | None, cache_size: int = 256, ttl: float = 60.0):
        self.store = store
        self.ttl = ttl  # seconds a list of snapshots or models is reused; published data itself never changes
        self.contexts = LRU(32)
        self.results = LRU(cache_size)
        self.pairs = LRU(2000)
        self.snapshot_lists: dict = {}   # season -> (time fetched, list)
        self.model_list = None           # (time fetched, list)
        self.report = None               # (time fetched, backtest report)

    # ------------------------------------------------------------------ plumbing

    def _store(self) -> Store:
        if self.store is None:
            raise ApiError("store_unavailable", "No prediction store is configured.", 503)
        return self.store

    def _guard(self, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except ApiError:
            raise
        except Exception as e:  # database or file problems become a clear 503
            raise ApiError("store_unavailable", "The prediction store could not be read.", 503, {"reason": type(e).__name__})

    def _snapshots_for(self, season: int):
        """The snapshots of a season, reused for `ttl` seconds. If the store fails, the last list seen is used and flagged."""
        hit = self.snapshot_lists.get(season)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1], False
        try:
            snaps = self._store().snapshots(season)
            self.snapshot_lists[season] = (time.monotonic(), snaps)
            return snaps, False
        except ApiError:
            raise
        except Exception as e:
            if hit:
                return hit[1], True
            raise ApiError("store_unavailable", "The prediction store could not be read.", 503, {"reason": type(e).__name__})

    def _models_list(self):
        hit = self.model_list
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        try:
            models = self._store().models()
            self.model_list = (time.monotonic(), models)
            return models
        except ApiError:
            raise
        except Exception as e:
            if hit:
                return hit[1]
            raise ApiError("store_unavailable", "The prediction store could not be read.", 503, {"reason": type(e).__name__})

    def _resolve(self, season: int, model: str | None = None, as_of: str | None = None, kind: str | None = None):
        snaps, stale = self._snapshots_for(season)
        if model:
            snaps = [s for s in snaps if s.model_id == model]
        if as_of:
            snaps = [s for s in snaps if s.as_of == as_of]
        if kind:
            snaps = [s for s in snaps if s.kind == kind]
        if not snaps:
            raise ApiError("snapshot_not_found", f"No published predictions for season {season}" + (f" with model {model}" if model else ""), 404,
                           {"season": season, "model": model, "as_of": as_of})
        # prefer predictions from a model that did not train on the season, then the latest date, then the newest publish
        snaps = sorted(snaps, key=lambda s: (s.in_sample, _neg(s.as_of), _neg(s.created_at)))
        return snaps[0], stale

    def resolve_snapshot(self, season: int, model: str | None = None, as_of: str | None = None, kind: str | None = None) -> Snapshot:
        return self._resolve(season, model, as_of, kind)[0]

    def _pair(self, snap: Snapshot, lo: str, hi: str):
        """One stored pair. A pair never changes, so one seen before is served from memory without asking the store."""
        key = (snap.model_id, snap.season, snap.as_of, lo, hi)
        cached = self.pairs.get(key)
        if cached is not None:
            return cached, False
        try:
            row = self._store().pair(snap, lo, hi)
            if row is not None:
                self.pairs.put(key, row)
            return row, False
        except ApiError:
            raise
        except Exception as e:
            raise ApiError("store_unavailable", "The prediction store could not be read.", 503, {"reason": type(e).__name__})

    def _teams_table(self, snap: Snapshot) -> pd.DataFrame:
        key = ("teams", snap.model_id, snap.season, snap.as_of)
        t = self.contexts.get(key)
        if t is None:
            t = self._guard(self._store().teams, snap)
            self.contexts.put(key, t)
        return t

    def cache_age(self, snap: Snapshot) -> int:
        """Seconds a client may reuse a response: a day for settled history, five minutes for the live snapshot."""
        return 300 if snap.kind == "latest" else 86_400

    # ------------------------------------------------------------------ simple lists

    def models(self) -> dict:
        return {"models": [_model_json(m) for m in self._guard(self._store().models)]}

    def seasons(self) -> dict:
        snaps = self._guard(self._store().snapshots)
        with_bracket = self._guard(self._store().bracket_seasons)
        out = {}
        for s in snaps:
            out.setdefault(s.season, []).append(_snap_json(s))
        # has_bracket: a season in progress has predictions but no field until Selection Sunday, and /v1/bracket answers 404 for it
        return {"seasons": [{"season": k, "has_bracket": k in with_bracket, "snapshots": v} for k, v in sorted(out.items())]}

    def health(self) -> dict:
        ok = bool(self.store and self.store.ping())
        active, latest = None, None
        if ok:
            try:
                m = self.store.active_model()
                active = _model_json(m) if m else None
                snaps = self.store.snapshots()
                latest = _snap_json(max(snaps, key=lambda s: (s.as_of, s.created_at))) if snaps else None
            except Exception:
                ok = False
        return {"status": "ok" if ok else "degraded", "store": ok, "active_model": active, "latest_snapshot": latest}

    def teams(self, season: int, model: str | None = None) -> tuple[dict, Snapshot]:
        """The teams that have predictions for a season, with seed and region where the team is in the tournament field."""
        snap, stale = self._resolve(season, model)
        table = self._teams_table(snap)
        key = ("field", season)
        field = self.contexts.get(key)
        if field is None:
            b = self._guard(self._store().bracket, season)
            field = {} if b is None else {t.team: (int(t.seed), t.region) for t in b["teams"].itertuples()}
            self.contexts.put(key, field)
        rows = []
        for t in table.itertuples():
            seed, region = field.get(t.team, (None, None))
            rows.append({"id": t.team, "name": None if t.name is None or (isinstance(t.name, float) and np.isnan(t.name)) else t.name, "seed": seed, "region": region})
        rows.sort(key=lambda r: (r["name"] or r["id"]).lower())
        notes = []
        if snap.in_sample:
            notes.append(f"in-sample: this model was trained on {season}")
        if stale:
            notes.append("served from memory: the prediction store could not be reached")
        return {"season": season, "snapshot": _snap_json(snap), "teams": rows, "notes": notes}, snap

    def backtest(self) -> dict:
        """The newest published backtest report. Reused for a few minutes; if the store fails, the last one seen is served."""
        hit = self.report
        if hit and time.monotonic() - hit[0] < REPORT_TTL:
            return hit[1]
        try:
            doc = self._store().backtest()
        except ApiError:
            raise
        except Exception as e:
            if hit:
                return dict(hit[1], notes=["served from memory: the prediction store could not be reached"])
            raise ApiError("store_unavailable", "The prediction store could not be read.", 503, {"reason": type(e).__name__})
        if doc is None:
            raise ApiError("backtest_not_found", "No backtest report has been published", 404)
        body = {"model_id": doc["model_id"], "data_hash": doc["data_hash"], "created_at": doc["created_at"], **doc["report"]}
        self.report = (time.monotonic(), body)
        return body

    # ------------------------------------------------------------------ matchups

    def matchup(self, team_a: str, team_b: str, season: int, model: str | None = None, as_of: str | None = None) -> tuple[dict, Snapshot]:
        if team_a == team_b:
            raise ApiError("same_team", "team_a and team_b are the same team", 400, {"team": team_a})
        snap, stale = self._resolve(season, model, as_of)
        teams = self._teams_table(snap).set_index("team")
        for t in (team_a, team_b):
            if t not in teams.index:
                raise ApiError("unknown_team", f"No team '{t}' in the {season} predictions", 404,
                               {"team": t, "suggestions": difflib.get_close_matches(t, list(teams.index), n=3, cutoff=0.5)})
        lo, hi = sorted((team_a, team_b))
        row, stale_pair = self._pair(snap, lo, hi)
        stale = stale or stale_pair
        if row is None:
            raise ApiError("pair_not_found", "That pair was not published for this snapshot", 404, {"team_a": team_a, "team_b": team_b})
        flip = team_a != lo
        sign = -1.0 if flip else 1.0
        p = float(row["p_a_win"])
        p_a = 1 - p if flip else p
        margin = None if row["margin_a"] is None else sign * float(row["margin_a"])
        score_a, score_b = (row["score_b"], row["score_a"]) if flip else (row["score_a"], row["score_b"])
        va, vb = teams.loc[team_a]["values"] or {}, teams.loc[team_b]["values"] or {}
        factors, interaction = [], None
        if row.get("factors"):
            for g, effect in row["factors"].items():
                effect = sign * float(effect)
                if g == "interaction":
                    interaction = effect
                    continue
                key = HEADLINE.get(g)
                factors.append({"group": g, "effect": _r(effect), "headline": key, "a": _r(va.get(key)), "b": _r(vb.get(key))})
            factors.sort(key=lambda f: -abs(f["effect"] or 0))
        model_row = next((m for m in self._models_list() if m.id == snap.model_id), None)
        notes = ["neutral court"]
        if snap.in_sample:
            notes.append(f"in-sample: this model was trained on {season}, so the probability is not a fair test of the model")
        if stale:
            notes.append("served from memory: the prediction store could not be reached")
        body = {
            "team_a": {"id": team_a, "name": teams.loc[team_a]["name"]},
            "team_b": {"id": team_b, "name": teams.loc[team_b]["name"]},
            "p_a_win": _r(p_a), "p_b_win": _r(1 - p_a),
            "projected": {"score_a": _r(score_a, 1), "score_b": _r(score_b, 1), "margin_a": _r(margin, 1), "total": _r(row["total"], 1), "basis": "40-minute game"},
            "factors": factors[:TOP_FACTORS], "interaction": _r(interaction),
            "model": _model_json(model_row) if model_row else {"id": snap.model_id},
            "snapshot": _snap_json(snap), "notes": notes,
        }
        return body, snap

    # ------------------------------------------------------------------ brackets

    def _bracket_context(self, snap: Snapshot) -> dict:
        key = ("bracket", snap.model_id, snap.season, snap.as_of)
        ctx = self.contexts.get(key)
        if ctx is not None:
            return ctx
        b = self._guard(self._store().bracket, snap.season)
        if b is None:
            raise ApiError("bracket_not_found", f"No bracket field published for {snap.season}", 404, {"season": snap.season})
        field = b["teams"].sort_values(["region", "seed", "team"]).reset_index(drop=True)  # same order from every store
        ff = tuple(tuple(p) for p in b["final_four"])
        try:
            bracket = Bracket.from_teams(field[["team", "seed", "region"]], ff)
        except ValueError as e:
            raise ApiError("bracket_invalid", str(e), 500)
        pairs = self._guard(self._store().pairs_among, snap, list(bracket.teams))
        T = len(bracket.teams)
        if len(pairs) != T * (T - 1) // 2:
            raise ApiError("incomplete_snapshot", f"Expected {T * (T - 1) // 2} pairs for the field, found {len(pairs)}", 503, {"season": snap.season})
        idx = bracket.index
        P = np.full((T, T), 0.5)
        a = pairs["team_a"].map(idx).to_numpy()
        c = pairs["team_b"].map(idx).to_numpy()
        P[a, c] = pairs["p_a_win"].to_numpy(float)
        P[c, a] = 1 - P[a, c]
        # projected points: SC[i, j] is team i's score against team j, MG[i, j] the margin of i over j (NaN if the model has no score model)
        SC = np.full((T, T), np.nan)
        MG = np.full((T, T), np.nan)
        SC[a, c] = pairs["score_a"].to_numpy(float)
        SC[c, a] = pairs["score_b"].to_numpy(float)
        MG[a, c] = pairs["margin_a"].to_numpy(float)
        MG[c, a] = -MG[a, c]
        names = dict(zip(field["team"], field["name"].fillna(field["team"])))
        hist = {tuple(int(x) for x in k.split("-")): v for k, v in (snap.meta.get("seed_history") or {}).items()}
        ctx = {"bracket": bracket, "P": P, "SC": SC, "MG": MG, "names": names, "history": hist, "results": self._played(bracket, b.get("results") or [])}
        self.contexts.put(key, ctx)
        return ctx

    @staticmethod
    def _played(bracket: Bracket, raw: list) -> list:
        """Stored results with the round and game position added, so a page can put each one on the bracket. Round 0 is the
        First Four. Anything that names a team outside the bracket, or a winner who did not play, is dropped."""
        idx = bracket.index
        out = []
        for r in raw:
            try:
                ta, tb, w = r["team_a"], r["team_b"], r["winner"]
            except (KeyError, TypeError):
                continue
            if ta not in idx or tb not in idx or ta == tb or w not in (ta, tb):
                continue
            a, b = idx[ta], idx[tb]
            rnd = Service._lca_round(bracket, a, b)
            slot = next(k for k, s in enumerate(bracket.slots) if a in s)
            out.append({"round": rnd, "round_name": PLAYED_ROUND_NAMES.get(rnd, f"round {rnd}"), "game": slot >> rnd, "team_a": ta, "team_b": tb, "winner": w})
        return sorted(out, key=lambda x: (x["round"], x["game"]))

    @staticmethod
    def _lca_round(bracket: Bracket, a: int, b: int) -> int:
        ka = next(k for k, s in enumerate(bracket.slots) if a in s)
        kb = next(k for k, s in enumerate(bracket.slots) if b in s)
        r = 0
        while ka != kb:
            ka, kb, r = ka >> 1, kb >> 1, r + 1
        return r

    def _validate_fixed(self, bracket: Bracket, results: list) -> list:
        idx = bracket.index
        clean = []
        for i, r in enumerate(results):
            if not isinstance(r, dict) or not {"team_a", "team_b", "winner"} <= set(r):
                raise ApiError("bad_result", "each result needs team_a, team_b and winner", 422, {"index": i})
            for k in ("team_a", "team_b", "winner"):
                if r[k] not in idx:
                    raise ApiError("bad_result", f"{k} '{str(r[k])[:60]}' is not in this bracket", 422, {"index": i, "team": str(r[k])[:60]})
            if r["winner"] not in (r["team_a"], r["team_b"]) or r["team_a"] == r["team_b"]:
                raise ApiError("bad_result", "winner must be team_a or team_b, and the two teams must differ", 422, {"index": i})
            clean.append((r["team_a"], r["team_b"], r["winner"]))
        if len({frozenset(c[:2]) for c in clean}) != len(clean):
            raise ApiError("bad_result", "the same game appears twice", 422)
        return clean

    def _check_possible(self, bracket: Bracket, levels, fixed: list) -> None:
        idx = bracket.index
        for i, (ta, tb, _) in enumerate(fixed):
            a, b = idx[ta], idx[tb]
            r = self._lca_round(bracket, a, b)
            if r == 0:
                continue
            j = next(k for k, s in enumerate(bracket.slots) if a in s) >> r
            left, right = levels.arrays[r - 1][2 * j], levels.arrays[r - 1][2 * j + 1]
            meet = left[a] * right[b] + left[b] * right[a]
            if meet <= 1e-12:
                raise ApiError("impossible_result", f"{ta} and {tb} cannot meet in round {r} given the other results", 422, {"index": i, "round": r})

    def bracket(self, season: int, model: str | None = None, scoring: tuple = DEFAULT_SCORING, results: list | None = None) -> tuple[dict, Snapshot]:
        snap, stale = self._resolve(season, model, None, "selection_sunday")
        ctx = self._bracket_context(snap)
        bracket, P = ctx["bracket"], ctx["P"]
        fixed = self._validate_fixed(bracket, results) if results else []
        key = ("odds", snap.model_id, snap.season, snap.as_of, scoring, tuple(sorted(fixed)))
        hit = self.results.get(key)
        if hit is not None:
            return (dict(hit, notes=hit["notes"] + ["served from memory: the prediction store could not be reached"]) if stale else hit), snap
        levels = exact_levels(bracket, P, fixed or None)
        if fixed:
            self._check_possible(bracket, levels, fixed)
        sc = pick.Scoring(scoring)
        picks = {"best_expected_points": pick.best_expected_bracket(bracket, levels, sc), "model_favorite": pick.favorite_bracket(bracket, P),
                 "seed_chalk": pick.seed_bracket(bracket)}
        names = ctx["names"]
        table = reach_table(bracket, levels).drop(columns=["se_champion"])
        table.insert(1, "name", table["team"].map(names))
        field = json.loads(table.round(4).to_json(orient="records"))
        out_picks = {}
        for k, pk in picks.items():
            rows = []
            for r in range(1, len(pk.winners)):
                prev = pk.winners[r - 1]
                for j, w in enumerate(pk.winners[r]):
                    a, b = prev[2 * j], prev[2 * j + 1]
                    rows.append({"round": r, "round_name": ROUND_NAMES.get(r, f"round {r}"), "game": j, "team_a": bracket.teams[a], "team_b": bracket.teams[b],
                                 "pick": bracket.teams[w], "p_pick_wins_game": _r(levels.arrays[r][j, w]), "p_a_beats_b": _r(P[a, b]),
                                 "score_a": _r(ctx["SC"][a, b], 1), "score_b": _r(ctx["SC"][b, a], 1), "margin_a": _r(ctx["MG"][a, b], 1)})
            out_picks[k] = rows
        first = upsets.first_round_table(bracket, exact_levels(bracket, P, fixed or None), ctx["history"]) if bracket.n_rounds == 6 else None
        first_json = []
        if first is not None:
            for r in first.itertuples():
                first_json.append({"region": r.region, "favorite": r.favorite, "underdog": r.underdog, "favorite_seed": int(r.favorite_seed), "underdog_seed": int(r.underdog_seed),
                                   "p_underdog_model": _r(r.p_underdog_model), "p_underdog_history": _r(r.p_underdog_history),
                                   "model_picks_upset": bool(r.model_picks_upset), "model_above_history": bool(r.model_above_history), "flagged": bool(r.flagged)})
        notes = ["neutral court", "independent games: team-strength uncertainty is not modeled"]
        if snap.in_sample:
            notes.append(f"in-sample: this model was trained on {season}")
        if fixed:
            notes.append(f"{len(fixed)} games locked as played")
        if stale:
            notes.append("served from memory: the prediction store could not be reached")
        body = {"season": season, "snapshot": _snap_json(snap), "scoring": list(scoring), "locked_results": len(fixed), "field": field,
                "picks": out_picks, "expected_points": {k: _r(pick.expected_points(pk, levels, sc), 1) for k, pk in picks.items()},
                "first_round": first_json, "results": ctx["results"], "notes": notes}
        if not stale:
            self.results.put(key, body)
        return body, snap


def _neg(text: str):
    """Sort key that puts later strings first."""
    return tuple(-ord(c) for c in (text or ""))
