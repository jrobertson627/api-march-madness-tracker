"""Where published predictions live, behind one interface.

`FileStore` keeps them in a folder (for tests, local work and a first publish). `PostgresStore` keeps them in the
`predictions` schema (see migrations/001_predictions.sql). Reads are what the web API uses. `write_snapshot` is what
the publisher uses. Snapshots are immutable: writing one that exists is an error unless `replace=True`.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

PAIR_COLUMNS = ["team_a", "team_b", "p_a_win", "margin_a", "total", "score_a", "score_b", "factors"]


@dataclass
class ModelVersion:
    id: str
    name: str
    run_id: str
    kind: str                       # live | backtest
    training_seasons: list
    data_hash: str
    created_at: str = ""
    metrics: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    active: bool = False


@dataclass
class Snapshot:
    model_id: str
    season: int
    as_of: str                      # YYYY-MM-DD
    kind: str                       # selection_sunday | latest
    in_sample: bool
    n_teams: int
    meta: dict = field(default_factory=dict)
    created_at: str = ""


@dataclass
class Bundle:
    """Everything one publish writes."""
    model: ModelVersion
    snapshot: Snapshot
    pairs: pd.DataFrame             # PAIR_COLUMNS; team_a sorts before team_b; factors is a dict per row
    teams: pd.DataFrame             # team, name, values (dict)
    bracket: dict | None = None     # {"teams": DataFrame[team, name, seed, region], "final_four": [[r, r], [r, r]]}


def _date(x) -> str:
    return x.isoformat() if hasattr(x, "isoformat") else str(x)


class Store:
    def models(self) -> list[ModelVersion]: raise NotImplementedError
    def snapshots(self, season: int | None = None) -> list[Snapshot]: raise NotImplementedError
    def pair(self, snap: Snapshot, team_a: str, team_b: str) -> dict | None: raise NotImplementedError
    def pairs_among(self, snap: Snapshot, teams: list[str]) -> pd.DataFrame: raise NotImplementedError
    def teams(self, snap: Snapshot) -> pd.DataFrame: raise NotImplementedError
    def bracket(self, season: int) -> dict | None: raise NotImplementedError
    def ping(self) -> bool: raise NotImplementedError
    def write_snapshot(self, bundle: Bundle, replace: bool = False, activate: bool = False) -> dict: raise NotImplementedError

    def active_model(self) -> ModelVersion | None:
        return next((m for m in self.models() if m.active), None)


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #

class FileStore(Store):
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._pairs: dict = {}
        self._teams: dict = {}

    # layout: models.json, brackets/<season>.json, snapshots/<model>__<season>__<as_of>/{snapshot.json,pairs.csv,teams.json}
    def _dir(self, s: Snapshot) -> Path:
        return self.root / "snapshots" / f"{s.model_id}__{s.season}__{s.as_of}"

    def models(self) -> list[ModelVersion]:
        p = self.root / "models.json"
        return [ModelVersion(**m) for m in json.loads(p.read_text(encoding="utf-8"))] if p.exists() else []

    def snapshots(self, season: int | None = None) -> list[Snapshot]:
        out = []
        for f in sorted((self.root / "snapshots").glob("*/snapshot.json")) if (self.root / "snapshots").exists() else []:
            s = Snapshot(**json.loads(f.read_text(encoding="utf-8")))
            if season is None or s.season == season:
                out.append(s)
        return out

    def _load_pairs(self, snap: Snapshot) -> pd.DataFrame:
        key = (snap.model_id, snap.season, snap.as_of)
        if key not in self._pairs:
            df = pd.read_csv(self._dir(snap) / "pairs.csv")
            df["factors"] = df["factors"].map(lambda x: json.loads(x) if isinstance(x, str) and x else None)
            self._pairs[key] = df.set_index(["team_a", "team_b"], drop=False)
        return self._pairs[key]

    def pair(self, snap, team_a, team_b):
        df = self._load_pairs(snap)
        try:
            row = df.loc[(team_a, team_b)]
        except KeyError:
            return None
        return {k: (None if isinstance(row[k], float) and pd.isna(row[k]) else row[k]) for k in PAIR_COLUMNS}

    def pairs_among(self, snap, teams):
        df = self._load_pairs(snap)
        keep = set(teams)
        return df[df["team_a"].isin(keep) & df["team_b"].isin(keep)].reset_index(drop=True)

    def teams(self, snap):
        key = (snap.model_id, snap.season, snap.as_of)
        if key not in self._teams:
            rows = json.loads((self._dir(snap) / "teams.json").read_text(encoding="utf-8"))
            self._teams[key] = pd.DataFrame(rows)
        return self._teams[key]

    def bracket(self, season):
        p = self.root / "brackets" / f"{season}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return {"teams": pd.DataFrame(d["teams"]), "final_four": d["final_four"], "as_of": d.get("as_of")}

    def ping(self):
        return self.root.exists()

    def write_snapshot(self, bundle, replace=False, activate=False):
        s, m = bundle.snapshot, bundle.model
        d = self._dir(s)
        if d.exists() and not replace:
            raise FileExistsError(f"snapshot already published: {d.name} (pass replace=True to overwrite)")
        d.mkdir(parents=True, exist_ok=True)
        pairs = bundle.pairs[PAIR_COLUMNS].copy()
        pairs["factors"] = pairs["factors"].map(lambda x: json.dumps(x) if isinstance(x, dict) else "")
        pairs.to_csv(d / "pairs.csv", index=False)
        (d / "teams.json").write_text(json.dumps(bundle.teams.to_dict("records"), default=float), encoding="utf-8")
        snap = asdict(s) | {"created_at": s.created_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        (d / "snapshot.json").write_text(json.dumps(snap), encoding="utf-8")
        models = {x.id: x for x in self.models()}
        if m.id not in models:
            models[m.id] = ModelVersion(**(asdict(m) | {"created_at": m.created_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}))
        if activate:
            for x in models.values():
                x.active = x.id == m.id
        (self.root / "models.json").write_text(json.dumps([asdict(x) for x in models.values()], default=str, indent=1), encoding="utf-8")
        if bundle.bracket:
            (self.root / "brackets").mkdir(exist_ok=True)
            b = bundle.bracket
            (self.root / "brackets" / f"{s.season}.json").write_text(
                json.dumps({"as_of": s.as_of, "final_four": b["final_four"], "teams": b["teams"].to_dict("records")}, default=float), encoding="utf-8")
        self._pairs.pop((s.model_id, s.season, s.as_of), None)
        self._teams.pop((s.model_id, s.season, s.as_of), None)
        return {"pairs": len(pairs), "teams": len(bundle.teams), "dir": str(d)}


# --------------------------------------------------------------------------- #
# postgres
# --------------------------------------------------------------------------- #

class PostgresStore(Store):
    """Reads through the pool in data.db (DATABASE_URL, ideally the read-only role). Writes use `conn` if given
    (so a test can run inside a transaction it rolls back), otherwise a new connection from PUBLISH_DATABASE_URL."""

    def __init__(self, conn=None):
        self._conn = conn

    def _q(self, sql, params=None):
        if self._conn is not None:
            return self._conn.execute(sql, params).fetchall()
        from data.db import query
        return query(sql, params)

    @staticmethod
    def _snap(r) -> Snapshot:
        return Snapshot(r["model_id"], r["season"], _date(r["as_of"]), r["kind"], r["in_sample"], r["n_teams"], r["meta"] or {}, _date(r["created_at"]))

    def models(self):
        rows = self._q("SELECT * FROM predictions.model_versions ORDER BY created_at")
        return [ModelVersion(r["id"], r["name"], r["run_id"], r["kind"], list(r["training_seasons"]), r["data_hash"], _date(r["created_at"]),
                             r["metrics"] or {}, r["manifest"] or {}, r["active"]) for r in rows]

    def snapshots(self, season=None):
        sql = "SELECT * FROM predictions.snapshots" + (" WHERE season = %s" if season is not None else "") + " ORDER BY created_at"
        return [self._snap(r) for r in self._q(sql, (season,) if season is not None else None)]

    def pair(self, snap, team_a, team_b):
        rows = self._q("SELECT team_a, team_b, p_a_win, margin_a, total, score_a, score_b, factors FROM predictions.pair_predictions "
                       "WHERE model_id = %s AND season = %s AND as_of = %s AND team_a = %s AND team_b = %s",
                       (snap.model_id, snap.season, snap.as_of, team_a, team_b))
        return dict(rows[0]) if rows else None

    def pairs_among(self, snap, teams):
        rows = self._q("SELECT team_a, team_b, p_a_win, margin_a, total, score_a, score_b, factors FROM predictions.pair_predictions "
                       "WHERE model_id = %s AND season = %s AND as_of = %s AND team_a = ANY(%s) AND team_b = ANY(%s)",
                       (snap.model_id, snap.season, snap.as_of, list(teams), list(teams)))
        return pd.DataFrame(rows, columns=PAIR_COLUMNS)

    def teams(self, snap):
        rows = self._q('SELECT team, name, "values" FROM predictions.team_snapshots WHERE model_id = %s AND season = %s AND as_of = %s',
                       (snap.model_id, snap.season, snap.as_of))
        return pd.DataFrame(rows, columns=["team", "name", "values"])

    def bracket(self, season):
        b = self._q("SELECT as_of, final_four FROM predictions.brackets WHERE season = %s", (season,))
        if not b:
            return None
        t = self._q("SELECT team, name, seed, region FROM predictions.bracket_teams WHERE season = %s ORDER BY region, seed, team", (season,))
        return {"teams": pd.DataFrame(t, columns=["team", "name", "seed", "region"]), "final_four": b[0]["final_four"], "as_of": _date(b[0]["as_of"])}

    def ping(self):
        try:
            self._q("SELECT 1 AS ok")
            return True
        except Exception:
            return False

    def write_snapshot(self, bundle, replace=False, activate=False):
        import psycopg
        from psycopg.types.json import Jsonb

        own = self._conn is None
        conn = self._conn
        if own:
            import os
            from data import db  # noqa: F401  (loads .env)
            url = os.environ.get("PUBLISH_DATABASE_URL")
            if not url:
                url = os.environ.get("DATABASE_URL")
                if not url:
                    raise RuntimeError("Neither PUBLISH_DATABASE_URL nor DATABASE_URL is set")
                print("note: PUBLISH_DATABASE_URL is not set, writing with DATABASE_URL (use a separate write credential in production)", file=__import__("sys").stderr)
            conn = psycopg.connect(url, prepare_threshold=None, row_factory=psycopg.rows.dict_row)
        try:
            with conn.transaction():
                s, m = bundle.snapshot, bundle.model
                key = (s.model_id, s.season, s.as_of)
                exists = conn.execute("SELECT 1 FROM predictions.snapshots WHERE model_id = %s AND season = %s AND as_of = %s", key).fetchone()
                if exists and not replace:
                    raise FileExistsError(f"snapshot already published: {key}")
                if exists:
                    for t in ("pair_predictions", "team_snapshots", "snapshots"):
                        conn.execute(f"DELETE FROM predictions.{t} WHERE model_id = %s AND season = %s AND as_of = %s", key)
                conn.execute(
                    "INSERT INTO predictions.model_versions (id, name, run_id, kind, training_seasons, data_hash, metrics, manifest) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (m.id, m.name, m.run_id, m.kind, list(m.training_seasons), m.data_hash, Jsonb(m.metrics), Jsonb(m.manifest)))
                conn.execute("INSERT INTO predictions.snapshots (model_id, season, as_of, kind, in_sample, n_teams, meta) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                             (s.model_id, s.season, s.as_of, s.kind, s.in_sample, s.n_teams, Jsonb(s.meta)))
                with conn.cursor() as cur, cur.copy(
                    "COPY predictions.pair_predictions (model_id, season, as_of, team_a, team_b, p_a_win, margin_a, total, score_a, score_b, factors) FROM STDIN"
                ) as cp:
                    for r in bundle.pairs.itertuples(index=False):
                        f = r.factors
                        cp.write_row((s.model_id, s.season, s.as_of, r.team_a, r.team_b, r.p_a_win, _nan(r.margin_a), _nan(r.total), _nan(r.score_a), _nan(r.score_b),
                                      json.dumps(f) if isinstance(f, dict) else None))
                for r in bundle.teams.itertuples(index=False):
                    conn.execute('INSERT INTO predictions.team_snapshots (model_id, season, as_of, team, name, "values") VALUES (%s, %s, %s, %s, %s, %s)',
                                 (s.model_id, s.season, s.as_of, r.team, _text(r.name), Jsonb(r.values)))
                if bundle.bracket:
                    conn.execute("DELETE FROM predictions.bracket_teams WHERE season = %s", (s.season,))
                    conn.execute("DELETE FROM predictions.brackets WHERE season = %s", (s.season,))
                    conn.execute("INSERT INTO predictions.brackets (season, as_of, final_four) VALUES (%s, %s, %s)", (s.season, s.as_of, Jsonb(bundle.bracket["final_four"])))
                    for r in bundle.bracket["teams"].itertuples(index=False):
                        conn.execute("INSERT INTO predictions.bracket_teams (season, team, name, seed, region) VALUES (%s, %s, %s, %s, %s)",
                                     (s.season, r.team, _text(r.name), int(r.seed), r.region))
                if activate:
                    conn.execute("UPDATE predictions.model_versions SET active = false WHERE active AND id <> %s", (m.id,))
                    conn.execute("UPDATE predictions.model_versions SET active = true WHERE id = %s", (m.id,))
        finally:
            if own:
                conn.close()
        return {"pairs": len(bundle.pairs), "teams": len(bundle.teams)}


def _text(x):
    """A string, or None for a missing name (pandas turns None into NaN, which would be stored as the text 'NaN')."""
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def _nan(x):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else float(x)
