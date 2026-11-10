"""Check PostgresStore against the real database without keeping anything.

    python scripts/check_store_postgres.py

Inside one transaction that is always rolled back: apply 001_predictions.sql, write a small invented snapshot through
PostgresStore, read it back through the same store, and compare every answer with FileStore on the same data. Then
roll back. Uses DATABASE_URL. Nothing is committed.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from api.sample import sample_bundle  # noqa: E402
from api.store import FileStore, PostgresStore  # noqa: E402
from data.db import get_pool  # noqa: E402


class Rollback(Exception):
    pass


def main() -> None:
    bundle = sample_bundle(season=2030, model_id="check-model", extra=2)
    bundle.bracket["teams"].loc[bundle.bracket["teams"]["team"] == "e1", "name"] = np.nan  # a team with no display name
    bundle.teams.loc[bundle.teams["team"] == "e1", "name"] = np.nan
    results = []
    pool = get_pool()
    with pool.connection() as conn:
        existed = bool(conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'predictions'").fetchone())
        before = conn.execute("SELECT count(*) AS n FROM predictions.model_versions").fetchone()["n"] if existed else 0
    with tempfile.TemporaryDirectory() as d:
        files = FileStore(d)
        files.write_snapshot(bundle, activate=True)
        fsnap = files.snapshots(2030)[0]
        with pool.connection() as conn:
            conn.autocommit = False
            try:
                with conn.transaction():
                    conn.execute((ROOT / "migrations" / "001_predictions.sql").read_text(encoding="utf-8"))  # harmless if already applied
                    if existed:  # the real tables are in use, so give the check its own model id and keep real rows out of its way
                        conn.execute("UPDATE predictions.model_versions SET active = false WHERE active")
                    pg = PostgresStore(conn)
                    info = pg.write_snapshot(bundle, activate=True)
                    results.append(("write snapshot", f"ok ({info['pairs']} pairs)"))
                    try:
                        pg.write_snapshot(bundle)
                        results.append(("second write refused", "FAILED: no error"))
                    except FileExistsError:
                        results.append(("second write refused", "ok"))
                    with conn.transaction():
                        pg.write_snapshot(sample_bundle(season=2030, model_id="check-model", extra=2, seed=3), replace=True)
                    pg.write_snapshot(bundle, replace=True)
                    results.append(("replace works", "ok"))

                    psnap = pg.snapshots(2030)[0]
                    results.append(("snapshot metadata equal", "ok" if (psnap.model_id, psnap.season, psnap.as_of, psnap.in_sample, psnap.n_teams) == (fsnap.model_id, fsnap.season, fsnap.as_of, fsnap.in_sample, fsnap.n_teams) else "FAILED"))
                    results.append(("active model", "ok" if pg.active_model() and pg.active_model().id == "check-model" else "FAILED"))
                    pr, fr = pg.pair(psnap, "e1", "w1"), files.pair(fsnap, "e1", "w1")
                    same = all(abs(pr[k] - fr[k]) < 1e-4 for k in ("p_a_win", "margin_a", "total", "score_a", "score_b"))
                    same = same and set(pr["factors"]) == set(fr["factors"]) and all(abs(pr["factors"][k] - fr["factors"][k]) < 1e-9 for k in pr["factors"])
                    results.append(("one pair equals the file store's", "ok" if same else f"FAILED: {pr} vs {fr}"))
                    field = list(bundle.bracket["teams"]["team"])
                    a, b = pg.pairs_among(psnap, field), files.pairs_among(fsnap, field)
                    merged = a.merge(b, on=["team_a", "team_b"], suffixes=("_pg", "_file"))
                    results.append(("pairs among the field equal", "ok" if len(a) == len(b) == len(merged) and np.allclose(merged.p_a_win_pg, merged.p_a_win_file, atol=1e-4) else "FAILED"))
                    pt, ft = pg.teams(psnap).sort_values("team").reset_index(drop=True), files.teams(fsnap).sort_values("team").reset_index(drop=True)
                    results.append(("team table equal", "ok" if list(pt["team"]) == list(ft["team"]) and pt["values"].iloc[0] == ft["values"].iloc[0] else "FAILED"))
                    pb, fb = pg.bracket(2030), files.bracket(2030)
                    results.append(("bracket equal", "ok" if len(pb["teams"]) == len(fb["teams"]) == 68 and [list(x) for x in pb["final_four"]] == fb["final_four"] else "FAILED"))
                    miss = pb["teams"][pb["teams"]["team"] == "e1"]["name"].iloc[0]
                    results.append(("a missing name stays null, not the text NaN", "ok" if miss is None or (miss != miss) else f"FAILED: {miss!r}"))
                    results.append(("unknown pair is None", "ok" if pg.pair(psnap, "e1", "nobody") is None else "FAILED"))
                    results.append(("ping", "ok" if pg.ping() else "FAILED"))
                    raise Rollback()
            except Rollback:
                pass
            except Exception as e:
                results.append(("unexpected error", f"FAILED: {type(e).__name__}: {e}"))
    with pool.connection() as conn:
        left = conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'predictions'").fetchone()
        if existed:
            after = conn.execute("SELECT count(*) AS n FROM predictions.model_versions").fetchone()["n"]
            active = conn.execute("SELECT count(*) AS n FROM predictions.model_versions WHERE active").fetchone()["n"]
            stray = conn.execute("SELECT count(*) AS n FROM predictions.model_versions WHERE id = 'check-model'").fetchone()["n"]
            results.append(("after rollback: real model rows unchanged", "ok" if after == before else f"FAILED: {before} before, {after} after"))
            results.append(("after rollback: the active model flag is as before", "ok" if active == 1 else f"FAILED: {active} active models"))
            results.append(("after rollback: no test rows left", "ok" if stray == 0 else "FAILED: test rows remain"))
        else:
            results.append(("after rollback: schema predictions", "gone" if not left else "STILL THERE"))
    width = max(len(a) for a, _ in results)
    for a, b in results:
        print(f"{a:<{width}}  {b}")
    pool.close()
    sys.exit(1 if any(str(b).startswith(("FAILED", "STILL")) for _, b in results) else 0)


if __name__ == "__main__":
    main()
