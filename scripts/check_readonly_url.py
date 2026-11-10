"""Check that DATABASE_URL is the read-only connection the web service should use.

    python scripts/check_readonly_url.py

Connects with DATABASE_URL (from the environment, or from .env if the variable is not set), prints which database role
it is, confirms it can read the predictions tables and the scraper tables, then tries a write, an update and a delete
inside a transaction that is always rolled back. It never prints the connection string. Exit code 0 means the
connection is read-only and can read what the API needs.

To test the new role before it goes on Render, set the variable for this one command only:
    PowerShell:  $env:DATABASE_URL = "<the api_reader connection string>"; python scripts/check_readonly_url.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from data.db import get_pool  # noqa: E402


class Rollback(Exception):
    pass


def attempt(conn, label, sql, results, must_fail):
    try:
        with conn.transaction():
            conn.execute(sql)
        results.append((label, "FAILED: the connection could do this" if must_fail else "ok"))
    except psycopg.errors.InsufficientPrivilege:
        results.append((label, "ok" if must_fail else "FAILED: permission denied"))
    except Exception as e:
        results.append((label, f"FAILED: {type(e).__name__}"))


def main() -> None:
    results = []
    pool = get_pool()
    with pool.connection() as conn:
        conn.autocommit = False
        me = conn.execute("SELECT current_user AS u").fetchone()["u"]
        results.append(("connected as role", me))
        try:
            with conn.transaction():
                attempt(conn, "can read predictions.pair_predictions", "SELECT count(*) FROM predictions.pair_predictions", results, False)
                attempt(conn, "can read predictions.model_versions", "SELECT count(*) FROM predictions.model_versions", results, False)
                for t in ("schools", "school_seasons", "team_seasons", "team_season_stats", "game_teams", "games"):
                    attempt(conn, f"can read public.{t}", f"SELECT 1 FROM public.{t} LIMIT 1", results, False)
                rows = {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in ("public.schools", "predictions.snapshots")}
                results.append(("sees rows (not hidden by row security)", "ok" if all(v > 0 for v in rows.values()) else f"FAILED: {rows}"))
                attempt(conn, "cannot insert into predictions", "INSERT INTO predictions.model_versions (id, name, run_id, kind, training_seasons, data_hash) VALUES ('readonly-check', 'x', 'x', 'live', '{1}', 'x')", results, True)
                attempt(conn, "cannot update predictions", "UPDATE predictions.model_versions SET active = active", results, True)
                attempt(conn, "cannot delete from predictions", "DELETE FROM predictions.snapshots WHERE false", results, True)
                attempt(conn, "cannot change scraper tables", "UPDATE public.schools SET display_name = display_name WHERE false", results, True)
                attempt(conn, "cannot create tables", "CREATE TABLE public.readonly_check_table (x int)", results, True)
                raise Rollback()
        except Rollback:
            pass
    pool.close()
    width = max(len(a) for a, _ in results)
    for a, b in results:
        print(f"{a:<{width}}  {b}")
    bad = [a for a, b in results if str(b).startswith("FAILED")]
    if bad:
        print("\nNOT read-only or missing access: " + "; ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
