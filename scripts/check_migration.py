"""Test the Phase 4 and 5 migrations (001, 002, 003) against the real database without keeping any change.

    python scripts/check_migration.py

Everything runs inside one transaction that is always rolled back (Postgres can roll back schema changes). The
script creates the tables, checks that they and their constraints behave, creates a throwaway api_reader role if none
exists, applies 002_api_reader.sql, checks the role can read but not write, then rolls back and confirms the schema
is gone. Nothing is committed. Uses PUBLISH_DATABASE_URL if set (it must be able to create objects), otherwise DATABASE_URL.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from data.db import get_pool, use_write_credential  # noqa: E402

MIGRATIONS = ROOT / "migrations"
INS_PAIR = "INSERT INTO predictions.pair_predictions (model_id, season, as_of, team_a, team_b, p_a_win) VALUES (%s, 2026, '2026-03-15', %s, %s, %s)"


class Rollback(Exception):
    pass


def expect_error(conn, sql, params, exc, label, results):
    """Run a statement that must fail, inside a savepoint so the transaction survives."""
    try:
        with conn.transaction():
            conn.execute(sql, params)
        results.append((label, "FAILED: no error raised"))
    except exc:
        results.append((label, "ok"))
    except Exception as e:
        results.append((label, f"FAILED: {type(e).__name__}: {e}"))


def checks(conn, results, existed):
    results.append(("schema predictions absent before", "ok" if not existed else "NOTE: it already exists; the migration is idempotent"))
    sql1 = (MIGRATIONS / "001_predictions.sql").read_text(encoding="utf-8")
    conn.execute(sql1)
    tables = {r["table_name"] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'predictions'").fetchall()}
    want = {"model_versions", "snapshots", "pair_predictions", "team_snapshots", "brackets", "bracket_teams"}
    results.append(("001 creates the six tables", "ok" if want <= tables else f"FAILED: missing {sorted(want - tables)}"))
    conn.execute(sql1)
    results.append(("001 runs twice without error", "ok"))

    sql3 = (MIGRATIONS / "003_results_and_backtest.sql").read_text(encoding="utf-8")
    conn.execute(sql3)
    conn.execute(sql3)
    results.append(("003 applies and runs twice without error", "ok"))
    col = conn.execute("SELECT data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'predictions' AND table_name = 'brackets' AND column_name = 'results'").fetchone()
    results.append(("003 adds brackets.results (jsonb, not null, default [])", "ok" if col and col["data_type"] == "jsonb" and col["is_nullable"] == "NO" and "[]" in col["column_default"] else f"FAILED: {col}"))
    rls = conn.execute("SELECT relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'predictions' AND c.relname = 'backtest_reports'").fetchone()
    results.append(("003 turns row security on for backtest_reports", "ok" if rls and rls["relrowsecurity"] else "FAILED"))
    conn.execute("DELETE FROM predictions.bracket_teams WHERE season = 2099")
    conn.execute("DELETE FROM predictions.brackets WHERE season = 2099")
    conn.execute("INSERT INTO predictions.brackets (season, as_of, final_four) VALUES (2099, '2099-03-15', '[]')")
    default = conn.execute("SELECT results FROM predictions.brackets WHERE season = 2099").fetchone()["results"]
    results.append(("an old-style bracket row gets results = []", "ok" if default == [] else f"FAILED: {default!r}"))

    conn.execute("INSERT INTO predictions.model_versions (id, name, run_id, kind, training_seasons, data_hash) VALUES ('t1', 'test', 'r', 'live', '{2022}', 'h')")
    conn.execute("INSERT INTO predictions.snapshots (model_id, season, as_of, kind, in_sample, n_teams) VALUES ('t1', 2026, '2026-03-15', 'selection_sunday', false, 68)")
    conn.execute(INS_PAIR, ("t1", "duke", "houston", 0.6))
    results.append(("valid rows insert", "ok"))
    expect_error(conn, INS_PAIR, ("t1", "a", "b", 1.5), psycopg.errors.CheckViolation, "probability above 1 rejected", results)
    expect_error(conn, INS_PAIR, ("t1", "zzz", "aaa", 0.5), psycopg.errors.CheckViolation, "team_a after team_b rejected", results)
    expect_error(conn, INS_PAIR, ("t1", "duke", "houston", 0.5), psycopg.errors.UniqueViolation, "duplicate pair rejected", results)
    expect_error(conn, INS_PAIR, ("nope", "a", "b", 0.5), psycopg.errors.ForeignKeyViolation, "pair without a snapshot rejected", results)
    conn.execute("INSERT INTO predictions.backtest_reports (model_id, data_hash, report) VALUES ('t1', 'h', '{\"limits\": []}')")
    results.append(("a valid backtest report inserts", "ok"))
    expect_error(conn, "INSERT INTO predictions.backtest_reports (model_id, data_hash, report) VALUES ('nope', 'h', '{}')", None, psycopg.errors.ForeignKeyViolation,
                 "a backtest report for an unknown model rejected", results)
    expect_error(conn, "INSERT INTO predictions.backtest_reports (model_id, data_hash, report) VALUES ('t1', 'h', NULL)", None, psycopg.errors.NotNullViolation,
                 "a backtest report without content rejected", results)
    if existed:  # the real tables are in use: step the real active model aside (this is rolled back)
        conn.execute("UPDATE predictions.model_versions SET active = false WHERE active")
    conn.execute("UPDATE predictions.model_versions SET active = true WHERE id = 't1'")
    conn.execute("INSERT INTO predictions.model_versions (id, name, run_id, kind, training_seasons, data_hash) VALUES ('t2', 'test2', 'r', 'live', '{2022}', 'h')")
    expect_error(conn, "UPDATE predictions.model_versions SET active = true WHERE id = 't2'", None, psycopg.errors.UniqueViolation, "a second active model rejected", results)

    made_role = False
    if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'api_reader'").fetchone():
        try:
            with conn.transaction():
                conn.execute("CREATE ROLE api_reader NOLOGIN")
            made_role = True
        except Exception as e:
            results.append(("create a throwaway api_reader", f"skipped: {type(e).__name__}"))
    if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'api_reader'").fetchone():
        results.append(("002 and read-only checks", "skipped: no api_reader role and none could be created"))
        return
    owner_schools = conn.execute("SELECT count(*) AS n FROM public.schools").fetchone()["n"]
    owner_pairs = conn.execute("SELECT count(*) AS n FROM predictions.pair_predictions").fetchone()["n"]
    conn.execute((MIGRATIONS / "002_api_reader.sql").read_text(encoding="utf-8"))
    results.append(("002 applies" + (" (with a throwaway role)" if made_role else ""), "ok"))
    conn.execute((MIGRATIONS / "002_api_reader.sql").read_text(encoding="utf-8"))
    results.append(("002 runs twice without error", "ok"))
    owner_reports = conn.execute("SELECT count(*) AS n FROM predictions.backtest_reports").fetchone()["n"]
    conn.execute((MIGRATIONS / "003_results_and_backtest.sql").read_text(encoding="utf-8"))  # grants the new table to the role that now exists
    results.append(("003 runs again after 002 (grants the new table)", "ok"))

    try:  # the creating role cannot switch to a role it just made unless it is granted SET
        with conn.transaction():
            conn.execute("GRANT api_reader TO CURRENT_USER WITH SET TRUE")
        can_switch = True
    except Exception:
        can_switch = False
    try:
        with conn.transaction():
            conn.execute("SET LOCAL ROLE api_reader")
            n = conn.execute("SELECT count(*) AS n FROM predictions.pair_predictions").fetchone()["n"]
            n2 = conn.execute("SELECT count(*) AS n FROM public.schools").fetchone()["n"]
            n3 = conn.execute("SELECT count(*) AS n FROM predictions.backtest_reports").fetchone()["n"]
            n4 = conn.execute("SELECT results FROM predictions.brackets WHERE season = 2099").fetchone()["results"]
        results.append(("api_reader can read backtest_reports and brackets.results", "ok" if n3 == owner_reports and n3 > 0 and n4 == [] else f"FAILED: saw {n3} of {owner_reports} reports, results {n4!r}"))
        results.append(("api_reader can read the predictions tables", "ok" if n == owner_pairs else f"FAILED: saw {n} of {owner_pairs} rows"))
        results.append(("api_reader sees the same schools rows as the owner", "ok" if n2 == owner_schools and owner_schools > 0 else f"FAILED: {n2} against {owner_schools}"))
    except psycopg.errors.InsufficientPrivilege:
        results.append(("api_reader read checks", "skipped: this connection may not switch to the role"))
        return
    expect_error(conn, "SET LOCAL ROLE api_reader; INSERT INTO predictions.model_versions (id, name, run_id, kind, training_seasons, data_hash) VALUES ('x', 'x', 'x', 'live', '{1}', 'x')", None, psycopg.errors.InsufficientPrivilege, "api_reader cannot write", results)
    expect_error(conn, "SET LOCAL ROLE api_reader; DELETE FROM predictions.snapshots", None, psycopg.errors.InsufficientPrivilege, "api_reader cannot delete", results)
    expect_error(conn, "SET LOCAL ROLE api_reader; INSERT INTO predictions.backtest_reports (model_id, data_hash, report) VALUES ('t1', 'h', '{}')", None,
                 psycopg.errors.InsufficientPrivilege, "api_reader cannot write backtest_reports", results)
    expect_error(conn, "SET LOCAL ROLE api_reader; UPDATE predictions.brackets SET results = '[1]'", None, psycopg.errors.InsufficientPrivilege, "api_reader cannot change brackets.results", results)
    expect_error(conn, "SET LOCAL ROLE api_reader; UPDATE public.schools SET display_name = display_name", None, psycopg.errors.InsufficientPrivilege, "api_reader cannot change scraper tables", results)
    return can_switch


def main() -> None:
    results = []
    if use_write_credential():
        print("using PUBLISH_DATABASE_URL (a write credential)")
    pool = get_pool()
    with pool.connection() as conn:
        existed = bool(conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'predictions'").fetchone())
        real_before = conn.execute("SELECT count(*) AS n FROM predictions.model_versions").fetchone()["n"] if existed else 0
    with pool.connection() as conn:
        conn.autocommit = False
        try:
            with conn.transaction():
                try:
                    checks(conn, results, existed)
                except Exception as e:
                    results.append(("unexpected error", f"FAILED: {type(e).__name__}: {e}"))
                raise Rollback()  # always undo everything
        except Rollback:
            pass
    with pool.connection() as conn:
        left = conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'predictions'").fetchone()
        role = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'api_reader'").fetchone()
        if existed:
            after = conn.execute("SELECT count(*) AS n FROM predictions.model_versions").fetchone()["n"]
            active = conn.execute("SELECT count(*) AS n FROM predictions.model_versions WHERE active").fetchone()["n"]
            stray = conn.execute("SELECT count(*) AS n FROM predictions.model_versions WHERE id IN ('t1', 't2')").fetchone()["n"]
            results.append(("after rollback: real model rows unchanged", "ok" if after == real_before else f"FAILED: {real_before} before, {after} after"))
            results.append(("after rollback: one active model, as before", "ok" if active == 1 else f"FAILED: {active} active models"))
            results.append(("after rollback: no test rows left", "ok" if stray == 0 else "FAILED: test rows remain"))
        else:
            results.append(("after rollback: schema predictions", "gone" if not left else "STILL THERE"))
        results.append(("after rollback: role api_reader", "not present" if not role else "present (it existed before, or the throwaway role was not rolled back)"))
    width = max(len(a) for a, _ in results)
    for a, b in results:
        print(f"{a:<{width}}  {b}")
    pool.close()
    sys.exit(1 if any(str(b).startswith(("FAILED", "STILL")) for _, b in results) else 0)


if __name__ == "__main__":
    main()
