"""Apply one migration file for real, in a single transaction, and verify it.

    python scripts/apply_migration.py migrations/001_predictions.sql

Uses DATABASE_URL (a role that can create schemas). The file is applied in one transaction, so it either all
happens or none of it does. Afterwards it prints the tables in the predictions schema and confirms that no table in
any other schema was added, removed or had its row security setting changed. Test a migration first with
scripts/check_migration.py, which always rolls back.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.db import get_pool  # noqa: E402

OTHER = """
SELECT n.nspname AS schema, c.relname AS name, c.relkind AS kind, c.relrowsecurity AS rls
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast', 'predictions') AND c.relkind IN ('r', 'v', 'm', 'p')
ORDER BY 1, 2
"""


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/apply_migration.py migrations/001_predictions.sql")
    path = Path(sys.argv[1])
    sql = path.read_text(encoding="utf-8")
    pool = get_pool()
    with pool.connection() as conn:
        before = conn.execute(OTHER).fetchall()
        who = conn.execute("SELECT current_user AS u, current_database() AS db").fetchone()
    print(f"applying {path.name} as {who['u']} to database {who['db']}")
    with pool.connection() as conn:
        conn.autocommit = False
        with conn.transaction():
            conn.execute(sql)
    with pool.connection() as conn:
        after = conn.execute(OTHER).fetchall()
        tables = conn.execute(
            "SELECT c.relname AS table_name, c.relrowsecurity AS rls, (SELECT count(*) FROM pg_attribute a WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped) AS columns "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'predictions' AND c.relkind = 'r' ORDER BY 1"
        ).fetchall()
        counts = {t["table_name"]: conn.execute(f"SELECT count(*) AS n FROM predictions.{t['table_name']}").fetchone()["n"] for t in tables}
    print("tables in schema predictions:")
    for t in tables:
        print(f"  {t['table_name']:<18} columns {t['columns']:>2}   row security {'on' if t['rls'] else 'OFF'}   rows {counts[t['table_name']]}")
    changed = before != after
    print("other schemas unchanged:", "yes" if not changed else "NO, DIFFERENCES FOUND")
    print(f"({len(before)} tables, views and materialized views outside predictions, before and after)")
    pool.close()
    sys.exit(1 if changed or any(not t["rls"] for t in tables) else 0)


if __name__ == "__main__":
    main()
