"""Postgres connection pool for the cbb-web-scraper Supabase database.

Configure with the DATABASE_URL environment variable (a standard Postgres
connection string from Supabase: Project Settings -> Database -> Connection
string). Every table in that database has RLS enabled, so the Supabase REST
API with the anon key would return nothing; this talks to Postgres directly.

Use the Session pooler string (port 5432) on hosts without IPv6 such as
Render's free tier, since the direct db.<ref>.supabase.co host is IPv6-only.
"""

import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

try:  # optional: load a local .env during development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_pool = None


def get_pool():
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(
            url,
            min_size=1,
            max_size=5,
            # prepare_threshold=None keeps this working behind Supabase's
            # transaction pooler (port 6543), which can't do prepared statements.
            kwargs={"row_factory": dict_row, "prepare_threshold": None},
            open=True,
        )
    return _pool


def query(sql, params=None):
    with get_pool().connection() as conn:
        return conn.execute(sql, params).fetchall()
