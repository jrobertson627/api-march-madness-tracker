-- Phase 5: actual tournament results for past seasons, and the published backtest report.
--
-- Additive only: one new column with a default, one new table. Safe to run again. Apply it BEFORE the code that
-- reads these columns is deployed (the API's bracket route selects brackets.results).
--
--   brackets.results          [{"team_a": "duke", "team_b": "houston", "winner": "duke"}, ...] in slugs; [] if not played
--   backtest_reports          one immutable row per publish; the API serves the newest

ALTER TABLE predictions.brackets ADD COLUMN IF NOT EXISTS results jsonb NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS predictions.backtest_reports (
    id         bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id   text        NOT NULL REFERENCES predictions.model_versions (id),
    data_hash  text        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    report     jsonb       NOT NULL
);

-- Same lock-down as the other tables: row level security on, nothing for the public Supabase roles.
ALTER TABLE predictions.backtest_reports ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON predictions.backtest_reports FROM %I', r);
        END IF;
    END LOOP;

    -- The read-only role gets the new table too (002 only covered the tables that existed when it ran).
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api_reader') THEN
        GRANT SELECT ON predictions.backtest_reports TO api_reader;
        IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'predictions' AND tablename = 'backtest_reports' AND policyname = 'api_reader_select') THEN
            CREATE POLICY api_reader_select ON predictions.backtest_reports FOR SELECT TO api_reader USING (true);
        END IF;
    END IF;
END $$;
