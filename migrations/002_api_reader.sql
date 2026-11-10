-- Phase 4: a read-only role for the web service.
--
-- STEP 1 (you run this yourself, with a password only you know; do not paste the password anywhere else):
--
--     CREATE ROLE api_reader WITH LOGIN PASSWORD '<a long random password>';
--
-- Then set the Render service's DATABASE_URL to a connection string for api_reader. On Supabase's pooler the user
-- name is usually api_reader.<project-ref>; check the pooler string in Project Settings, Database.
--
-- STEP 2 (below): grants and policies. Safe to run again. It does nothing if api_reader does not exist yet.

DO $$
DECLARE t text;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api_reader') THEN
        RAISE NOTICE 'api_reader does not exist yet; create it first (see the top of this file)';
        RETURN;
    END IF;

    GRANT USAGE ON SCHEMA predictions TO api_reader;
    GRANT SELECT ON ALL TABLES IN SCHEMA predictions TO api_reader;

    FOREACH t IN ARRAY ARRAY['model_versions', 'snapshots', 'pair_predictions', 'team_snapshots', 'brackets', 'bracket_teams'] LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'predictions' AND tablename = t AND policyname = 'api_reader_select') THEN
            EXECUTE format('CREATE POLICY api_reader_select ON predictions.%I FOR SELECT TO api_reader USING (true)', t);
        END IF;
    END LOOP;

    -- The existing /teams routes read these scraper tables. They have row level security on, so the role needs a
    -- read-only policy as well as the grant. This ADDS a select policy for api_reader to six scraper tables; it
    -- changes nothing for any other role.
    GRANT USAGE ON SCHEMA public TO api_reader;
    FOREACH t IN ARRAY ARRAY['schools', 'school_seasons', 'team_seasons', 'team_season_stats', 'game_teams', 'games'] LOOP
        EXECUTE format('GRANT SELECT ON public.%I TO api_reader', t);
        IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = t AND policyname = 'api_reader_select') THEN
            EXECUTE format('CREATE POLICY api_reader_select ON public.%I FOR SELECT TO api_reader USING (true)', t);
        END IF;
    END LOOP;
END $$;
