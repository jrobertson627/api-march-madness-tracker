-- Phase 4: tables for published predictions.
--
-- A new schema, separate from the scraper's tables. Rows are written once by the publisher and never updated, apart
-- from the single `active` model flag. Safe to run twice (IF NOT EXISTS). Nothing here touches existing tables.
-- Test it first with:  python scripts/check_migration.py   (runs inside a transaction that is rolled back)

CREATE SCHEMA IF NOT EXISTS predictions;

CREATE TABLE IF NOT EXISTS predictions.model_versions (
    id               text PRIMARY KEY,
    name             text        NOT NULL,
    run_id           text        NOT NULL,
    kind             text        NOT NULL CHECK (kind IN ('live', 'backtest')),
    training_seasons integer[]   NOT NULL,
    data_hash        text        NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    metrics          jsonb       NOT NULL DEFAULT '{}'::jsonb,
    manifest         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    active           boolean     NOT NULL DEFAULT false
);
-- at most one active model
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_active ON predictions.model_versions ((true)) WHERE active;

CREATE TABLE IF NOT EXISTS predictions.snapshots (
    model_id   text        NOT NULL REFERENCES predictions.model_versions (id),
    season     integer     NOT NULL CHECK (season BETWEEN 1990 AND 2100),
    as_of      date        NOT NULL,
    kind       text        NOT NULL CHECK (kind IN ('selection_sunday', 'latest')),
    in_sample  boolean     NOT NULL,
    n_teams    integer     NOT NULL CHECK (n_teams > 1),
    meta       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, season, as_of)
);

CREATE TABLE IF NOT EXISTS predictions.pair_predictions (
    model_id  text    NOT NULL,
    season    integer NOT NULL,
    as_of     date    NOT NULL,
    team_a    text    NOT NULL,   -- school slug; team_a sorts before team_b
    team_b    text    NOT NULL,
    p_a_win   real    NOT NULL CHECK (p_a_win > 0 AND p_a_win < 1),
    margin_a  real,               -- projected points, team A minus team B
    total     real,               -- projected total for a 40-minute game
    score_a   real,
    score_b   real,
    factors   jsonb,              -- {"efficiency": 0.06, ..., "interaction": 0.01}, effects on team A's win probability
    PRIMARY KEY (model_id, season, as_of, team_a, team_b),
    CHECK (team_a < team_b),
    FOREIGN KEY (model_id, season, as_of) REFERENCES predictions.snapshots (model_id, season, as_of)
);

CREATE TABLE IF NOT EXISTS predictions.team_snapshots (
    model_id text    NOT NULL,
    season   integer NOT NULL,
    as_of    date    NOT NULL,
    team     text    NOT NULL,
    name     text,
    values   jsonb   NOT NULL DEFAULT '{}'::jsonb,   -- headline numbers: adj_em, elo, sos, recent form, ...
    PRIMARY KEY (model_id, season, as_of, team),
    FOREIGN KEY (model_id, season, as_of) REFERENCES predictions.snapshots (model_id, season, as_of)
);

CREATE TABLE IF NOT EXISTS predictions.brackets (
    season     integer PRIMARY KEY,
    as_of      date    NOT NULL,
    final_four jsonb   NOT NULL        -- [["East","West"],["South","Midwest"]]: the two semifinals
);

CREATE TABLE IF NOT EXISTS predictions.bracket_teams (
    season integer NOT NULL REFERENCES predictions.brackets (season),
    team   text    NOT NULL,
    name   text,
    seed   integer NOT NULL CHECK (seed BETWEEN 1 AND 16),
    region text    NOT NULL,
    PRIMARY KEY (season, team)
);

-- Lock the tables down: row level security on, no access for the public Supabase roles. The owner (the publisher's
-- credential) is unaffected. The read-only role gets its access from 002_api_reader.sql.
ALTER TABLE predictions.model_versions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions.snapshots       ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions.pair_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions.team_snapshots  ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions.brackets        ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions.bracket_teams   ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON SCHEMA predictions FROM %I', r);
            EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA predictions FROM %I', r);
        END IF;
    END LOOP;
END $$;
