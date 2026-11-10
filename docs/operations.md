# Operating the prediction API

The model runs on your machine. The web service only reads published predictions. See `docs/api.md` for the endpoints.

## The pieces

| Piece | Where it runs | Credential |
|---|---|---|
| Publisher (`scripts/publish_predictions.py`) | Your machine | `PUBLISH_DATABASE_URL`, can write |
| API (`app.py`, gunicorn) | Render | `DATABASE_URL`, read-only (`api_reader`) |
| Tables | Supabase, schema `predictions` | `migrations/001_predictions.sql`, `002_api_reader.sql` |

## First-time setup

1. **Check the migrations without keeping anything.** Both of these run in a transaction that is always rolled back:

   ```bash
   python scripts/check_migration.py
   python scripts/check_store_postgres.py
   ```

2. **Apply `migrations/001_predictions.sql`** to the Supabase database (SQL editor, or `psql`). It creates the `predictions` schema and six tables
   and does not touch any existing table.
3. **Create the read-only role yourself**, so the password stays with you. In the SQL editor:

   ```sql
   CREATE ROLE api_reader WITH LOGIN PASSWORD '<a long random password>';
   ```

   Then run `migrations/002_api_reader.sql`. It grants read access to the `predictions` tables and to six scraper tables, and adds a read-only
   row policy for `api_reader` on those six scraper tables. It changes nothing for any other role.
4. **Publish** (writes need the write credential; put it in your own `.env`, never in the repo):

   ```bash
   # PUBLISH_DATABASE_URL=postgresql://postgres...   in .env
   python scripts/publish_predictions.py --backtest --seasons 2022 2023 2024 2025 2026 --to postgres --dry-run
   python scripts/publish_predictions.py --backtest --seasons 2022 2023 2024 2025 2026 --to postgres
   python scripts/publish_predictions.py --live artifacts/models/v1 --seasons 2026 --to postgres --activate
   ```

5. **Test the read-only connection before it goes on Render** (this sets the variable for one command only and never prints it):

   ```powershell
   $env:DATABASE_URL = "<the api_reader connection string>"
   python scripts/check_readonly_url.py
   Remove-Item Env:DATABASE_URL
   ```

   Every line should say `ok`. Run on the current `postgres` connection it reports that the connection can write, which is why the web service should not use it.
6. **Point the web service at the read-only role.** In the Render dashboard set `DATABASE_URL` to the `api_reader` connection string (on Supabase's pooler
   the user is usually `api_reader.<project-ref>`), `STORE=postgres`, and `CORS_ORIGINS` to the frontend's address (comma-separated).
7. **Deploy**, then check `GET /health`: `status` should be `ok`, with the active model and latest snapshot.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DATABASE_URL` | API | Read-only connection string |
| `STORE` | API | `postgres`, `files`, or `auto` (postgres when `DATABASE_URL` is set) |
| `FILE_STORE_DIR` | API | Folder for the file store (local work), default `artifacts/predictions` |
| `CORS_ORIGINS` | API | Allowed frontend origins; default is local development ports only |
| `PUBLISH_DATABASE_URL` | Publisher | Write connection string |

## Publishing a new model or season

- Snapshots never change. A new model gets a new id and new rows; `--activate` makes it the active model.
- Publishing a snapshot that exists is refused unless you pass `--replace`.
- The publisher checks every snapshot before writing (probabilities strictly between 0 and 1, every pair present, factors adding up) and stops if
  anything fails. `--dry-run` runs all checks and writes nothing.
- A snapshot is written in one transaction, so a failed publish leaves nothing half-written.

## Rolling back

The active model decides what the API uses by default. To go back, activate the earlier model:

```sql
UPDATE predictions.model_versions SET active = false WHERE active;
UPDATE predictions.model_versions SET active = true WHERE id = '<earlier model id>';
```

Nothing is deleted. A response always names the model and snapshot it came from.

## Size

About 66,000 pairs per full snapshot. Measured after the first publish: 393,873 pair rows took 166 MB with indexes, so roughly 28 MB per snapshot, and
the whole `predictions` schema was 167 MB. Keep the Selection Sunday snapshot for each season and the latest one, and delete older in-season snapshots when you add them.

## Render's free plan

Render's own documentation says a free web service spins down after 15 minutes without traffic, has an ephemeral filesystem, and is not meant for
production use. The first request after a spin-down waits for the service to start. The app itself imports in under a second and does not load any
machine-learning library, so the wait is mostly Render's. Nothing in the app writes to local disk.

## Checks after a deploy

- `GET /health` is `ok`.
- `GET /v1/seasons` lists the seasons you published.
- `GET /v1/predict/matchup?team_a=duke&team_b=houston&season=2026` returns a probability, and its `model.id` and `snapshot.in_sample` are what you expect.
- A browser request from the frontend's origin works, and one from another origin does not.
