# Phase 1 features

Every feature is a function of `(team, as_of)`: it may use games dated on or before `as_of` and nothing later.
A training row is a join of two teams' features at the game's as-of date.

## Build

```bash
python scripts/build_features.py                      # all seasons, weekly grid, per-game mode
python scripts/build_features.py --cadence daily      # daily as-of dates all season
python scripts/build_features.py --mode bracket --include ncaa
python -m unittest discover -s tests
```

Output goes to `artifacts/features/` (gitignored):

| File | Contents |
|---|---|
| `matchups.csv` | One row per game: both teams' features (`f_a`, `f_b`, `diff_f`), projections, location, labels |
| `team_asof.csv` | One row per team per as-of date |
| `columns.csv` | Generated column dictionary: definition, source, missing behavior, roadmap step |
| `report.txt` | Row counts, coverage, leak-guard result, sanity correlations, missing-value shares |

A full build takes about two minutes, most of it loading from the database.

## The as-of rule

`data/asof.py` owns it.

- **Grid.** Every Sunday from the first game through Selection Sunday, then every day through the national final.
  `--cadence daily` makes the whole season daily.
- **Per-game mode (default).** A game uses the last grid date strictly before it. A Sweet 16 row sees the round-one
  and round-two results.
- **Bracket mode.** NCAA games use Selection Sunday, for simulating a bracket before it is played (Phase 3).
- **Selection Sunday is always built**, so the bracket-time snapshot is in `team_asof.csv` in either mode.
- Games before the first grid date (the first week of the season) are dropped and counted in the report.
- **A season in progress** has no tournament games yet, so no Selection Sunday and no seed for any team. Its grid is every
  Sunday from the first game through the newest game, plus the newest game's date, and that last date is always built.
  A team_asof row includes games on its as-of date, so the last date sees every game played so far. Such a season has no
  entry in `cutoffs`.
- **`--through YYYY-MM-DD`** ignores games after that date, so a run can leave out a day still being played. A build
  with `--through` equals a build on data that stops there (`tests/test_features.py`, `SeasonInProgressTests`).
- Checked on the real data (2026-11-10): rebuilding 2022 to 2026 with 2026 cut at 1 February gave 24,655 rows up to that
  date, and all 154 numeric columns equal the full build's. Features for an earlier date do not change when later games
  arrive.

## Leakage policy

1. A feature dated `as_of` includes games with `game_date <= as_of`. A row's own game is always later.
2. Every compute function filters by date itself, so passing it too much data cannot leak.
3. The build asserts that each `team_asof` row's `max_input_date <= as_of` and that each matchup's `as_of < game_date`.
4. Season totals are never features: `team_seasons` (srs, sos, off_rtg, def_rtg, record) and `player_season_stats`
   include tournament games. Season SRS is used only as a sanity check in the report.
5. Prior-season data (Elo carry-over, returning players) is allowed because it is complete before the season starts.
6. `seed` is NaN before Selection Sunday. Conference and coach are season-level labels treated as fixed.

Tests that enforce this: `tests/test_asof.py`, and in `tests/test_features.py` the poisoning test (changing every game
on or after a date leaves earlier rows' features unchanged), the tournament-information test and the bracket-mode test.

## Modules

| Module | Role |
|---|---|
| `data/snapshots.py` | Phase 0: loaders, base team state, player and roster snapshots |
| `data/asof.py` | Grid, as-of assignment, leak guard |
| `data/ratings.py` | Adjusted efficiency and tempo, Elo, strength of schedule |
| `data/features.py` | `game_facts` loader, `team_asof`, matchups, projections, the feature registry |
| `scripts/build_features.py` | Command line entry point and report |

## Feature groups

See `columns.csv` for every column. In brief: record and scoring; raw efficiency; four factors; adjusted ratings
(`adj_off`, `adj_def`, `adj_em`, `adj_tempo`); Elo and strength of schedule (`sos_elo`, `sos_adj`); form
(`last5_margin`, `last10_margin`, `form_trend`); roster; coach; context (`conference`, `seed`); quality flags.

Matchup columns add `loc_a` (+1 team A home, -1 away, 0 neutral), `game_group` (`regular`, `conf_tourney`, `ncaa`),
projections (`proj_poss`, `proj_pts_a`, `proj_pts_b`, `proj_margin`, `proj_total`), `elo_expected_a`, and labels
(`a_win`, `margin_a`, `total`, `score_a`, `score_b`, `overtimes`).

`trainable` is 1 when both teams have features, both have played at least `--min-games` (default 8) and the game
is not an NIT/CBI-style postseason game. Rows are never dropped for failing it.

`with_both_orderings(matchups)` adds a mirrored copy of every row. Keep both copies of a game in the same fold by
grouping on `game_id` or season.

## Missing data

| Situation | Handling |
|---|---|
| Non-D1 opponent | In the raw record only. Not in ratings, Elo or form. Never a row. |
| No box score | In record, Elo and form. Not in efficiency or four factors. `box_coverage` shows the share used. |
| No game log | Falls back to box-score results, `schedule_from_box = 1`. |
| No prior season (2022) | Roster features NaN, `prior_season_available = 0`. Elo starts at 1500. |
| Early season | Ratings shrink toward the league average. Use `trainable` or `games_a`/`games_b` to filter. |
| Anything else missing | Stays NaN. Nothing is imputed. |

## Known limits

- `sos_adj` carries a small effect from the games against the team itself. `sos_elo` does not.
- Coach features count only seasons in the database (2022 onward), so long tenures look shorter than they are.
- Elo settings (`k = 24`, home advantage 90, carry 0.8, new teams 1400) and rating shrinkage (4 pseudo-games) are
  untuned starting values. Tune them in Phase 2 on tournament log loss.
- Early-season adjusted ratings are not yet pulled toward last season's (roadmap p1-16).
- A regular-season game on a Saturday uses data from the previous Sunday.

## Adding a feature

1. Compute it in a pure function that filters by `as_of`.
2. Add it to `FEATURES` in `data/features.py` (group, definition, source, missing behavior, whether to difference it).
3. Add a test, and rerun the poisoning test in `tests/test_features.py`.
4. Rebuild. `column_dictionary` fails if any output column is undocumented.
