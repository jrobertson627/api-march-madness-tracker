# Phase 1 decisions

Short records. Each says what was chosen, what it gives up and what would change it.

## 1. Features are keyed by (team, as_of) (2026-11-10)

Chosen: compute features per team per date, then join two teams per game.
Why: the same code serves tournament rows, regular-season rows and the Phase 4 live endpoint.
Gives up: more compute than one snapshot per season, and more code than tournament-only rows.
Revisit if: the extra regular-season rows do not lower tournament log loss in Phase 2.

## 2. Tournament games may use earlier tournament games (2026-11-10)

Chosen: per-game mode is the default. A game uses everything dated before it. Bracket mode (Selection Sunday for
every NCAA row) exists for Phase 3 simulations.
Why: a prediction made on the morning of a Sweet 16 game really can see round one and two.
Gives up: two modes to test and report.

## 3. Weekly grid, then daily from Selection Sunday (2026-11-10)

Chosen: Sundays until Selection Sunday, every day after.
Gives up: a regular-season Saturday game sees data up to six days old.
Revisit if: Phase 2 shows staleness hurts. `--cadence daily` is one flag away.

## 4. Rating method

Chosen: alternating updates with shrinkage, which converge to ridge regression, with a venue term. No matrix, no scipy.
Chosen: Elo as well as adjusted efficiency. Elo uses every game result, adjusted efficiency uses how points were scored.
Their rank correlation is 0.90 to 0.96 at Selection Sunday, so they overlap. Keep both until Phase 2 shows one adds nothing.
Starting values are untuned and will be tuned in Phase 2.

## 5. Missing data stays missing

Chosen: NaN plus flags, no imputation. Rows are flagged `trainable`, never dropped for missing values.
Why: the right treatment depends on the model. Gradient-boosted models take NaN natively, regularized linear ones need a choice.

## 6. Training rows

Chosen: tournament, conference tournament and regular-season games, all with as-of features. Phase 2 scores on
tournament games only, holds out the evaluation season, and decides by experiment whether the extra rows help.

## 7. File format

Chosen: CSV. Revisit if the files get unwieldy (then add pyarrow and write Parquet).

## 8. Coach history

Chosen: counted within the seasons in the database only (2022 onward). Revisit if earlier seasons are crawled.
