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

# Phase 2 decisions

## 9. Evaluation protocol (2026-11-10)

Chosen: leave-one-season-out as the main protocol, forward chaining as a check, nested leave-one-season-out inside
the training seasons for hyperparameters, training mix and calibration.
Gives up: leave-one-season-out trains on future seasons. Forward chaining shows how much that matters, on fewer games.
Revisit if: the two disagree materially.

## 10. Locked-box season

Chosen: select models with 2026 left out of training and testing, then score 2026 once. The runner refuses a second
run. Baselines are fixed models with nothing to select, so they use all five seasons.
Gives up: one season of selection data.

## 11. Decision rule

Chosen: log loss is primary. Keep a more complex model only if the paired 95% bootstrap interval excludes zero, or
it wins by at least 0.005 log loss in four of five seasons. The 0.005 threshold is a proposal.
Why: with 334 games the standard error of a log-loss difference is about 0.02, so most gains will be inconclusive.

## 12. Symmetry

Chosen: signed features only in linear models, no intercept, no centering; both orderings and swap-averaging for
trees; slope-only Platt calibration. The alphabetical ordering of team A carries no information (team A won 57% of
NCAA rows by chance), so no model may learn from it.

## 13. Overtime

Chosen: total and scores are rescaled to a 40-minute game for training. Margin keeps its real value. Both versions
are reported. Overtime is 4.5% of NCAA games, so the effect is measured, not assumed.

## 14. Dependencies and serving

Chosen: scikit-learn, scipy and joblib for training, in `requirements-train.txt`, so the Render web service
(`requirements.txt`) stays small. A logistic model with ridge scores exports to JSON and predicts with numpy.
Revisit if: boosted trees win, then the service needs scikit-learn.

## 15. Outside benchmarks

Betting lines exist and will be added later. The loader and paired comparison are built and tested on simulated
lines. Real lines are scored when the file arrives.

## 16. Final design and the locked-box result (2026-11-10)

Chosen on 2022 to 2025: boosted trees on the full feature set, monotone in the rating differences, no calibration,
ridge margin and total models for scores (`FINAL boosted full monotone + ridge scores`). It had the lowest selection
log loss (0.5562, a tie with the plain version at 0.5564) and passed the decision rule against the efficiency-gap
baseline (0.5684, ahead in 4 of 4 seasons). It did not clearly beat the logistic model on the same seasons.

Locked box, scored once, trained on 2022 to 2025: 2026 log loss 0.4881 per game, 0.4886 bracket mode, margin error
9.9 points, total error 11.2 points. On the same 67 games the efficiency-gap baseline scored 0.4687, so the final
model was worse than the simple baseline by 0.019 (interval -0.020 to +0.065). It was better than the seed logistic
(0.5140), Elo (0.5097) and a coin flip. Sixty-seven games cannot separate close models, so this does not show the
baseline is better, but it does not confirm that boosting helps either.

Reading: the efficiency-gap logistic, the logistic on all features and the boosted model are within noise of each
other. The decision to serve boosted trees rests on the selection seasons only. Revisit if more seasons or betting
lines change the picture.

Confirmed 2026-11-10: boosted trees stay the default after seeing the locked-box result. The efficiency-gap logistic
remains the simple fallback (it exports to JSON and needs no scikit-learn in the web service).

## 17. Betting lines as a benchmark (2026-11-10)

Source: a table the user copied from BetIQ's custom trend tool for the 2022 to 2026 NCAA tournaments (670 rows, 334
games with both teams in our matrix, no missing values). It is kept locally in `artifacts/lines/` (gitignored) and
is not committed. Whether its spread is the closing line is not confirmed.

Results, win-probability log loss on the 334 tournament games: lines 0.5383 (moneylines, margin removed), spread-only
0.5417 to 0.5413. Our models on shared games: efficiency-gap baseline 0.5482 (lines better by 0.010, interval
-0.005 to +0.024), seed logistic 0.5650 and Elo 0.5801 (both clearly worse than the lines). On 2022 to 2025 the final
design scored 0.5562 against the lines' 0.5585 (inside the noise; those seasons were used to choose the design).
On 2026, the one unbiased comparison, the final model scored 0.4881 against 0.4580 for the lines (lines better by
0.030, interval -0.011 to +0.077).

Reading: our best model and the market are within noise of each other on tournament games. They pick different
winners in 24 of 267 games and their probabilities correlate at 0.96. Lines were used only to benchmark. Using them
as a model input is a separate decision (they do not exist for later rounds before games are set).

# Phase 3 decisions

## 18. Bracket simulation and its backtest (2026-11-10)

Design: the bracket is rebuilt from seeds and regions (all 335 real games of 2022 to 2026 replay with no problems);
win probabilities for every pair of teams come from the Phase 2 model as of Selection Sunday; with independent games
the odds are computed exactly; picks are the exact best-expected-points bracket, the model favorite and seed chalk.
Pool strategy is skipped (no public pick percentages).

Backtest on 2022 to 2025 (each season scored by a model trained on the other three, never on 2026). Every team and
round is an event (did the team reach it), 408 events a season. Mean log loss over the four seasons: coin 0.3284,
seed-only 0.2512, efficiency gap 0.2485, final model 0.2414. The model beat the efficiency gap in all four seasons and
the seeds in two (2024 and 2025); seeds were slightly better in 2022 and 2023.

Team-strength uncertainty (tau): the best value on the grid was 0.3 and it beat tau 0 in 3 of 4 seasons, but by only
0.0002 log loss. Choosing tau on the other seasons and scoring the held-out one (nested) was worse than tau 0 (0.2419
against 0.2414). The rule was tightened to require the nested check as well, a change made after seeing these
numbers, and tau stays 0 (independent games). The grid stops at 0.3 with the best value at the edge, so larger values
were not tested.

Picks, realized points with 10-20-40-80-160-320 scoring: mean over 2022 to 2025 was 740 for best expected points,
797.5 for model favorite and 767.5 for seed chalk, although the model expected 836, 825 and 794. That is a handful of
upsets, so no strategy is shown to be better. Upset flags: games where the model picks the upset were 8 upsets in 19
(42%) against 25 in 102 (25%) unflagged. Games where the model is 15 points above seed history were 3 in 16 (19%), no
better than unflagged, so that rule is not supported.

Locked season 2026 (model trained on 2022 to 2025, scored once): advancement log loss coin 0.3305, seed-only 0.2104,
efficiency gap 0.1835, final model 0.1867 (the model was 0.003 behind the efficiency gap and 0.024 ahead of the
seeds). Realized points: seed chalk 1000, model favorite 1040, best expected points 1060. Both games the model picked
as upsets were upsets, against 6 of 30 unflagged. The model gave the real champion, Michigan, a 14.7% title chance.

Reading: the model's round-by-round odds beat seed-only odds in most seasons, and are about the same as the simple
efficiency-gap model. Five tournaments cannot rank pick strategies or settings finely.

# Phase 4 decisions

## 19. Publish, then serve (2026-11-10)

The model runs offline. A publisher script writes every pair of teams (win probability, projected scores, factor effects)
for a snapshot into new tables, and the web API only reads them. Reasons: the free Render plan spins down after 15
idle minutes and has an ephemeral filesystem, importing scikit-learn alone takes about 1.5 seconds, and the bracket
odds, picks and what-if results need only numpy and take under a millisecond from stored pair probabilities.
Gives up: answers are as fresh as the last publish, and there are no arbitrary inputs (other dates, other venues).

## 20. Tables and roles

A new `predictions` schema, separate from the scraper's tables, with row level security on and no access for the
public Supabase roles. The web service connects as a read-only `api_reader` role that you create yourself, so the
password never passes through the code or the chat. The role needs a read-only row policy on six scraper tables the
existing `/teams` routes read; that is the only change to existing tables. Migrations are tested with
`scripts/check_migration.py` and `scripts/check_store_postgres.py`, which always roll back.

## 21. History and honesty in the responses

Past seasons are published from models that did not train on them (the Phase 3 backtest models; 2026 from the model
trained on 2022 to 2025). The saved live model is also published for 2026 and marked in-sample. The API prefers an
out-of-sample snapshot, and every response says which model, which snapshot date and whether it is in-sample.

## 22. Scope

Neutral court only, new routes under `/v1`, the existing team routes unchanged. Factor effects come from group
ablation (zero a group's differences and see how far the probability moves) with the leftover shown as interaction;
they are descriptions of the model, not causes. CORS allows only local development origins until the frontend
address is known. Hosting stays on the free plan with cold starts accepted.
