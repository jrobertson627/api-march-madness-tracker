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
