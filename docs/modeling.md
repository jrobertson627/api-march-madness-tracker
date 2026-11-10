# Phase 2 modeling

Models read the Phase 1 matchup file (`artifacts/features/matchups.csv`) and predict the probability that team A
beats team B and the score of each team. They never touch the database.

## Run

```bash
pip install -r requirements-train.txt
python scripts/build_features.py                                              # the matrix, per-game mode
python scripts/build_features.py --mode bracket --include ncaa --tag bracket  # NCAA rows as of Selection Sunday
python scripts/run_experiments.py --set baselines
python scripts/run_experiments.py --set models                                # tuned models, 2026 left out
python scripts/report_models.py
python scripts/train_final.py --name "M5 logistic full" --out artifacts/models/v1
python -m unittest discover -s tests
```

Results go to `artifacts/models/` (gitignored): `experiments.csv` (one row per run, mode, season and metric),
`oof_predictions.csv` (every out-of-fold prediction, so reports never refit), `runs.csv` (every run ever made),
`plots/`, and `locked_box.json` once the locked season is scored.

## Evaluation protocol

| Protocol | Train on | Score on | Purpose |
|---|---|---|---|
| Leave-one-season-out (main) | Every row of the other seasons | NCAA rows of the held-out season | Model comparison |
| Forward chaining (check) | Earlier seasons only | 2024, 2025, 2026 NCAA rows (about 200 games) | Tests whether training on future seasons flatters results |
| Nested inner folds | The training seasons only, one season out at a time | NCAA and conference-tournament rows of the inner season | Hyperparameters, training mix, calibration |
| Locked box | 2022 to 2025 | 2026 NCAA rows, once | One number never used to make a choice |

Forward-chaining numbers are on a different, smaller set of games than leave-one-season-out, so do not compare
them across protocols. Compare models within a protocol.

### Locked box

The `models` set leaves 2026 out of training and testing. Choose the final design with that, then score 2026 once
with `run_locked` (`ml/experiment.py`). It refuses a second run unless forced and writes `locked_box.json`.
Baselines are fixed models with nothing to select, so they run on all five seasons.

## Modeling leakage rules

1. No row of the test season, any game type, is in a training set.
2. Imputers, scalers, calibrators and hyperparameters are fit on training rows only. Tuning and calibration use
   leave-one-season-out inside the training seasons.
3. A game and its swapped copy always fall on the same side, because splits are by season.
4. Test labels never reach a fit. `tests/test_ml_experiment.py` flips every label of a test season and checks the
   predictions do not change.
5. Seeds exist only for NCAA rows (announced on Selection Sunday), so `diff_seed` models train on NCAA rows.

## Symmetry

Team A is only the alphabetically earlier id, so a model must not learn from the ordering.

- Signed features (`diff_*`, `loc_a`, `proj_margin`) flip sign when A and B swap. `swap_features` does it.
- Logistic models use only signed columns, no intercept, features scaled without centering, missing differences set to 0.
- Tree models train on both orderings and predict `(f(x) + 1 - f(swap(x))) / 2`.
- Calibration is slope-only Platt or symmetrized isotonic, so `p(A, B) + p(B, A) = 1` still holds.
- A test checks the identity for every model.

## Models

| Id | Model | Notes |
|---|---|---|
| B0 | Coin flip | Log loss 0.693 |
| B1 | Logistic on seed difference | NCAA rows only |
| B2 | Seed-pair history, smoothed toward B1 | Sparse |
| B3 | Phase 1 Elo probability | No fitting |
| B4 | Logistic on `diff_adj_em` and `loc_a` | The efficiency-gap baseline |
| M5 | L2 logistic on `ratings_plus`, `full`, `full_with_seed` | C and training mix tuned in nested folds |
| M6 | Histogram gradient boosting | Depth, iterations, mix tuned; optional monotone constraints |
| S1, S2 | Ridge or boosted margin and total models | Scores derived from margin and total |

Feature sets are defined in `ml/dataset.py` (`feature_columns`). Training mixes (`MIXES`) weight regular,
conference-tournament and NCAA rows.

## Scores

Margin and total are predicted, then `score_a = (total + margin) / 2` and `score_b = (total - margin) / 2`, so the
scores, margin and win probability agree. The margin-implied probability is `Phi(margin / sigma)` with `sigma` the
training residual spread per game group. Margin targets are clipped at 30 points.

Overtime games add minutes, so total and scores are rescaled to a 40-minute game for training:
`total * 40 / (40 + 5 * overtimes)`. Margin keeps its real value. Score error is reported against both the raw and
the 40-minute totals (`total_mae_raw`, `total_mae_reg`).

## Metrics and the decision rule

Log loss is primary. Brier score, accuracy and calibration error are reported. Calibration error is noisy with
about 33 games per bin, so it is a diagnostic only.

To compare models, use a paired bootstrap on the games both were tested on (10,000 resamples). Keep a more complex
model over a simpler one only if the 95% interval of the log-loss difference excludes zero, or it wins by at least
0.005 log loss in four of five seasons. Otherwise keep the simpler model. Measured on the 334 NCAA games, the
bootstrap standard error of a log-loss difference is about 0.02, so many comparisons will be inconclusive. That is
a result, not a failure.

Every report prints how many configurations and runs have been tried. Many experiments on five seasons make the
best one partly luck.

## Both as-of modes

Each fitted model scores NCAA games twice: with per-game features (each game sees every earlier result) and with
bracket features (everything as of Selection Sunday). Bracket mode is the honest one for filling out a bracket
before the tournament, and it is expected to score worse. `experiments.csv` has a `mode` column.

## Outside benchmarks

`ml/benchmarks.py` scores a file of betting lines or ratings on the same games with the same metrics. The file
needs `season, game_date, team_a, team_b` and either `p_a_win` or `spread_a`, using the same team ids as the matrix.
Lines are not in the database yet, so this is tested on simulated lines. To add them:

```python
from ml.benchmarks import load_benchmark, score_benchmark, compare_to_model
bench = load_benchmark("lines.csv", sigma=11.0)   # estimate sigma on other seasons with estimate_sigma
```

### Scoring a lines file you supply

`ml/lines.py` reads a table in BetIQ's game-table layout and `scripts/score_lines.py` scores it. Nothing in the repo
downloads lines, and the code never contacts any site.

Columns (headers matched ignoring case, extra columns ignored, blank or `-` cells treated as missing), one row per
team per game:

| Column | Meaning |
|---|---|
| `Date` | Game date |
| `Team`, `Opponent` | BetIQ short names such as `Abl Christian` or `NC A&T` |
| `Spread` | The team's own spread, negative when it is the favorite |
| `Total (O/U)` | Over/under |
| `Money Line` | American odds for that team |
| `Score` | `team-opponent`, for example `69-63` |

```bash
python scripts/export_team_names.py
python scripts/score_lines.py path/to/lines.csv
```

- Names are matched through `BETIQ_ALIASES_RAW`. Of the 387 names on BetIQ's team list, 374 match, including all 162
  teams that appear in our tournament matrix. The unmatched are defunct or non-D1 teams. Unmatched names are printed.
- The two rows of a game are combined into one in the matrix's order. `p_ml` is the de-vigged moneyline probability and
  `p_spread` comes from the spread; `p_a_win` is `p_ml` when both moneylines exist and `p_spread` otherwise.
- The spread scale (sigma) is estimated on games that are not NCAA tournament games in the matrix, so it never sees the
  games being scored.
- Dates may differ by a day (late tip-offs); a pair that meets twice takes the nearer date.
- The report gives how many tournament games had a line, the lines' log loss, and a paired comparison with the stored
  predictions. Reading the stored 2026 predictions does not rerun the locked-box test, and nothing is tuned on it.
- Check that the spread is the closing line before drawing conclusions. Whether BetIQ's column is the closing line is
  not confirmed.
- Check the source's terms of use before collecting or storing lines. BetIQ's terms bar copying content without written
  consent and its robots.txt disallows its data endpoint.

### Matching team names

Lines write school names ("UConn", "Michigan St."), the matrix uses Sports Reference ids. Export the team table once,
then map the names. Anything unmatched is returned, never guessed, and ambiguous names are reported as unmatched:

```bash
python scripts/export_team_names.py        # artifacts/models/team_names.csv (read-only database query)
```

```python
from ml.team_names import map_team_columns
lines, unmatched = map_team_columns(lines, team_table)   # extend ALIASES or pass aliases={...} for the rest
print(unmatched)
```

Matching uses the school name inside the id (`.../schools/michigan-state/men`), because display names carry mascots.

## Exporting a model

`fit_final` trains on every season and returns a `Predictor` (`ml/predictor.py`). `save` writes `model.joblib` and
`manifest.json` (features, seasons and dates trained on, data hash, hyperparameters, calibration, sigma, package
versions, cross-validated metrics). `scripts/train_final.py` also writes `model_card.md` (data, features, hyperparameters, cross-validated results, limits). The predictor raises a clear error if columns are missing. For a logistic win
model with ridge scores, `export_json` writes the coefficients and `JsonPredictor` predicts with numpy and pandas
only, so the web service does not need scikit-learn.

## Adding a model

1. Add a class with `fit(X, y, w)` and `predict_proba(X)` to `ml/models.py` and register it in `WIN_MODELS`.
2. Make it symmetric (see above) and add it to the symmetry test.
3. Add a `Config` to `ml/sets.py` and run `scripts/run_experiments.py`.
4. Judge it with `scripts/report_models.py`, not by eye.

## Known limits

- 334 NCAA games over five seasons limits what can be shown. Treat small differences as noise.
- Elo and rating-shrinkage settings are still the Phase 1 starting values (tuning is a separate step).
- `sigma` is estimated in sample on the training rows, which slightly understates it for flexible models.
- Early-season ratings are not pulled toward last season's (roadmap p1-16).
