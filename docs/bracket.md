# Phase 3 bracket simulation

Turns the Phase 2 model into tournament odds, picks and upset flags. Everything is in `bracket/`.

## Run

```bash
python scripts/simulate_bracket.py --season 2026 --model artifacts/models/v1          # a season in the database
python scripts/simulate_bracket.py --season 2027 --bracket field.csv \
    --final-four "East-West,South-Midwest" --model artifacts/models/v1 \
    --features artifacts/features/team_asof_bracket.csv --as-of 2027-03-14               # a new tournament
python scripts/simulate_bracket.py --season 2026 --model artifacts/models/v1 --results played.csv   # update odds
python scripts/backtest_bracket.py             # select: 2022 to 2025, chooses tau
python scripts/backtest_bracket.py --locked    # score 2026 once
python -m unittest discover -s tests
```

Outputs go to `artifacts/bracket/<season>/` (gitignored): `advancement_<season>.csv`, `first_round_<season>.csv`,
`picks_<season>_<strategy>.csv`, `bracket_<season>.json` (one object for an API or page) and `report_<season>.txt`.

## How it works

1. **Structure** (`bracket/structure.py`). A bracket is 64 slots in bracket order: four regions of 16, each in the order
   1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15. Neighbouring slots meet. A slot with two teams (the same
   region and seed) is a First Four game, so the seeds involved can change each year. The Final Four pairing is read
   from the real semifinals for a past season, or given with `--final-four`.
2. **Pair matrix** (`bracket/probs.py`). Every pair of teams becomes a hypothetical neutral-court NCAA game as of
   Selection Sunday and the saved model predicts it. A team with no features gets a seed-only probability and is
   listed in the report.
3. **Odds** (`bracket/simulate.py`). With independent games the odds are computed exactly. With `--tau` above zero they
   are simulated (vectorized, 100,000 brackets in about 0.1 seconds). A level l array holds P(team wins game j of
   level l), so the sum over games is the chance of being in the next round.
4. **Picks** (`bracket/pick.py`). `seed chalk` always picks the better seed, `model favorite` picks the higher
   probability, and `best expected points` is the consistent bracket with the most expected points, computed exactly
   in one pass up the tree. Scoring is points per round, 10, 20, 40, 80, 160, 320 by default.
5. **Upset flags** (`bracket/upsets.py`). A first-round game is flagged when the model gives the underdog at least 50%,
   or at least 15 points more than the historical rate for that seed pairing (from other seasons only).

## Team-strength uncertainty (tau)

Independent games treat the model's probabilities as exact. With `tau > 0`, each simulated bracket gives every team a
random shift on its rating, `Normal(0, tau)` in logit units, so a team the model underrates can win several rounds in
the same bracket. `tau = 0` is the default. The backtest chooses tau on 2022 to 2025 and keeps a value above zero only
if it beats zero in at least three of the four seasons.

## Checks and tests

- `Bracket.replay` feeds real results through the structure and lists anything that does not fit. All 335 games of
  2022 to 2026 replay with no problems, which confirms the slot order and the Final Four pairings.
- Small brackets are enumerated exactly and the simulator must agree within four sampling errors.
- Title chances sum to 1 and the chance of reaching round r sums to the number of teams in that round.
- The best-expected-points calculation equals brute force on 8-team brackets.
- A fixed game always goes to its real winner.

## Backtest rules

Each season's probabilities come from a model trained on the other 2022 to 2025 seasons, never on the season itself and
never on 2026. The score is the log loss and Brier score of every team-round event (did the team reach this round),
about 400 events a season. Events in a bracket depend on each other, so they are not independent observations, and
five tournaments cannot prove one pick strategy better than another. 2026 is scored once.

The saved model `artifacts/models/v1` was trained on all five seasons. Using it for a past season gives in-sample odds,
and `simulate_bracket.py` says so.

## Known limits

- Ratings are frozen at Selection Sunday. A team's showing in round one does not change its later odds.
- Every game is treated as neutral. Teams playing close to home are not modeled.
- Injuries and availability are not in the data.
- Pool strategy using public pick percentages is not built (no data).
- The simulator reads one model. Its probabilities are within noise of betting lines (decision 17), not better.
