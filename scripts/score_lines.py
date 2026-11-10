"""Score betting lines (and our models against them) on the NCAA tournament games.

    python scripts/score_lines.py path/to/lines.csv

The lines file is one you supply, in BetIQ's game-table layout: Date, Team, Opponent, Location, Spread,
Total (O/U), Money Line, Score (one row per team per game). See ml/lines.py. Nothing here downloads data.

Needs artifacts/models/team_names.csv (python scripts/export_team_names.py) and the matrix
(python scripts/build_features.py). It prints how many tournament games were matched and which names were not,
the lines' own log loss, and a paired comparison with the models already in artifacts/models.

Reading the stored 2026 predictions here does not rerun or change the locked-box result.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml import metrics  # noqa: E402
from ml.benchmarks import compare_to_model, estimate_sigma  # noqa: E402
from ml.dataset import MATRIX_PATH, load_matchups  # noqa: E402
from ml.experiment import MODEL_DIR  # noqa: E402
from ml.lines import betiq_to_games, match_to_matrix  # noqa: E402
from ml.sets import FINAL_NAME  # noqa: E402

TEAMS = MODEL_DIR / "team_names.csv"
COMPARE = [FINAL_NAME, "M6 hgb full monotone", "B4 efficiency gap, all rows", "B3 elo probability", "B1 seed logistic"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lines")
    args = ap.parse_args()
    if not TEAMS.exists():
        sys.exit("run python scripts/export_team_names.py first")

    matrix = load_matchups(MATRIX_PATH)
    ncaa = matrix[matrix["game_group"] == "ncaa"]
    teams = pd.read_csv(TEAMS)

    # Spread-to-probability scale: estimate it on games that are NOT NCAA tournament games in the matrix.
    g0, _ = betiq_to_games(args.lines, teams, sigma=11.0)
    keys = ncaa[["season", "team_a", "team_b"]].assign(game_date=ncaa["game_date"])
    other = g0.merge(keys, on=["season", "game_date", "team_a", "team_b"], how="left", indicator=True)
    other = other[(other["_merge"] == "left_only") & other["spread_a"].notna() & other["margin_a"].notna()]
    sigma = estimate_sigma(other["spread_a"], other["margin_a"]) if len(other) >= 200 else 11.0
    lines, rep = betiq_to_games(args.lines, teams, sigma=sigma)
    print(f"lines file: {rep['rows']} rows, {rep['games']} games, {rep['rows_dropped_unmatched']} rows dropped for unmatched names")
    if rep["unmatched_names"]:
        print("unmatched names (add to BETIQ_ALIASES_RAW in ml/lines.py or pass aliases):", rep["unmatched_names"])
    print(f"spread scale sigma = {sigma:.2f} points ({'estimated from non-tournament games' if len(other) >= 200 else 'fallback'})")

    m = match_to_matrix(lines, ncaa).merge(ncaa[["game_id", "a_win"]], on="game_id")
    m = m[m["p_a_win"].notna()]
    print(f"\nNCAA tournament games with a line: {len(m)} of {len(ncaa)}")
    print(m.groupby("season").size().rename("games with a line").to_string())
    if m.empty:
        return
    y, p = m["a_win"], m["p_a_win"]
    print(f"\nlines on those games: log loss {metrics.log_loss(p, y):.4f}  brier {metrics.brier(p, y):.4f}  accuracy {metrics.accuracy(p, y):.3f}")
    print(m.assign(ll=metrics.game_log_loss(p, y)).groupby("season")["ll"].mean().round(4).rename("lines log loss").to_string())

    exp = pd.read_csv(MODEL_DIR / "experiments.csv")
    oof = pd.read_csv(MODEL_DIR / "oof_predictions.csv")
    ids = exp.drop_duplicates("name").set_index("name")["run_id"].to_dict()
    bench = m.rename(columns={"p_a_win": "p_bench"})[["season", "game_date", "team_a", "team_b", "p_bench"]]
    print("\nour models against the lines on shared games (negative diff: our model is better):")
    rows = []
    for name in COMPARE:
        if name not in ids:
            continue
        o = oof[(oof["run_id"] == ids[name]) & (oof["mode"] == "game")]
        r = compare_to_model(bench, o)
        if r.get("games"):
            rows.append({"model": name, "games": r["games"], "model_log_loss": r["model_log_loss"], "lines_log_loss": r["bench_log_loss"],
                         "diff": r["mean_diff"], "ci_low": r["lo"], "ci_high": r["hi"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False) if rows else "no stored predictions overlap")


if __name__ == "__main__":
    main()
