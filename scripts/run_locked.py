"""Score the locked-box season (2026) once with the final configuration.

    python scripts/run_locked.py

Trains on 2022 to 2025 only, tuning inside those seasons, and scores 2026 NCAA games in per-game and bracket
mode. It refuses to run a second time (artifacts/models/locked_box.json). The result is compared with the
baselines on the same 2026 games. Do not change the configuration after seeing this number.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.dataset import BRACKET_PATH, MATRIX_PATH, data_stamp, load_matchups  # noqa: E402
from ml.experiment import MODEL_DIR, run_locked, save_results  # noqa: E402
from ml.report import paired  # noqa: E402
from ml.sets import FINAL_NAME, find  # noqa: E402

SEASON = 2026
COMPARE = ["B0 coin flip", "B1 seed logistic", "B2 seed pairs", "B3 elo probability", "B4 efficiency gap, all rows"]


def main() -> None:
    df = load_matchups(MATRIX_PATH)
    bracket = load_matchups(BRACKET_PATH) if BRACKET_PATH.exists() else None
    cfg = find(FINAL_NAME)
    res = run_locked(cfg, df, SEASON, bracket=bracket, data_hash=data_stamp(MATRIX_PATH, df)["hash"])
    save_results(res)

    m = res.metrics[res.metrics["season"].astype(str) == "all"]
    cols = ["mode", "n", "log_loss", "brier", "accuracy", "ece", "margin_mae", "total_mae_reg", "total_mae_raw", "score_mae_raw", "margin_prob_log_loss"]
    print(f"locked box: {SEASON}, trained on 2022 to 2025, chosen {res.oof['chosen'].iloc[0]}")
    print(m[cols].round(4).to_string(index=False))

    exp = pd.read_csv(MODEL_DIR / "experiments.csv")
    oof = pd.read_csv(MODEL_DIR / "oof_predictions.csv")
    ids = exp.drop_duplicates("name").set_index("name")["run_id"].to_dict()
    print(f"\nbaselines on the same {SEASON} games (per-game mode):")
    rows = []
    for name in COMPARE:
        b = exp[(exp["name"] == name) & (exp["mode"] == "game") & (exp["season"].astype(str) == str(SEASON))]
        sub = oof[(oof["run_id"] == ids[name]) & (oof["mode"] == "game") & (oof["season"] == SEASON)]
        r = paired(oof[oof["season"] == SEASON], name, FINAL_NAME, ids | {FINAL_NAME: cfg.run_id()}, n=10000)
        rows.append({"baseline": name, "baseline_log_loss": float(b["log_loss"].iloc[0]), "diff_vs_baseline": r["mean_diff"], "ci_low": r["lo"], "ci_high": r["hi"], "games": r["games"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nnegative diff means the final model has the lower log loss. 67 games is a small test, so read intervals, not points.")


if __name__ == "__main__":
    main()
