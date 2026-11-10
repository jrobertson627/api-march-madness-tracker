"""Run named sets of experiments and save the results.

    python scripts/run_experiments.py --set baselines
    python scripts/run_experiments.py --set models
    python scripts/run_experiments.py --set models --only hgb

Sets are defined in ml/sets.py. The "models" set leaves 2026 (the locked box)
out of training and testing. Results go to artifacts/models/ (experiments.csv,
oof_predictions.csv, runs.csv). Bracket-mode results are included when
artifacts/features/matchups_bracket.csv exists:

    python scripts/build_features.py --mode bracket --include ncaa --tag bracket
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.dataset import BRACKET_PATH, MATRIX_PATH, data_stamp, load_matchups  # noqa: E402
from ml.experiment import run_experiment, save_results  # noqa: E402
from ml.report import summary  # noqa: E402
from ml.sets import SETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--only", default="", help="run only configs whose name contains this text")
    ap.add_argument("--no-bracket", action="store_true")
    args = ap.parse_args()

    df = load_matchups(MATRIX_PATH)
    stamp = data_stamp(MATRIX_PATH, df)
    bracket = None
    if not args.no_bracket and BRACKET_PATH.exists():
        bracket = load_matchups(BRACKET_PATH)
    print(f"matrix {stamp}; bracket rows: {0 if bracket is None else len(bracket)}")

    configs = [c for c in SETS[args.set]() if args.only.lower() in c.name.lower()]
    for cfg in configs:
        t0 = time.time()
        res = run_experiment(cfg, df, bracket=bracket, data_hash=stamp["hash"])
        save_results(res)
        row = res.metrics[(res.metrics["mode"] == "game") & (res.metrics["season"].astype(str) == "all")].iloc[0]
        print(f"{cfg.name:<46} log_loss {row['log_loss']:.4f}  brier {row['brier']:.4f}  acc {row['accuracy']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    exp = pd.read_csv(ROOT / "artifacts" / "models" / "experiments.csv")
    print("\nper-game mode, pooled over tested seasons:")
    print(summary(exp, "game").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
