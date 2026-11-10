"""Fit a named configuration on all seasons and export it.

    python scripts/train_final.py --name "M5 logistic ratings_plus" --out artifacts/models/v1

Writes model.joblib and manifest.json (and model.json when the model is
logistic with ridge scores, for serving without scikit-learn). Cross-validated
metrics for the same run id are copied into the manifest when experiments.csv
has them. Run this only after the choice of model is final; it trains on the
locked-box season too.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.dataset import MATRIX_PATH, data_stamp, load_matchups  # noqa: E402
from ml.experiment import MODEL_DIR  # noqa: E402
from ml.model_card import write_model_card  # noqa: E402
from ml.predictor import fit_final  # noqa: E402
from ml.sets import find  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = find(args.name)
    df = load_matchups(MATRIX_PATH)
    cv = []
    path = MODEL_DIR / "experiments.csv"
    if path.exists():
        e = pd.read_csv(path)
        cv = e[(e["run_id"] == cfg.run_id()) & (e["season"].astype(str) == "all")].to_dict("records")
    pred = fit_final(cfg, df, stamp=data_stamp(MATRIX_PATH, df), cv_metrics=cv)
    out = pred.save(args.out)
    plot = MODEL_DIR / "plots" / "reliability.png"
    write_model_card(out, pred.manifest, str(plot) if plot.exists() else None)
    try:
        pred.export_json(out / "model.json")
    except NotImplementedError as e:
        print(f"no JSON export: {e}")
    print(f"saved {out}; chose {pred.manifest['chosen_hyperparameters']}; {len(cv)} cross-validation rows attached")


if __name__ == "__main__":
    main()
