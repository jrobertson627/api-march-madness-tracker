"""Tune the Phase 1 rating settings (shrinkage, Elo k and carry) on a small grid.

    python scripts/tune_features.py            # uses artifacts/features/raw_cache.pkl if present
    python scripts/tune_features.py --refresh  # reload from the database

Selection uses 2022 to 2025 only (2026 is the locked box). For each setting the
features are rebuilt from cached raw data (no roster features, which these
ratings do not use), then a logistic model with one rating difference and
location is scored by leave-one-season-out on the NCAA and conference-tournament
games of the held-out season. A setting replaces the default only if it passes
the decision rule against the default: the paired 95% interval excludes zero,
or it wins by 0.005 log loss in every selection season.

Results: artifacts/models/feature_tuning.csv
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.features import build_from_raw, load_raw  # noqa: E402
from ml import metrics  # noqa: E402
from ml.dataset import add_derived, feature_columns, training_weights  # noqa: E402
from ml.models import LogisticModel  # noqa: E402

CACHE = ROOT / "artifacts" / "features" / "raw_cache.pkl"
OUT = ROOT / "artifacts" / "models" / "feature_tuning.csv"
SEASONS = [2022, 2023, 2024, 2025]
ALL_SEASONS = [2022, 2023, 2024, 2025, 2026]
DEFAULTS = {"shrink_games": 4.0, "elo_k": 24.0, "elo_carry": 0.8}
SHRINK_GRID = [2.0, 4.0, 8.0]
ELO_GRID = [(k, c) for k in (16.0, 24.0, 32.0) for c in (0.6, 0.8)]


def raw_data(refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    raw = load_raw(ALL_SEASONS)
    CACHE.write_bytes(pickle.dumps(raw))
    return raw


def matrix(raw, shrink, k, carry) -> pd.DataFrame:
    m = build_from_raw(raw, shrink_games=shrink, elo_kwargs={"k": k, "carry": carry}, with_players=False)["matchups"]
    return add_derived(m[m["trainable"] == 1]).reset_index(drop=True)


def losses(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Per-game log loss, leave one selection season out, on NCAA and conference-tournament games."""
    cols = feature_columns(feature_set, df.columns)
    pool = df[df["season"].isin(SEASONS)].reset_index(drop=True)
    parts = []
    for s in SEASONS:
        tr = pool[pool["season"] != s]
        te = pool[(pool["season"] == s) & pool["game_group"].isin(["ncaa", "conf_tourney"])]
        model = LogisticModel(C=1.0).fit(tr[cols], tr["a_win"].to_numpy(), training_weights(tr, "equal"))
        parts.append(pd.DataFrame({"game_id": te["game_id"].to_numpy(), "season": s,
                                   "loss": metrics.game_log_loss(model.predict_proba(te[cols]), te["a_win"].to_numpy())}))
    return pd.concat(parts, ignore_index=True)


def compare(candidate: pd.DataFrame, default: pd.DataFrame) -> dict:
    m = candidate.merge(default, on=["game_id", "season"], suffixes=("_c", "_d"))
    return metrics.keep_complex(m["loss_c"], m["loss_d"], m["season"].to_numpy(), n=5000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    raw = raw_data(args.refresh)

    rows = []
    t0 = time.time()
    base_shrink = losses(matrix(raw, DEFAULTS["shrink_games"], DEFAULTS["elo_k"], DEFAULTS["elo_carry"]), "rating_core")
    for s in SHRINK_GRID:
        l = base_shrink if s == DEFAULTS["shrink_games"] else losses(matrix(raw, s, DEFAULTS["elo_k"], DEFAULTS["elo_carry"]), "rating_core")
        r = compare(l, base_shrink)
        rows.append({"setting": "shrink_games", "value": s, "metric_model": "logistic on diff_adj_em", "log_loss": l["loss"].mean(), **r})
        print(f"shrink_games={s:<5} log_loss {l['loss'].mean():.4f}  diff vs default {r['mean_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  keep={r['keep']}  ({time.time() - t0:.0f}s)", flush=True)

    base_elo = losses(matrix(raw, DEFAULTS["shrink_games"], DEFAULTS["elo_k"], DEFAULTS["elo_carry"]), "elo_core")
    for k, c in ELO_GRID:
        isdef = (k, c) == (DEFAULTS["elo_k"], DEFAULTS["elo_carry"])
        l = base_elo if isdef else losses(matrix(raw, DEFAULTS["shrink_games"], k, c), "elo_core")
        r = compare(l, base_elo)
        rows.append({"setting": "elo_k,carry", "value": f"{k:g},{c:g}", "metric_model": "logistic on diff_elo", "log_loss": l["loss"].mean(), **r})
        print(f"elo k={k:<4g} carry={c:<4g} log_loss {l['loss'].mean():.4f}  diff vs default {r['mean_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  keep={r['keep']}  ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    chosen = out[out["keep"] & (out["mean_diff"] < 0)]
    print("\nsettings that pass the decision rule against the default:")
    print(chosen[["setting", "value", "mean_diff", "lo", "hi"]].round(4).to_string(index=False) if len(chosen) else "none. Keep the defaults.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
