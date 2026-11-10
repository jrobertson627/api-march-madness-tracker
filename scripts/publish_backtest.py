"""Publish the backtest report that the /v1/backtest route serves.

    python scripts/publish_backtest.py --dry-run                 # build and check, write nothing
    python scripts/publish_backtest.py                           # write to the local file store (artifacts/predictions)
    python scripts/publish_backtest.py --to postgres             # needs PUBLISH_DATABASE_URL (or DATABASE_URL) and migration 003

Reads the result files from Phases 2 and 3 (no models are run, nothing is recomputed except the paired bootstrap):
  artifacts/models/experiments.csv, oof_predictions.csv      per-game scores and the games behind them
  artifacts/bracket/backtest_select_scores.csv, _picks.csv, _flags.csv   2022 to 2025 advancement scores, picks, upset flags
  artifacts/bracket/locked_2026.json                          2026, scored once
  artifacts/models/team_names.csv, artifacts/models/v1/manifest.json
Each publish adds one immutable report; the API serves the newest.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from api import backtest as bt  # noqa: E402
from api.store import FileStore, PostgresStore  # noqa: E402

MODELS = ROOT / "artifacts" / "models"
BRACKET = ROOT / "artifacts" / "bracket"

SELECT_SPECS = [("coin flip", "B0 coin flip (select seasons)"), ("seed", "B1 seed logistic (select seasons)"), ("elo", "B3 elo probability (select seasons)"),
                ("efficiency gap", "B4 efficiency gap, all rows (select seasons)"), ("model", "M6 hgb full monotone")]
LOCKED_SPECS = [("coin flip", "B0 coin flip"), ("seed", "B1 seed logistic"), ("elo", "B3 elo probability"), ("efficiency gap", "B4 efficiency gap, all rows")]
FINAL = "FINAL boosted full monotone + ridge scores"
SELECT_PAIRS = [("M6 hgb full monotone", "B4 efficiency gap, all rows (select seasons)", "model", "efficiency gap"),
                ("M6 hgb full monotone", "B1 seed logistic (select seasons)", "model", "seed")]
LOCKED_PAIRS = [(FINAL, "B4 efficiency gap, all rows", "model", "efficiency gap"), (FINAL, "B1 seed logistic", "model", "seed")]


def paired_rows(oof, run_ids, pairs):
    from ml.report import paired

    out = []
    for cand, base, cand_label, base_label in pairs:
        try:
            out.append(bt.paired_row(paired(oof, base, cand, run_ids), cand_label, base_label))
        except Exception as e:  # a comparison that cannot be made is left out and said so
            print(f"  skipped paired comparison {cand_label} vs {base_label}: {type(e).__name__}: {e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", choices=["files", "postgres"], default="files")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "predictions"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = pd.read_csv(MODELS / "experiments.csv")
    oof = pd.read_csv(MODELS / "oof_predictions.csv")
    run_ids = exp.drop_duplicates("name").set_index("name")["run_id"].to_dict()
    scores = pd.read_csv(BRACKET / "backtest_select_scores.csv")
    picks = pd.read_csv(BRACKET / "backtest_select_picks.csv")
    flags = pd.read_csv(BRACKET / "backtest_select_flags.csv")
    locked_raw = json.loads((BRACKET / "locked_2026.json").read_text(encoding="utf-8"))
    names_df = pd.read_csv(MODELS / "team_names.csv")
    names = dict(zip(names_df["team"], names_df["display_name"]))
    manifest = json.loads((MODELS / "v1" / "manifest.json").read_text(encoding="utf-8"))
    model_id = f"v1-{manifest['created'][:10]}"
    data_hash = manifest["data_stamp"].get("hash", "")

    seasons = sorted(int(s) for s in scores["season"].unique())
    flag_rows, flag_summary = bt.upset_flags(flags, names)
    locked_games = [r for r in bt.game_rows(exp, LOCKED_SPECS) if r["season"] == locked_raw["season"]]
    locked_games += bt.game_rows(exp, [("model", FINAL)], protocol="locked")
    locked_games = [r for r in locked_games if r["season"] != "all"]
    locked = {"season": int(locked_raw["season"]), "advancement": bt.locked_advancement_rows(locked_raw), "games_scored": locked_games,
              "picks": bt.pick_rows(locked_raw["picks"]), "paired": paired_rows(oof, run_ids, LOCKED_PAIRS)}
    report = bt.build_report(seasons=seasons, advancement=bt.advancement_rows(scores), games_scored=bt.game_rows(exp, SELECT_SPECS, protocol="loso"),
                             picks=bt.pick_rows(picks), flags=flag_rows, flag_summary=flag_summary,
                             paired=paired_rows(oof, run_ids, SELECT_PAIRS), locked=locked)

    sel, size = report["selection"], len(json.dumps(report))
    print(f"report for {model_id} (data {data_hash}): {size / 1024:.0f} KB, seasons {sel['seasons']}, locked {locked['season']}")
    print(f"  advancement rows {len(sel['advancement'])}, game rows {len(sel['games_scored'])}, pick rows {len(sel['picks'])}, flagged games {len(sel['upset_flags'])}, paired {len(sel['paired'])} + {len(locked['paired'])}")
    print("  upset summary:", sel["upset_summary"])
    for line in report["limits"]:
        print("  -", line)
    problems = []
    if not sel["advancement"] or not sel["games_scored"]:
        problems.append("selection panel is empty")
    if not locked["advancement"] or not locked["games_scored"]:
        problems.append("2026 panel is empty")
    if size > 300_000:
        problems.append("report is larger than 300 KB")
    if problems:
        sys.exit(f"not publishing: {problems}")
    if args.dry_run:
        print("dry run: nothing written")
        return
    store = FileStore(args.out) if args.to == "files" else PostgresStore()
    print("wrote", store.write_backtest(report, model_id, data_hash))


if __name__ == "__main__":
    main()
