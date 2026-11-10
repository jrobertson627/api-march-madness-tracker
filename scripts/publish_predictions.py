"""Publish model predictions for a season: every pair of teams, with scores and factor effects.

    python scripts/publish_predictions.py --backtest --seasons 2022 2023 2024 2025 2026      # out-of-sample history
    python scripts/publish_predictions.py --live artifacts/models/v1 --seasons 2026 --activate
    python scripts/publish_predictions.py --backtest --seasons 2026 --dry-run                # checks, writes nothing
    python scripts/publish_predictions.py --live artifacts/models/v1 --seasons 2026 --to postgres   # needs PUBLISH_DATABASE_URL

--backtest uses, for each season, a model trained on the other 2022 to 2025 seasons (never on the season itself, and
never on 2026 except when 2026 is the season), so those snapshots are out-of-sample. --live uses a saved model and
marks a snapshot in-sample when the model trained on that season.

Needs the matrices from scripts/build_features.py (bracket mode), artifacts/models/team_names.csv and
artifacts/bracket/cache/meta.csv (made by scripts/backtest_bracket.py).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from api import publish  # noqa: E402
from api.store import FileStore, ModelVersion, PostgresStore  # noqa: E402
from bracket import upsets  # noqa: E402
from bracket.structure import Bracket  # noqa: E402
from ml.dataset import MATRIX_PATH, data_stamp, load_matchups  # noqa: E402
from ml.fitcache import fitted_predictor  # noqa: E402
from ml.predictor import Predictor  # noqa: E402
from ml.sets import FINAL_NAME, find  # noqa: E402

TEAM_ASOF = ROOT / "artifacts" / "features" / "team_asof_bracket.csv"
META = ROOT / "artifacts" / "bracket" / "cache" / "meta.csv"
NAMES = ROOT / "artifacts" / "models" / "team_names.csv"
POOL = (2022, 2023, 2024, 2025)


def season_bracket(season, meta, ncaa, names):
    """The bracket field, Final Four pairing and played results of a season, from the matrix files. Slugs are used in the output."""
    field_ids = meta[(meta["season"] == season) & meta["ncaa_seed"].notna()][["team", "ncaa_seed", "ncaa_region"]].rename(columns={"ncaa_seed": "seed", "ncaa_region": "region"})
    results = ncaa[ncaa["season"] == season][["team_a", "team_b", "round", "winner"]]
    bracket = Bracket.from_results(field_ids, results)
    field = publish.field_table(pd.DataFrame({"team": bracket.teams, "seed": bracket.seed, "region": bracket.region}), names)
    final_four = ((bracket.regions[0], bracket.regions[1]), (bracket.regions[2], bracket.regions[3]))
    return bracket, field, final_four, publish.results_json(results, bracket.teams)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--backtest", action="store_true")
    g.add_argument("--live", metavar="MODEL_DIR")
    ap.add_argument("--results-only", action="store_true", help="only write the played tournament results of each season (no predictions are made)")
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--to", choices=["files", "postgres"], default="files")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "predictions"))
    ap.add_argument("--activate", action="store_true", help="make the model the active one")
    ap.add_argument("--replace", action="store_true", help="overwrite a snapshot that already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.results_only and not (args.backtest or args.live):
        ap.error("choose --backtest, --live MODEL_DIR or --results-only")

    df = load_matchups(MATRIX_PATH)
    all_games = load_matchups(MATRIX_PATH, trainable_only=False)
    ncaa = all_games[all_games["game_group"] == "ncaa"].copy()
    ncaa["winner"] = np.where(ncaa["a_win"] == 1, ncaa["team_a"], ncaa["team_b"])
    ncaa["round"] = ncaa["round"].astype(int)
    stamp = data_stamp(MATRIX_PATH, df)["hash"]
    ta = pd.read_csv(TEAM_ASOF, parse_dates=["as_of"])
    meta = pd.read_csv(META)
    names = dict(zip(pd.read_csv(NAMES)["team"], pd.read_csv(NAMES)["display_name"]))
    store = FileStore(args.out) if args.to == "files" else PostgresStore()

    if args.results_only:
        for season in sorted(args.seasons):
            _, _, _, played = season_bracket(season, meta, ncaa, names)
            print(f"{season}: {len(played)} played games", "(dry run)" if args.dry_run else "")
            if not args.dry_run:
                print("  wrote", store.write_results(season, played))
        return

    live = Predictor.load(args.live) if args.live else None
    for season in sorted(args.seasons):
        if live is not None:
            m = live.manifest
            model = ModelVersion(f"{Path(args.live).name}-{m['created'][:10]}", m["name"], m["run_id"], "live", list(m["training_seasons"]), m["data_stamp"].get("hash", stamp),
                                 m["created"], {"held_out": m.get("cross_validated_metrics", [])}, {k: m[k] for k in ("config", "chosen_hyperparameters", "packages", "limits") if k in m})
            predictor = live
        else:
            train = [s for s in POOL if s != season]
            cfg = find(FINAL_NAME)
            predictor = fitted_predictor(FINAL_NAME, df, train, stamp)
            model = ModelVersion(f"backtest-{season}-{cfg.run_id()[:6]}", FINAL_NAME, cfg.run_id(), "backtest", train, stamp, "", {}, {"config": cfg.name})
        as_of = ta[ta["season"] == season]["as_of"].max()
        pairs, ids = publish.pair_predictions(predictor, ta, season, as_of)
        slugs = [publish.slug_of(i) for i in ids]
        problems = publish.check_pairs(pairs, slugs)
        if problems:
            sys.exit(f"{season}: not publishing, problems found: {problems}")

        bracket, field, final_four, played = season_bracket(season, meta, ncaa, names)
        history = upsets.seed_history(ncaa[ncaa["season"] != season])
        ta_rows = ta[(ta["season"] == season) & (ta["as_of"] == as_of)]
        bundle = publish.make_bundle(model, season, as_of, "selection_sunday", pairs, publish.team_rows(ta_rows, names), field, final_four, history, results=played)
        print(publish.size_report(bundle), "| checks passed")
        if args.dry_run:
            continue
        info = store.write_snapshot(bundle, replace=args.replace, activate=args.activate and live is not None)
        print("  wrote", info)


if __name__ == "__main__":
    main()
