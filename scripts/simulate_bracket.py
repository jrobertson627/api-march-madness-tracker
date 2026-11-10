"""Simulate a tournament and write odds, picks and upset flags.

    python scripts/simulate_bracket.py --season 2026 --model artifacts/models/v1
    python scripts/simulate_bracket.py --season 2027 --bracket field.csv --final-four "East-West,South-Midwest" \
        --features artifacts/features/team_asof_bracket.csv --as-of 2027-03-14 --model artifacts/models/v1
    python scripts/simulate_bracket.py --season 2026 --model artifacts/models/v1 --results played.csv   # update odds

The bracket file has columns team, seed and region (team is the Sports Reference id used in the matrix). For a season
already in the database the bracket comes from team_seasons. Output goes to artifacts/bracket/<season>/.

A saved model that was trained on the season being simulated gives in-sample odds; the report says so.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bracket import pick, upsets  # noqa: E402
from bracket.outputs import write_outputs  # noqa: E402
from bracket.probs import pair_matrix, seed_fallback  # noqa: E402
from bracket.simulate import exact_levels, reach_table, simulate  # noqa: E402
from bracket.structure import Bracket  # noqa: E402
from ml.dataset import MATRIX_PATH, data_stamp, load_matchups  # noqa: E402
from ml.predictor import Predictor  # noqa: E402

DEFAULT_FEATURES = ROOT / "artifacts" / "features" / "team_asof_bracket.csv"


def parse_final_four(text):
    if not text:
        return None
    pairs = tuple(tuple(x.strip() for x in part.split("-")) for part in text.split(","))
    if len(pairs) != 2 or any(len(p) != 2 for p in pairs):
        sys.exit('--final-four looks like "East-West,South-Midwest"')
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--model", required=True, help="a saved Predictor folder")
    ap.add_argument("--bracket", help="CSV with team, seed, region (default: from team_seasons)")
    ap.add_argument("--final-four", help='the two semifinals, for example "East-West,South-Midwest"')
    ap.add_argument("--features", default=str(DEFAULT_FEATURES), help="team_asof table built in bracket mode")
    ap.add_argument("--as-of", help="Selection Sunday (default: the latest as_of for the season in the features file)")
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--tau", type=float, default=0.0, help="team-strength uncertainty; 0 means independent games")
    ap.add_argument("--scoring", default="10,20,40,80,160,320")
    ap.add_argument("--results", help="CSV of games already played: team_a, team_b, winner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scoring = pick.Scoring(tuple(int(x) for x in args.scoring.split(",")))
    predictor = Predictor.load(args.model)
    ta = pd.read_csv(args.features, parse_dates=["as_of"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else ta[ta["season"] == args.season]["as_of"].max()

    df = load_matchups(MATRIX_PATH)
    history_games = load_matchups(MATRIX_PATH, trainable_only=False)
    history_games = history_games[(history_games["game_group"] == "ncaa") & (history_games["season"] != args.season)]
    notes = []
    if args.season in predictor.manifest.get("training_seasons", []):
        notes.append(f"WARNING: the model was trained on {args.season}, so these odds are in-sample")

    if args.bracket:
        teams = pd.read_csv(args.bracket)
        bracket = Bracket.from_teams(teams, parse_final_four(args.final_four))
    else:
        from data.db import get_pool
        from data.snapshots import load_team_meta

        meta = load_team_meta([args.season])
        get_pool().close()
        teams = meta[meta["ncaa_seed"].notna()][["team", "ncaa_seed", "ncaa_region"]].rename(columns={"ncaa_seed": "seed", "ncaa_region": "region"})
        played = load_matchups(MATRIX_PATH, trainable_only=False)
        played = played[(played["game_group"] == "ncaa") & (played["season"] == args.season)].assign(winner=lambda d: np.where(d["a_win"] == 1, d["team_a"], d["team_b"]))
        results = played[["team_a", "team_b", "round", "winner"]].assign(round=lambda d: d["round"].astype(int))
        final_four = parse_final_four(args.final_four)
        bracket = Bracket.from_teams(teams, final_four) if final_four else Bracket.from_results(teams, results)

    fixed = pd.read_csv(args.results) if args.results else None
    fallback = seed_fallback(df[df["season"] != args.season])
    P, missing = pair_matrix(predictor, ta, bracket, args.season, as_of, fallback)
    levels = exact_levels(bracket, P, fixed) if args.tau == 0 else simulate(bracket, P, args.sims, args.tau, fixed, args.seed)
    if missing:
        notes.append(f"seed-only fallback used for {len(missing)} team(s) without features: {missing}")

    picks = {"best expected points": pick.best_expected_bracket(bracket, levels, scoring), "model favorite": pick.favorite_bracket(bracket, P), "seed chalk": pick.seed_bracket(bracket)}
    first_round = upsets.first_round_table(bracket, exact_levels(bracket, P), upsets.seed_history(history_games)) if bracket.n_rounds == 6 else None

    table = reach_table(bracket, levels)
    show = table.head(12)[["team", "seed", "region", "p_final_four", "p_championship_game", "p_champion"]].assign(team=lambda d: d["team"].str.extract(r"schools/([^/]+)/")[0])
    lines = [f"season {args.season}, as of {as_of.date()}, {len(bracket.teams)} teams, {len(bracket.playin_slots)} First Four games",
             f"odds: {'exact (independent games)' if args.tau == 0 else f'simulated, {args.sims:,} brackets, tau {args.tau}'}"
             + (f", with {len(fixed)} games already played" if fixed is not None else "")]
    lines += notes + ["", "most likely champions:", show.round(3).to_string(index=False), "",
                      "expected points (scoring " + ", ".join(map(str, scoring.points)) + "): "
                      + ", ".join(f"{n} {pick.expected_points(p, levels, scoring):.0f}" for n, p in picks.items())]
    if first_round is not None:
        lines += ["", "first-round games the model flags:"]
        fl = first_round[first_round["flagged"]][["region", "favorite_seed", "underdog_seed", "favorite", "underdog", "p_underdog_model", "p_underdog_history"]]
        lines += [fl.assign(favorite=fl.favorite.str.extract(r"schools/([^/]+)/")[0], underdog=fl.underdog.str.extract(r"schools/([^/]+)/")[0]).round(3).to_string(index=False)]
    report = "\n".join(lines)
    meta_out = {"season": args.season, "as_of": str(as_of.date()), "model_run_id": predictor.manifest.get("run_id"), "model_training_seasons": predictor.manifest.get("training_seasons"),
                "matrix_hash": data_stamp(MATRIX_PATH)["hash"], "sims": None if args.tau == 0 else args.sims, "tau": args.tau, "scoring": list(scoring.points),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "notes": notes}
    out = Path(args.out) if args.out else ROOT / "artifacts" / "bracket" / str(args.season)
    paths = write_outputs(out, args.season, bracket, levels, picks, first_round, meta_out, report)
    print(report)
    print("\nwrote", ", ".join(p.name for p in paths), "to", out)


if __name__ == "__main__":
    main()
