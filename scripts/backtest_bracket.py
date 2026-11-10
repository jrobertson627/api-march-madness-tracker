"""Backtest the bracket simulator on past tournaments.

    python scripts/backtest_bracket.py            # select: 2022 to 2025, choose tau, write artifacts/bracket/tau.json
    python scripts/backtest_bracket.py --locked   # score 2026 once with that tau (refuses a second run)

Each season's win probabilities come from a model trained on the other 2022 to 2025 seasons, never on the season
being scored and never on 2026. The locked run trains on 2022 to 2025 and scores 2026.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bracket import pick, upsets  # noqa: E402
from bracket.backtest import POOL, TAU_GRID, Tournament, build_tournament, choose_tau, event_scores, flag_outcomes, flag_summary, levels_for, strategy_scores  # noqa: E402
from bracket.probs import pair_matrix, seed_fallback, seed_matrix  # noqa: E402
from bracket.simulate import exact_levels  # noqa: E402
from data.db import get_pool  # noqa: E402
from data.snapshots import load_team_meta  # noqa: E402
from ml.dataset import MATRIX_PATH, data_stamp, load_matchups  # noqa: E402
from ml.experiment import fit_components  # noqa: E402
from ml.predictor import Predictor  # noqa: E402
from ml.sets import FINAL_NAME, find  # noqa: E402

OUT = ROOT / "artifacts" / "bracket"
CACHE = OUT / "cache"
TEAM_ASOF = ROOT / "artifacts" / "features" / "team_asof_bracket.csv"
ALL_SEASONS = (2022, 2023, 2024, 2025, 2026)
N_SIMS = 100_000
LOCKED_SEASON = 2026
MAIN, EFF = FINAL_NAME, "B4 efficiency gap, all rows"


def load_meta() -> pd.DataFrame:
    path = CACHE / "meta.csv"
    if path.exists():
        return pd.read_csv(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_team_meta(list(ALL_SEASONS))
    get_pool().close()
    meta.to_csv(path, index=False)
    return meta


def load_tournaments(matrix_hash: str) -> dict:
    m = load_matchups(MATRIX_PATH, trainable_only=False)
    ncaa = m[m["game_group"] == "ncaa"].copy()
    ncaa["winner"] = np.where(ncaa["a_win"] == 1, ncaa["team_a"], ncaa["team_b"])
    ncaa["round"] = ncaa["round"].astype(int)
    meta = load_meta()
    out = {}
    for s in ALL_SEASONS:
        teams = meta[(meta["season"] == s) & meta["ncaa_seed"].notna()][["team", "ncaa_seed", "ncaa_region"]].rename(columns={"ncaa_seed": "seed", "ncaa_region": "region"})
        out[s] = build_tournament(s, teams, ncaa[ncaa["season"] == s][["team_a", "team_b", "round", "winner"]])
    return out, ncaa


def fitted_predictor(name: str, df: pd.DataFrame, train_seasons, matrix_hash: str) -> Predictor:
    """Fit a Phase 2 configuration on the given seasons only (cached on disk)."""
    cfg = find(name)
    key = f"{cfg.run_id()}_{matrix_hash}_{'-'.join(map(str, train_seasons))}.joblib"
    path = CACHE / key
    CACHE.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return Predictor(joblib.load(path), {})
    idx = np.flatnonzero(df["season"].isin(list(train_seasons)).to_numpy())
    fitted = fit_components(cfg, df, idx)
    joblib.dump(fitted, path)
    return Predictor(fitted, {})


def season_probabilities(season: int, t: Tournament, df, ta, ncaa, matrix_hash: str, train_seasons) -> dict:
    train = df[df["season"].isin(list(train_seasons))]
    fallback = seed_fallback(train)
    as_of = ta[ta["season"] == season]["as_of"].max()
    out = {"as_of": as_of}
    P, missing = pair_matrix(fitted_predictor(MAIN, df, train_seasons, matrix_hash), ta, t.bracket, season, as_of, fallback)
    out["main"], out["missing"] = P, missing
    out["eff"], _ = pair_matrix(fitted_predictor(EFF, df, train_seasons, matrix_hash), ta, t.bracket, season, as_of, fallback)
    out["seed"] = seed_matrix(t.bracket, fallback)
    out["coin"] = np.full_like(P, 0.5)
    out["history"] = upsets.seed_history(ncaa[(ncaa["season"].isin(list(train_seasons)))])
    return out


def select(df, ta, tournaments, ncaa, matrix_hash) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores, tau_loss, picks_rows, flag_rows, lines = [], {}, [], [], []
    probs = {}
    for s in POOL:
        train_seasons = [x for x in POOL if x != s]
        t = tournaments[s]
        pr = season_probabilities(s, t, df, ta, ncaa, matrix_hash, train_seasons)
        probs[s] = pr
        lines.append(f"{s}: trained on {train_seasons}; teams without features (seed fallback): {pr['missing'] or 'none'}")
        for name in ("coin", "seed", "eff"):
            scores.append({"season": s, "method": name, "tau": 0.0, **event_scores(exact_levels(t.bracket, pr[name]).reach(), t.realized)})
        tau_loss[s] = {}
        for tau in TAU_GRID:
            lv = levels_for(t.bracket, pr["main"], tau, N_SIMS, seed=s)
            sc = event_scores(lv.reach(), t.realized)
            tau_loss[s][tau] = sc["log_loss"]
            scores.append({"season": s, "method": "model", "tau": tau, **sc})
    loss = pd.DataFrame(tau_loss).T
    tau = choose_tau(loss)
    lines.append(f"tau chosen on {list(POOL)}: {tau}")

    for s in POOL:
        t, pr = tournaments[s], probs[s]
        lv = levels_for(t.bracket, pr["main"], tau["tau"], N_SIMS, seed=s)
        picks_rows.append(strategy_scores(t, pr["main"], lv))
        flags = upsets.first_round_table(t.bracket, exact_levels(t.bracket, pr["main"]), pr["history"])
        flag_rows.append(flag_outcomes(t, flags))

    sc = pd.DataFrame(scores)
    picks_df, flags_df = pd.concat(picks_rows, ignore_index=True), pd.concat(flag_rows, ignore_index=True)
    sc.to_csv(OUT / "backtest_select_scores.csv", index=False)
    picks_df.to_csv(OUT / "backtest_select_picks.csv", index=False)
    flags_df.to_csv(OUT / "backtest_select_flags.csv", index=False)
    (OUT / "tau.json").write_text(json.dumps({**tau, "seasons": list(POOL), "matrix_hash": matrix_hash, "n_sims": N_SIMS}, indent=1))

    print("\n".join(lines))
    pooled = sc[sc["tau"] == 0.0].groupby("method")[["log_loss", "brier"]].mean().round(4)
    print("\nadvancement log loss and Brier, mean of the four seasons (model at tau = 0):")
    print(pooled.to_string())
    print("\nadvancement log loss by season:")
    print(sc[sc["tau"] == 0.0].pivot(index="season", columns="method", values="log_loss").round(4).to_string())
    print("\nmodel advancement log loss by season and tau:")
    print(loss.round(4).to_string())
    print("\nrealized points by strategy (10-20-40-80-160-320 scoring):")
    print(picks_df.pivot(index="season", columns="strategy", values="realized_points").to_string())
    print("mean:", picks_df.groupby("strategy")["realized_points"].mean().round(1).to_dict())
    print("model's own expected points:", picks_df.groupby("strategy")["expected_points"].mean().round(1).to_dict())
    print("\nfirst-round upset flags (four seasons, 128 games):")
    print(flag_summary(flags_df).round(3).to_string(index=False))


def locked(df, ta, tournaments, ncaa, matrix_hash) -> None:
    guard = OUT / "locked_2026.json"
    if guard.exists():
        sys.exit(f"the locked season was already scored: {guard}")
    tau = json.loads((OUT / "tau.json").read_text())["tau"]
    s, t = LOCKED_SEASON, tournaments[LOCKED_SEASON]
    pr = season_probabilities(s, t, df, ta, ncaa, matrix_hash, POOL)
    rows = []
    for name in ("coin", "seed", "eff"):
        rows.append({"method": name, **event_scores(exact_levels(t.bracket, pr[name]).reach(), t.realized)})
    lv = levels_for(t.bracket, pr["main"], tau, N_SIMS, seed=s)
    rows.append({"method": f"model (tau {tau})", **event_scores(lv.reach(), t.realized)})
    table = pd.DataFrame(rows)
    picks_df = strategy_scores(t, pr["main"], lv)
    flags = flag_outcomes(t, upsets.first_round_table(t.bracket, exact_levels(t.bracket, pr["main"]), pr["history"]))
    print(f"2026 (locked): trained on {list(POOL)}, tau {tau}, teams without features: {pr['missing'] or 'none'}")
    print(table.round(4).to_string(index=False))
    print(picks_df.round(1).to_string(index=False))
    print(flag_summary(flags).round(3).to_string(index=False))
    champ = t.bracket.teams[t.actual[6][0]]
    reach = lv.reach()
    print(f"title chance the model gave the real champion ({champ.split('/schools/')[1].split('/')[0]}): {reach[t.bracket.index[champ], -1]:.3f}")
    guard.write_text(json.dumps({"season": s, "tau": tau, "scores": table.to_dict("records"), "picks": picks_df.to_dict("records")}, default=float, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true")
    args = ap.parse_args()
    df = load_matchups(MATRIX_PATH)
    stamp = data_stamp(MATRIX_PATH, df)["hash"]
    ta = pd.read_csv(TEAM_ASOF, parse_dates=["as_of"])
    tournaments, ncaa = load_tournaments(stamp)
    (locked if args.locked else select)(df, ta, tournaments, ncaa, stamp)


if __name__ == "__main__":
    main()
