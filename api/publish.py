"""Turn a saved model into a published snapshot: every pair of teams, with scores and factor effects.

Offline only. The web service never imports this module. Feature building and prediction are in `pair_predictions`;
the rest is plain data handling that the tests cover.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from api.ids import slug_of
from api.store import Bundle, ModelVersion, Snapshot

HEADLINE_VALUES = ["adj_em", "adj_off", "adj_def", "adj_tempo", "elo", "sos_elo", "last10_margin", "win_pct", "efg", "returning_min_pct",
                   "coach_seasons_at_school", "seed"]
BYTES_PER_PAIR = 420  # measured after the first publish (166 MB for 393,873 rows, indexes included)


def _num(x, n=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def assemble_pairs(slug_a, slug_b, pred: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """One row per pair. `pred` has p_a_win and, with a score model, margin_hat, total_reg_hat, score_a_hat, score_b_hat.
    `effects` has one column per factor group plus `interaction`."""
    groups = [c for c in effects.columns if c != "p_a_win"]
    factors = [{g: _num(v) for g, v in zip(groups, row)} for row in effects[groups].to_numpy()]
    out = pd.DataFrame({"team_a": list(slug_a), "team_b": list(slug_b), "p_a_win": pred["p_a_win"].to_numpy(float)})
    for dst, src in (("margin_a", "margin_hat"), ("total", "total_reg_hat"), ("score_a", "score_a_hat"), ("score_b", "score_b_hat")):
        out[dst] = pred[src].to_numpy(float) if src in pred else np.nan
    out["factors"] = factors
    return out


def check_pairs(pairs: pd.DataFrame, teams: list[str]) -> list[str]:
    """Problems that must stop a publish. An empty list means the pairs are sound."""
    problems = []
    n = len(teams)
    if len(pairs) != n * (n - 1) // 2:
        problems.append(f"expected {n * (n - 1) // 2} pairs for {n} teams, found {len(pairs)}")
    if pairs[["team_a", "team_b"]].duplicated().any():
        problems.append("duplicate pairs")
    if not (pairs["team_a"] < pairs["team_b"]).all():
        problems.append("team_a must sort before team_b in every pair")
    if not set(pairs["team_a"]) | set(pairs["team_b"]) <= set(teams):
        problems.append("pairs mention teams that are not in the snapshot")
    p = pairs["p_a_win"]
    if p.isna().any() or not ((p > 0) & (p < 1)).all():
        problems.append("probabilities must be strictly between 0 and 1")
    if pairs["factors"].notna().all():
        gap = [abs(sum(f.values()) - (pp - 0.5)) for f, pp in zip(pairs["factors"], p)]
        if max(gap, default=0) > 1e-4:
            problems.append(f"factor effects plus interaction do not equal the move from 50% (largest gap {max(gap):.2e})")
    for c in ("margin_a", "total", "score_a", "score_b"):
        if pairs[c].notna().any() and pairs[c].isna().any():
            problems.append(f"{c} is missing for some pairs")
    return problems


def seed_history_json(history: dict) -> dict:
    return {f"{hi}-{lo}": round(float(v), 4) for (hi, lo), v in history.items()}


def field_table(teams: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Bracket field with slug ids: team, name, seed, region."""
    t = teams.copy()
    t["name"] = t["team"].map(lambda x: names.get(x))
    t["team"] = t["team"].map(slug_of)
    return t[["team", "name", "seed", "region"]]


def team_rows(team_asof_rows: pd.DataFrame, names: dict) -> pd.DataFrame:
    rows = []
    for r in team_asof_rows.to_dict("records"):
        values = {k: _num(r[k], 3) for k in HEADLINE_VALUES if k in r and pd.notna(r[k])}
        rows.append({"team": slug_of(r["team"]), "name": names.get(r["team"]), "values": values})
    return pd.DataFrame(rows)


def pair_predictions(predictor, team_asof: pd.DataFrame, season: int, as_of, explain: bool = True):
    """Predict every pair of D1 teams that have features at `as_of`. Returns (pairs, teams_with_features)."""
    from data.features import build_matchups
    from ml.dataset import add_derived
    from ml.explain import factor_effects

    as_of = pd.Timestamp(as_of)
    ta = team_asof[(team_asof["season"] == season) & (team_asof["as_of"] == as_of)]
    ids = sorted(ta["team"].unique(), key=slug_of)
    pairs = pd.DataFrame(list(itertools.combinations(ids, 2)), columns=["team_a", "team_b"])
    pairs = pairs.assign(season=season, as_of=as_of, game_date=as_of + pd.Timedelta(days=1), game_group="ncaa", loc_a=0.0, overtimes=0)
    rows = add_derived(build_matchups(pairs, team_asof, min_games=0, as_of_mode="bracket"))
    pred = predictor.predict(rows)
    effects = factor_effects(predictor, rows) if explain else pd.DataFrame({"p_a_win": pred["p_a_win"], "interaction": 0.0})
    slugs_a = [slug_of(x) for x in pairs["team_a"]]
    slugs_b = [slug_of(x) for x in pairs["team_b"]]
    return assemble_pairs(slugs_a, slugs_b, pred, effects), ids


def make_bundle(model: ModelVersion, season: int, as_of, kind: str, pairs: pd.DataFrame, teams: pd.DataFrame, field: pd.DataFrame | None,
                final_four, seed_history: dict, notes: list | None = None) -> Bundle:
    in_sample = season in model.training_seasons
    snap = Snapshot(model.id, season, str(pd.Timestamp(as_of).date()), kind, in_sample, len(teams),
                    {"seed_history": seed_history_json(seed_history), "notes": notes or []})
    bracket = None if field is None else {"teams": field, "final_four": [list(p) for p in final_four]}
    return Bundle(model, snap, pairs, teams, bracket)


def size_report(bundle: Bundle) -> str:
    n = len(bundle.pairs)
    return f"{bundle.snapshot.season} {bundle.model.id}: {n:,} pairs, {len(bundle.teams)} teams, about {n * BYTES_PER_PAIR / 1e6:.1f} MB, in_sample={bundle.snapshot.in_sample}"
