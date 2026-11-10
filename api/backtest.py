"""The backtest report: how the model did on past tournaments, assembled from the Phase 2 and 3 result files.

Pure data handling (pandas only, no scikit-learn), so the tests run on small invented tables. `scripts/publish_backtest.py`
reads the real files, calls these functions and writes one immutable report through a Store.

Two parts are kept apart on purpose:
  selection  the seasons used to choose the design (2022 to 2025)
  locked     the season scored once after the design was chosen (2026), the only unbiased comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.ids import slug_of

METHOD_LABELS = {"coin": "coin flip", "seed": "seed", "eff": "efficiency gap", "model": "model"}


def _num(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _method(raw: str) -> str | None:
    """'model (tau 0.0)' and 'model' are the model; 'coin', 'seed' and 'eff' are the baselines."""
    key = str(raw).split(" ")[0]
    return METHOD_LABELS.get(key)


def advancement_rows(scores: pd.DataFrame, tau: float = 0.0) -> list[dict]:
    """Log loss and Brier score of each method's odds that every team reaches every round, per season and pooled.
    `scores` has season, method, tau, log_loss, brier, events. Only the rows for the chosen `tau` are used."""
    s = scores[np.isclose(scores["tau"].astype(float), tau)] if "tau" in scores else scores
    out = []
    for r in s.itertuples():
        label = _method(r.method)
        if label:
            out.append({"season": int(r.season), "method": label, "log_loss": _num(r.log_loss), "brier": _num(r.brier), "events": int(r.events)})
    pooled = {}
    for r in out:
        p = pooled.setdefault(r["method"], [0.0, 0.0, 0])
        p[0] += r["log_loss"] * r["events"]
        p[1] += r["brier"] * r["events"]
        p[2] += r["events"]
    for label, (ll, br, n) in pooled.items():
        out.append({"season": "all", "method": label, "log_loss": _num(ll / n), "brier": _num(br / n), "events": n})
    return out


def locked_advancement_rows(locked: dict) -> list[dict]:
    season = int(locked["season"])
    out = []
    for r in locked["scores"]:
        label = _method(r["method"])
        if label:
            out.append({"season": season, "method": label, "log_loss": _num(r["log_loss"]), "brier": _num(r["brier"]), "events": int(r["events"])})
    return out


def pick_rows(picks: pd.DataFrame | list) -> list[dict]:
    rows = picks.to_dict("records") if isinstance(picks, pd.DataFrame) else list(picks)
    return [{"season": int(r["season"]), "strategy": r["strategy"], "realized": _num(r["realized_points"], 1), "expected": _num(r["expected_points"], 1)} for r in rows]


def game_rows(exp: pd.DataFrame, specs: list[tuple[str, str]], protocol: str | None = None) -> list[dict]:
    """Per-game scores from the experiments table. `specs` is [(label, run name)]. Seasons come from the table; the
    row for all tested seasons together is returned with season 'all'."""
    out = []
    for label, name in specs:
        e = exp[(exp["name"] == name) & (exp["mode"] == "game")]
        if protocol:
            e = e[e["protocol"] == protocol]
        for r in e.itertuples():
            season = str(r.season)
            out.append({"season": season if season == "all" else int(season), "method": label, "n": int(r.n), "log_loss": _num(r.log_loss),
                        "brier": _num(r.brier), "accuracy": _num(r.accuracy)})
    return out


def paired_row(result: dict, candidate: str, baseline: str) -> dict:
    """One paired comparison from ml.metrics.keep_complex. A negative `diff` means the candidate had the lower log loss."""
    return {"candidate": candidate, "baseline": baseline, "games": int(result["games"]), "diff": _num(result["mean_diff"]), "low": _num(result["lo"]),
            "high": _num(result["hi"]), "seasons_ahead": int(result["season_wins"]), "seasons": int(result["seasons"]),
            "interval_excludes_zero": bool(result["lo"] > 0 or result["hi"] < 0)}


def upset_flags(flags: pd.DataFrame, names: dict) -> tuple[list[dict], dict]:
    """First-round games with the model's upset signals and what happened. Returns (rows, summary)."""
    rows = []
    for r in flags.itertuples():
        fav, dog = slug_of(r.favorite), slug_of(r.underdog)
        rows.append({"season": int(r.season), "region": r.region, "favorite": fav, "favorite_name": names.get(r.favorite),
                     "underdog": dog, "underdog_name": names.get(r.underdog), "favorite_seed": int(r.favorite_seed),
                     "underdog_seed": int(r.underdog_seed), "p_underdog_model": _num(r.p_underdog_model),
                     "model_picks_upset": bool(r.model_picks_upset), "model_above_history": bool(r.model_above_history), "upset": bool(r.upset)})

    def share(mask):
        return {"games": int(mask.sum()), "upsets": int(flags.loc[mask, "upset"].astype(bool).sum())}

    picked = flags["model_picks_upset"].astype(bool)
    above = flags["model_above_history"].astype(bool)
    flagged = flags["flagged"].astype(bool)
    summary = {"all_games": share(pd.Series(True, index=flags.index)), "model_picks_upset": share(picked), "model_above_history": share(above), "no_flag": share(~flagged)}
    return rows, summary


def limits_text(selection: dict, locked: dict | None, flag_summary: dict) -> list[str]:
    sel_n = max((r["n"] for r in selection["games_scored"] if r["season"] == "all" and r["method"] == "model"), default=None)
    lines = []
    if sel_n:
        lines.append(f"{sel_n} tournament games in {selection['seasons'][0]} to {selection['seasons'][-1]} were used to choose the design.")
    if locked:
        lock_n = max((r["n"] for r in locked["games_scored"] if r["method"] == "model"), default=None)
        if lock_n:
            lines.append(f"{lock_n} games in {locked['season']} were scored once after the design was chosen. That is the only comparison the design choice did not influence.")
    pairs = selection["paired"] + (locked["paired"] if locked else [])
    if pairs:
        inside = sum(not p["interval_excludes_zero"] for p in pairs)
        if inside == len(pairs):
            lines.append("Every paired comparison below has an interval that includes zero, so the differences could be noise.")
        else:
            lines.append(f"{inside} of {len(pairs)} paired comparisons below have an interval that includes zero, so those differences could be noise. The others exclude zero.")
    s = flag_summary
    lines.append(f"Games where the model picked the upset were upsets {s['model_picks_upset']['upsets']} of {s['model_picks_upset']['games']} times; "
                 f"games with no flag, {s['no_flag']['upsets']} of {s['no_flag']['games']}. The 'above seed history' flag was right {s['model_above_history']['upsets']} of "
                 f"{s['model_above_history']['games']} times and is not supported by this evidence.")
    lines.append("Lower log loss and Brier score are better. A coin flip scores 0.693 per game.")
    lines.append("Probabilities are for a neutral court, with team ratings frozen at Selection Sunday and games treated as independent.")
    return lines


def build_report(*, seasons: list[int], advancement: list[dict], games_scored: list[dict], picks: list[dict], flags: list[dict], flag_summary: dict,
                 paired: list[dict], locked: dict | None = None) -> dict:
    """Put the parts together. `locked` (optional) is {season, advancement, games_scored, picks, paired}."""
    selection = {"seasons": list(seasons), "advancement": advancement, "games_scored": games_scored, "picks": picks, "upset_flags": flags,
                 "upset_summary": flag_summary, "paired": paired}
    report = {"selection": selection, "locked": locked}
    report["limits"] = limits_text(selection, locked, flag_summary)
    return report
