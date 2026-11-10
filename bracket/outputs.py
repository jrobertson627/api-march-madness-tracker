"""Turn simulation results into tables and files for people and for the API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bracket.pick import Picks, Scoring
from bracket.simulate import Levels, reach_table
from bracket.structure import Bracket

ROUND_NAMES = {1: "round of 64", 2: "round of 32", 3: "sweet 16", 4: "elite 8", 5: "final four", 6: "championship"}


def picks_table(bracket: Bracket, picks: Picks, levels: Levels) -> pd.DataFrame:
    """One row per game: who plays (according to the picks), who is picked, and the chance the pick is right."""
    rows = []
    for r in range(1, len(picks.winners)):
        prev = picks.winners[r - 1]
        for j, w in enumerate(picks.winners[r]):
            a, b = prev[2 * j], prev[2 * j + 1]
            rows.append({"round": r, "round_name": ROUND_NAMES.get(r, f"round {r}"), "game": j,
                         "team_a": bracket.teams[a], "team_b": bracket.teams[b], "pick": bracket.teams[w],
                         "pick_seed": bracket.seed[w], "p_pick_wins_game": float(levels.arrays[r][j, w])})
    return pd.DataFrame(rows)


def bracket_json(bracket: Bracket, levels: Levels, picks: dict[str, Picks], meta: dict) -> dict:
    """Everything a page or an API needs, in one object."""
    table = reach_table(bracket, levels)
    return {
        "meta": meta,
        "teams": json.loads(table.to_json(orient="records")),
        "picks": {name: json.loads(picks_table(bracket, p, levels).to_json(orient="records")) for name, p in picks.items()},
    }


def write_outputs(out_dir: Path, season, bracket: Bracket, levels: Levels, picks: dict[str, Picks], first_round: pd.DataFrame | None,
                  meta: dict, report: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def put(name, writer):
        p = out_dir / name
        writer(p)
        paths.append(p)

    put(f"advancement_{season}.csv", lambda p: reach_table(bracket, levels).to_csv(p, index=False))
    if first_round is not None:
        put(f"first_round_{season}.csv", lambda p: first_round.to_csv(p, index=False))
    for name, pk in picks.items():
        slug = name.replace(" ", "_")
        put(f"picks_{season}_{slug}.csv", lambda p, pk=pk: picks_table(bracket, pk, levels).to_csv(p, index=False))
    put(f"bracket_{season}.json", lambda p: p.write_text(json.dumps(bracket_json(bracket, levels, picks, meta), indent=1, default=float)))
    put(f"report_{season}.txt", lambda p: p.write_text(report, encoding="utf-8"))
    return paths
