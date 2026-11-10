"""Build the Phase 1 feature files.

    python scripts/build_features.py
    python scripts/build_features.py --seasons 2024 2025 2026 --cadence daily
    python scripts/build_features.py --mode bracket --include ncaa

Writes artifacts/features/{matchups.csv, team_asof.csv, columns.csv, report.txt}
and prints the same report. The build fails if any row uses a game dated after
its as-of date.
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.db import get_pool, query  # noqa: E402
from data.features import GAME_GROUPS, build_feature_matrix, column_dictionary  # noqa: E402

OUT = ROOT / "artifacts" / "features"


def season_srs(seasons) -> pd.DataFrame:
    """Season-level SRS. It includes tournament games, so it is only a sanity check."""
    rows = query(
        """
        SELECT s.canonical_source_path AS team, ss.ending_year AS season, ts.srs
        FROM team_seasons ts
        JOIN school_seasons ss ON ss.id = ts.school_season_id
        JOIN schools s ON s.id = ss.school_id
        WHERE ss.ending_year = ANY(%s)
        """,
        (list(seasons),),
    )
    df = pd.DataFrame(rows)
    df["srs"] = pd.to_numeric(df["srs"], errors="coerce")
    return df


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without scipy: Pearson on ranks."""
    return a.rank().corr(b.rank())


def report(result, seasons) -> None:
    m, ta, cutoffs, rep = result["matchups"], result["team_asof"], result["cutoffs"], result["report"]
    print(f"features built for seasons {seasons}")
    print(f"cadence={rep['cadence']}  as-of mode={rep['mode']}  team_asof rows={rep['team_asof_rows']}")
    print(f"grid dates per season: {rep['grid_dates']}")
    print(f"games dropped because they come before the first as-of date: {rep['games_without_asof']}")
    print("leak guards passed: every team_asof row uses games on or before its as_of; every matchup as_of is before its game")

    print("\nmatchup rows by season and game group (features = both teams found, trainable = also min games):")
    t = m.groupby(["season", "game_group"]).agg(
        rows=("game_id", "size"), features=("both_have_features", "sum"), trainable=("trainable", "sum")
    )
    print(t.to_string())
    ncaa = m[m["game_group"] == "ncaa"]
    print("\nNCAA tournament games with features, per season (67 played):")
    print(ncaa.groupby("season")["both_have_features"].sum().to_string())

    snap = ta.assign(cutoff=ta["season"].map(cutoffs))
    snap = snap[snap["as_of"] == snap["cutoff"]]
    seeded = snap[snap["seed"].notna()]
    print("\nSelection Sunday snapshot, seeded teams: box-score coverage")
    print(seeded.groupby("season")["box_coverage"].agg(["count", "mean", "min"]).round(3).to_string())

    print("\nsanity: rank correlation of adj_em at Selection Sunday (check only, never a feature)")
    srs = season_srs(seasons)
    rows = []
    for s, g in snap.groupby("season"):
        gs = g.merge(srs[srs["season"] == s], on=["team", "season"], how="left")
        rows.append(
            dict(
                season=s,
                vs_seed=spearman(g["adj_em"], g["seed"]),
                vs_srs=spearman(gs["adj_em"], gs["srs"]),
                elo_vs_adj_em=spearman(g["elo"], g["adj_em"]),
            )
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("(expect vs_seed strongly negative, since seed 1 is best; vs_srs and elo_vs_adj_em strongly positive)")

    tr = m[m["trainable"] == 1]
    feats = [c for c in tr.columns if c.endswith(("_a",)) and c[:-2] in ta.columns and c[:-2] not in ("conference",)]
    nan = tr[feats].isna().mean().sort_values(ascending=False)
    print(f"\nmissing share in trainable rows (a-side columns with any missing), {len(tr)} rows:")
    print(nan[nan > 0].round(3).to_string() if (nan > 0).any() else "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--include", nargs="+", default=list(GAME_GROUPS), choices=GAME_GROUPS)
    ap.add_argument("--cadence", default="weekly", choices=["weekly", "daily"])
    ap.add_argument("--mode", default="game", choices=["game", "bracket"])
    ap.add_argument("--min-games", type=int, default=8)
    args = ap.parse_args()

    result = build_feature_matrix(
        args.seasons, include=args.include, cadence=args.cadence, mode=args.mode, min_games=args.min_games
    )
    OUT.mkdir(parents=True, exist_ok=True)
    result["matchups"].to_csv(OUT / "matchups.csv", index=False)
    result["team_asof"].to_csv(OUT / "team_asof.csv", index=False)
    column_dictionary(result["team_asof"].columns, result["matchups"].columns).to_csv(OUT / "columns.csv", index=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(result, sorted(args.seasons))
        print(f"\nwrote files to {OUT}")
    text = buf.getvalue()
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    get_pool().close()


if __name__ == "__main__":
    main()
