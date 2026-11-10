"""Print the experiment report: pooled and per-season metrics, paired comparisons, plots.

    python scripts/report_models.py
    python scripts/report_models.py --baseline "B3 elo probability (select seasons)"

Paired comparisons use the games both runs were tested on, 10,000 bootstrap
resamples, and the decision rule in docs/modeling.md. The report also prints
how many runs have been made, because the more that are tried on five seasons,
the more the best one is partly luck.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.experiment import MODEL_DIR  # noqa: E402
from ml.report import load_results, paired, per_season, reliability_plot, summary  # noqa: E402

DEFAULT_BASELINE = "B4 efficiency gap, all rows (select seasons)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--plot", default="", help="run name to draw a reliability plot for (default: best selected model)")
    args = ap.parse_args()

    exp, oof = load_results()
    names = exp.drop_duplicates("name").set_index("name")["run_id"].to_dict()
    runs = pd.read_csv(MODEL_DIR / "runs.csv") if (MODEL_DIR / "runs.csv").exists() else pd.DataFrame()
    print(f"{len(names)} distinct configurations, {len(runs)} runs logged, matrix hash(es): {sorted(exp['data_hash'].dropna().astype(str).unique())}")

    for mode in ("game", "bracket"):
        if (exp["mode"] == mode).any():
            print(f"\n== {mode} mode: pooled over tested seasons ==")
            print(summary(exp, mode).round(4).to_string(index=False))
            print(f"\n-- log loss by season ({mode}) --")
            print(per_season(exp, mode).to_string())

    if args.baseline not in names:
        print(f"\nbaseline {args.baseline!r} has no results; run scripts/run_experiments.py --set models first")
        return
    print(f"\n== paired comparison against: {args.baseline} (per-game mode) ==")
    print("negative mean_diff means the candidate has the lower log loss; keep = the decision rule says keep the complex model")
    rows = []
    for name in names:
        if name == args.baseline or not name.startswith(("M", "S")):
            continue
        r = paired(oof, args.baseline, name, names)
        rows.append({"candidate": name, "games": r["games"], "mean_diff": r["mean_diff"], "ci_low": r["lo"], "ci_high": r["hi"],
                     "season_wins": f"{r['season_wins']}/{r['seasons']}", "keep": r["keep"]})
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    pool = exp[(exp["mode"] == "game") & (exp["season"].astype(str) == "all") & exp["name"].str.startswith(("M", "S"))]
    target = args.plot or (pool.sort_values("log_loss").iloc[0]["name"] if len(pool) else "")
    if target in names:
        path = reliability_plot(oof, names[target], MODEL_DIR / "plots" / "reliability.png")
        print(f"\nreliability plot for {target!r}: {path}")


if __name__ == "__main__":
    main()
