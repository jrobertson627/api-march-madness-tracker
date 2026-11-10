"""Read experiment results back: summaries, paired comparisons, calibration plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml import metrics
from ml.experiment import MODEL_DIR


def load_results(model_dir: Path = MODEL_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(model_dir / "experiments.csv"), pd.read_csv(model_dir / "oof_predictions.csv")


def summary(experiments: pd.DataFrame, mode: str = "game") -> pd.DataFrame:
    """One row per run: pooled metrics over all tested seasons."""
    e = experiments[(experiments["mode"] == mode) & (experiments["season"].astype(str) == "all")]
    cols = ["name", "protocol", "n", "log_loss", "brier", "accuracy", "ece", "margin_mae", "total_mae_reg", "margin_prob_log_loss"]
    return e[[c for c in cols if c in e]].sort_values("log_loss").reset_index(drop=True)


def per_season(experiments: pd.DataFrame, mode: str = "game", metric: str = "log_loss") -> pd.DataFrame:
    e = experiments[(experiments["mode"] == mode) & (experiments["season"].astype(str) != "all")]
    return e.pivot_table(index="name", columns="season", values=metric).round(4)


def paired(oof: pd.DataFrame, baseline: str, candidate: str, names: dict[str, str], mode: str = "game", n: int = 10_000) -> dict:
    """Candidate versus baseline on the games both were tested on. `names` maps run name to run_id."""
    a = oof[(oof["run_id"] == names[candidate]) & (oof["mode"] == mode)].set_index("game_id")
    b = oof[(oof["run_id"] == names[baseline]) & (oof["mode"] == mode)].set_index("game_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    res = metrics.keep_complex(
        metrics.game_log_loss(a["p"], a["a_win"]), metrics.game_log_loss(b["p"], b["a_win"]), a["season"].to_numpy(), n=n
    )
    return {"candidate": candidate, "baseline": baseline, "games": len(common), **res}


def reliability_plot(oof: pd.DataFrame, run_id: str, path: Path, mode: str = "game", bins: int = 10) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = oof[(oof["run_id"] == run_id) & (oof["mode"] == mode)]
    t = metrics.reliability_table(g["p"], g["a_win"], bins)
    rng = np.random.default_rng(0)
    lo, hi = [], []
    for k, idx in enumerate(np.array_split(np.argsort(g["p"].to_numpy(), kind="stable"), min(bins, len(g)))):
        y = g["a_win"].to_numpy()[idx]
        boot = rng.choice(y, size=(2000, len(y))).mean(axis=1)
        lo.append(np.percentile(boot, 2.5))
        hi.append(np.percentile(boot, 97.5))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1)
    ax.errorbar(t["mean_p"], t["frac_win"], yerr=[t["frac_win"] - np.array(lo), np.array(hi) - t["frac_win"]], fmt="o", capsize=3)
    ax.set(xlabel="predicted probability team A wins", ylabel="observed share", xlim=(0, 1), ylim=(0, 1),
           title=f"Reliability, {len(g)} NCAA games ({mode})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
