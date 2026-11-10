"""A short, readable record of what a saved model is and how well it did."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRIC_COLS = ["mode", "protocol", "n", "log_loss", "brier", "accuracy", "ece", "margin_mae", "total_mae_reg"]


def model_card(manifest: dict, plot_path: str | None = None) -> str:
    m = manifest
    seasons = m["training_seasons"]
    lines = [
        f"# Model card: {m['name']}",
        "",
        f"Run id `{m['run_id']}`, trained {m['created']}.",
        "",
        "## What it predicts",
        "",
        "The probability that team A beats team B (team A is the alphabetically earlier team id), and, when a score model",
        "is included, the margin, the total for a 40-minute game, and both teams' scores. Probabilities are symmetric:",
        "swapping the teams gives one minus the probability.",
        "",
        "## Training data",
        "",
        f"- Seasons: {seasons[0]} to {seasons[-1]} ({len(seasons)} seasons), {m['training_rows']:,} games",
        f"- Game dates: {m['training_game_dates'][0]} to {m['training_game_dates'][1]}",
        f"- Matrix hash: `{m['data_stamp'].get('hash', 'unknown')}`",
        "- Row types: regular season, conference tournament and NCAA tournament, weighted by the training mix below",
        "",
        "## Model",
        "",
        f"- Win model: `{m['config']['model']}` on feature set `{m['config']['feature_set']}` ({len(m['win_features'])} columns)",
        f"- Hyperparameters chosen by nested leave-one-season-out: `{m['chosen_hyperparameters'] or 'defaults'}`",
        f"- Training mix: `{m['chosen_hyperparameters'].get('mix', m['config']['mix'])}`",
        f"- Calibration: `{m['calibration'] or 'none'}`",
        f"- Score model: `{m['config']['score_model'] or 'none'}`",
        "",
        "Features: " + ", ".join(f"`{c}`" for c in m["win_features"]),
        "",
        "## Held-out results",
        "",
        "Scores on seasons the model was not trained on. Protocol `loso` is leave-one-season-out, `forward` trains on earlier",
        "seasons only, and `locked` is the 2026 locked-box season, scored once with the final design.",
        "",
    ]
    cv = pd.DataFrame(m.get("cross_validated_metrics") or [])
    if cv.empty:
        lines.append("No cross-validation rows were attached to this model.")
    else:
        cols = [c for c in METRIC_COLS if c in cv]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in cv[cols].iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    if plot_path:
        lines += ["", f"Reliability plot: `{plot_path}`"]
    lines += [
        "",
        "## Known limits",
        "",
        f"- {m['limits']}",
        "- Per-game mode lets each game use every earlier result. Bracket mode (Selection Sunday for every round) is the",
        "  honest number for filling out a bracket and is expected to score lower.",
        "- Elo and rating settings are the Phase 1 defaults. The tuning grid found nothing better.",
        "",
        "## Packages",
        "",
        ", ".join(f"{k} {v}" for k, v in m["packages"].items()),
        "",
        "## Use",
        "",
        "```python",
        "from ml.predictor import Predictor",
        "pred = Predictor.load('<this folder>')",
        "pred.predict(matchups)   # needs the columns in the manifest",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_model_card(out_dir: Path | str, manifest: dict, plot_path: str | None = None) -> Path:
    path = Path(out_dir) / "model_card.md"
    path.write_text(model_card(manifest, plot_path), encoding="utf-8")
    return path
