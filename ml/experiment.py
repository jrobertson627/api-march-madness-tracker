"""Run one model configuration across folds and score it.

Everything that learns something (tuning, calibration, imputation, the model)
is fit on the training seasons of a fold. The test season is only predicted.
Hyperparameters and calibration use leave-one-season-out inside the training
seasons (see ml.splits.inner).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml import metrics
from ml.calibration import make_calibrator
from ml.dataset import ROOT, feature_columns, training_weights
from ml.models import ScoreModel, make_win_model
from ml.splits import Split, forward_chain, inner, locked_split, loso

MODEL_DIR = ROOT / "artifacts" / "models"
LABEL_COLS = ["game_id", "season", "game_date", "team_a", "team_b", "game_group", "round", "a_win", "margin_a", "total", "total_reg", "score_a", "score_b"]


@dataclass
class Config:
    name: str
    model: str
    feature_set: str
    params: dict = field(default_factory=dict)
    mix: str = "equal"
    grid: dict = field(default_factory=dict)  # values to try; the key "mix" tunes the training mix
    calibration: str | None = None
    protocol: str = "loso"  # loso | forward
    exclude_seasons: tuple = ()
    score_model: str | None = None  # ridge | hgb
    score_params: dict = field(default_factory=dict)
    seed: int = 0

    def run_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


def score_columns(columns) -> list[str]:
    extra = ["sum_adj_off", "sum_adj_def", "sum_adj_tempo", "sum_tempo"]
    return feature_columns("full", columns) + [c for c in extra if c in columns]


def _combos(cfg: Config) -> list[dict]:
    if not cfg.grid:
        return [{}]
    keys = sorted(cfg.grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(cfg.grid[k] for k in keys))]


def _build(cfg: Config, combo: dict):
    params = {**cfg.params, **{k: v for k, v in combo.items() if k != "mix"}}
    return make_win_model(cfg.model, **params), combo.get("mix", cfg.mix)


def _fit(cfg: Config, combo: dict, df: pd.DataFrame, idx: np.ndarray, cols: list[str]):
    model, mix = _build(cfg, combo)
    rows = df.iloc[idx]
    model.fit(rows[cols], rows["a_win"].to_numpy(), training_weights(rows, mix))
    return model


def _inner_oof(cfg: Config, combo: dict, df: pd.DataFrame, train_idx: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Out-of-fold predictions for the NCAA and conference-tournament rows of the training seasons."""
    parts = []
    for sp in inner(df, train_idx):
        model = _fit(cfg, combo, df, sp.train, cols)
        rows = df.iloc[sp.test]
        parts.append(pd.DataFrame({"p": model.predict_proba(rows[cols]), "y": rows["a_win"].to_numpy(), "season": rows["season"].to_numpy()}))
    return pd.concat(parts, ignore_index=True)


@dataclass
class Fitted:
    model: object
    calibrator: object | None
    score_model: ScoreModel | None
    combo: dict
    cols: list[str]
    score_cols: list[str]


def fit_components(cfg: Config, df: pd.DataFrame, train_idx: np.ndarray) -> Fitted:
    """Tune, fit and calibrate using only the rows at `train_idx`."""
    cols = feature_columns(cfg.feature_set, df.columns)
    combos = _combos(cfg)
    oof_cache: dict[int, pd.DataFrame] = {}
    best = 0
    if len(combos) > 1:
        scores = []
        for i, combo in enumerate(combos):
            oof_cache[i] = _inner_oof(cfg, combo, df, train_idx, cols)
            scores.append(metrics.log_loss(oof_cache[i]["p"], oof_cache[i]["y"]))
        best = int(np.argmin(scores))
    combo = combos[best]
    model = _fit(cfg, combo, df, train_idx, cols)

    calibrator = make_calibrator(cfg.calibration)
    if calibrator is not None:
        oof = oof_cache.get(best)
        if oof is None:
            oof = _inner_oof(cfg, combo, df, train_idx, cols)
        calibrator.fit(oof["p"].to_numpy(), oof["y"].to_numpy())

    score_model = None
    scols = score_columns(df.columns)
    if cfg.score_model:
        tr = df.iloc[train_idx]
        score_model = ScoreModel(kind=cfg.score_model, **cfg.score_params).fit(
            tr[scols], tr["margin_a"], tr["total_reg"], training_weights(tr, combo.get("mix", cfg.mix)), group=tr["game_group"]
        )
    return Fitted(model, calibrator, score_model, combo, cols, scols)


def predict_frame(fit: Fitted, rows: pd.DataFrame) -> pd.DataFrame:
    """Probabilities (raw and calibrated) and, if a score model was fit, margin, total and scores."""
    p_raw = fit.model.predict_proba(rows[fit.cols])
    pred = rows[[c for c in LABEL_COLS if c in rows]].copy()
    pred["p_raw"] = p_raw
    pred["p"] = fit.calibrator.transform(p_raw) if fit.calibrator is not None else p_raw
    if fit.score_model is not None:
        pred = pd.concat([pred, fit.score_model.predict(rows[fit.score_cols], group=rows["game_group"])], axis=1)
    return pred


def fit_predict_split(cfg: Config, df: pd.DataFrame, train_idx: np.ndarray, tests: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fit on the training rows, then predict each test frame ({mode: rows})."""
    fit = fit_components(cfg, df, train_idx)
    out = {}
    for mode, rows in tests.items():
        pred = predict_frame(fit, rows)
        pred["mode"] = mode
        pred["chosen"] = json.dumps(fit.combo, sort_keys=True)
        out[mode] = pred
    return out


def score_predictions(pred: pd.DataFrame) -> dict:
    """Metrics for one set of predictions on NCAA games."""
    y, p = pred["a_win"].to_numpy(), pred["p"].to_numpy()
    row = {
        "n": len(pred),
        "log_loss": metrics.log_loss(p, y),
        "brier": metrics.brier(p, y),
        "accuracy": metrics.accuracy(p, y),
        "ece": metrics.ece(p, y),
    }
    if "margin_hat" in pred:
        row |= {
            "margin_mae": metrics.mae(pred["margin_hat"], pred["margin_a"]),
            "margin_rmse": metrics.rmse(pred["margin_hat"], pred["margin_a"]),
            "total_mae_reg": metrics.mae(pred["total_reg_hat"], pred["total_reg"]),
            "total_mae_raw": metrics.mae(pred["total_reg_hat"], pred["total"]),
            "score_mae_raw": (metrics.mae(pred["score_a_hat"], pred["score_a"]) + metrics.mae(pred["score_b_hat"], pred["score_b"])) / 2,
            "margin_prob_log_loss": metrics.log_loss(pred["p_from_margin"], y),
            "margin_agrees_share": float(((pred["margin_hat"] > 0) == (p > 0.5)).mean()),
        }
    return row


@dataclass
class ExperimentResult:
    config: Config
    oof: pd.DataFrame
    metrics: pd.DataFrame


def _splits(cfg: Config, df: pd.DataFrame) -> list[Split]:
    if cfg.protocol == "loso":
        return list(loso(df, cfg.exclude_seasons))
    if cfg.protocol == "forward":
        return list(forward_chain(df, 2, cfg.exclude_seasons))
    raise KeyError(f"unknown protocol {cfg.protocol!r}")


def _bracket_rows(rows: pd.DataFrame, bracket: pd.DataFrame | None):
    if bracket is None:
        return None
    b = bracket.drop_duplicates("game_id").set_index("game_id")
    keep = rows["game_id"].isin(b.index)
    return b.loc[rows.loc[keep, "game_id"]].reset_index() if keep.any() else None


def _collect(cfg: Config, results: list[pd.DataFrame], data_hash: str | None) -> ExperimentResult:
    oof = pd.concat(results, ignore_index=True)
    oof.insert(0, "run_id", cfg.run_id())
    rows = []
    for mode, g in oof.groupby("mode"):
        for season, gs in [*g.groupby("season"), ("all", g)]:
            rows.append(
                {"run_id": cfg.run_id(), "name": cfg.name, "model": cfg.model, "feature_set": cfg.feature_set, "mix": cfg.mix,
                 "calibration": cfg.calibration or "none", "score_model": cfg.score_model or "", "protocol": cfg.protocol,
                 "mode": mode, "season": season, **score_predictions(gs), "seed": cfg.seed, "data_hash": data_hash or ""}
            )
    return ExperimentResult(cfg, oof, pd.DataFrame(rows))


def run_experiment(cfg: Config, df: pd.DataFrame, bracket: pd.DataFrame | None = None, data_hash: str | None = None) -> ExperimentResult:
    """Fold by fold: train on the training seasons, predict the test season in per-game mode and,
    if `bracket` rows are given, in bracket mode too (same fitted model)."""
    results = []
    for sp in _splits(cfg, df):
        test = df.iloc[sp.test]
        tests = {"game": test}
        b = _bracket_rows(test, bracket)
        if b is not None:
            tests["bracket"] = b
        results.extend(fit_predict_split(cfg, df, sp.train, tests).values())
    return _collect(cfg, results, data_hash)


def run_locked(cfg: Config, df: pd.DataFrame, season: int, bracket: pd.DataFrame | None = None,
               data_hash: str | None = None, state_dir: Path = MODEL_DIR, force: bool = False) -> ExperimentResult:
    """Score the locked-box season once. Refuses a second run unless forced, and logs the config."""
    state_dir.mkdir(parents=True, exist_ok=True)
    guard = state_dir / "locked_box.json"
    if guard.exists() and not force:
        raise RuntimeError(f"the locked-box season was already scored ({guard.read_text()[:120]}...). Pass force=True to override.")
    sp = locked_split(df, season)
    test = df.iloc[sp.test]
    tests = {"game": test}
    b = _bracket_rows(test, bracket)
    if b is not None:
        tests["bracket"] = b
    res = _collect(cfg, list(fit_predict_split(cfg, df, sp.train, tests).values()), data_hash)
    res.metrics["protocol"] = "locked"
    guard.write_text(json.dumps({"season": season, "config": asdict(cfg), "run_id": cfg.run_id(),
                                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "metrics": res.metrics[res.metrics["season"] == "all"].to_dict("records")}, default=str, indent=1))
    return res


def save_results(res: ExperimentResult, out_dir: Path = MODEL_DIR) -> None:
    """Idempotent: a rerun of the same config replaces its rows. runs.csv logs every run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, new, key in (("experiments.csv", res.metrics, ["run_id"]), ("oof_predictions.csv", res.oof, ["run_id"])):
        path = out_dir / fname
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old["run_id"].isin(new["run_id"].unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)
    log = pd.DataFrame([{"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": res.config.run_id(), "name": res.config.name}])
    log.to_csv(out_dir / "runs.csv", mode="a", header=not (out_dir / "runs.csv").exists(), index=False)
