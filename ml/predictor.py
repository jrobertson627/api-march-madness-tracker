"""A fitted model packaged for Phases 3 and 4.

`fit_final` trains on every season with the chosen configuration and returns a
`Predictor`. `save` writes model.joblib and manifest.json; `load` restores it.
For a logistic win model with a ridge score model, `export_json` writes the
coefficients so a service can predict with numpy alone (`JsonPredictor`).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.calibration import logit, sigmoid
from ml.dataset import swap_features
from ml.experiment import Config, Fitted, fit_components, predict_frame
from ml.models import LogisticModel, normal_cdf


def _versions() -> dict:
    import sklearn

    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__}


class Predictor:
    def __init__(self, fitted: Fitted, manifest: dict):
        self.fitted = fitted
        self.manifest = manifest

    @property
    def required_columns(self) -> list[str]:
        cols = list(self.fitted.cols)
        if self.fitted.score_model is not None:
            cols += [c for c in self.fitted.score_cols if c not in cols] + ["game_group"]
        return cols

    def predict(self, matchups: pd.DataFrame) -> pd.DataFrame:
        """Probability team A wins and, with a score model, margin, total and both scores."""
        missing = [c for c in self.required_columns if c not in matchups.columns]
        if missing:
            raise ValueError(f"matchups are missing columns the model needs: {missing}")
        pred = predict_frame(self.fitted, matchups)
        keep = [c for c in ("p", "p_raw", "margin_hat", "total_reg_hat", "score_a_hat", "score_b_hat", "p_from_margin") if c in pred]
        return pred[keep].rename(columns={"p": "p_a_win"})

    def save(self, out_dir: Path | str) -> Path:
        import joblib

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.fitted, out / "model.joblib")
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=1, default=str))
        return out

    @classmethod
    def load(cls, out_dir: Path | str) -> "Predictor":
        import joblib

        out = Path(out_dir)
        return cls(joblib.load(out / "model.joblib"), json.loads((out / "manifest.json").read_text()))

    def export_json(self, path: Path | str) -> Path:
        f = self.fitted
        if not isinstance(f.model, LogisticModel):
            raise NotImplementedError("JSON export supports logistic win models only")
        if f.calibrator is not None and f.calibrator.params()["kind"] != "platt":
            raise NotImplementedError("JSON export supports Platt calibration only")
        if f.score_model is not None and f.score_model.kind != "ridge":
            raise NotImplementedError("JSON export supports ridge score models only")
        doc = {
            "manifest": self.manifest,
            "win": {"columns": f.model.cols_, "scale": f.model.scale_.reindex(f.model.cols_).tolist(), "coef": f.model.clf_.coef_[0].tolist()},
            "calibration": None if f.calibrator is None else f.calibrator.params(),
        }
        s = f.score_model
        if s is not None:
            doc["score"] = {
                "margin_columns": s.m_cols_, "margin_scale": s.m_scale_.reindex(s.m_cols_).tolist(), "margin_coef": s.m_model_.coef_.tolist(),
                "total_columns": s.t_cols_, "total_fill": s.t_fill_.tolist(), "total_mean": s.t_mean_.tolist(), "total_std": s.t_std_.tolist(),
                "total_coef": s.t_model_.coef_.tolist(), "total_intercept": float(s.t_model_.intercept_), "sigma": s.sigma_,
            }
        Path(path).write_text(json.dumps(doc, indent=1, default=str))
        return Path(path)


class JsonPredictor:
    """Predicts from an exported JSON file with numpy and pandas only."""

    def __init__(self, doc: dict):
        self.doc = doc

    @classmethod
    def load(cls, path: Path | str) -> "JsonPredictor":
        return cls(json.loads(Path(path).read_text()))

    def predict(self, matchups: pd.DataFrame) -> pd.DataFrame:
        d = self.doc
        w = d["win"]
        Z = (matchups[w["columns"]] / np.array(w["scale"])).fillna(0.0)
        p_raw = sigmoid(Z.to_numpy() @ np.array(w["coef"]))
        p = p_raw if d["calibration"] is None else sigmoid(d["calibration"]["slope"] * logit(p_raw) + d["calibration"]["intercept"])
        out = pd.DataFrame({"p_a_win": p, "p_raw": p_raw}, index=matchups.index)
        if "score" in d:
            s = d["score"]
            margin = ((matchups[s["margin_columns"]] / np.array(s["margin_scale"])).fillna(0.0)).to_numpy() @ np.array(s["margin_coef"])
            Xt = matchups[s["total_columns"]].fillna(pd.Series(s["total_fill"], index=s["total_columns"]))
            total = (((Xt - np.array(s["total_mean"])) / np.array(s["total_std"])).to_numpy() @ np.array(s["total_coef"])) + s["total_intercept"]
            sigma = np.array([s["sigma"].get(str(g), s["sigma"]["all"]) for g in matchups["game_group"]])
            out["margin_hat"], out["total_reg_hat"] = margin, total
            out["score_a_hat"], out["score_b_hat"] = (total + margin) / 2, (total - margin) / 2
            out["p_from_margin"] = normal_cdf(margin / sigma)
        return out


def fit_final(cfg: Config, df: pd.DataFrame, stamp: dict | None = None, cv_metrics: list[dict] | None = None) -> Predictor:
    """Fit on every row of every season with the configuration's tuning, calibration and score model."""
    fitted = fit_components(cfg, df, np.arange(len(df)))
    manifest = {
        "name": cfg.name,
        "run_id": cfg.run_id(),
        "config": asdict(cfg),
        "chosen_hyperparameters": fitted.combo,
        "win_features": fitted.cols,
        "score_features": fitted.score_cols if fitted.score_model is not None else [],
        "training_seasons": sorted(int(s) for s in df["season"].unique()),
        "training_rows": int(len(df)),
        "training_game_dates": [str(df["game_date"].min().date()), str(df["game_date"].max().date())],
        "data_stamp": stamp or {},
        "calibration": None if fitted.calibrator is None else fitted.calibrator.params(),
        "sigma": None if fitted.score_model is None else fitted.score_model.sigma_,
        "cross_validated_metrics": cv_metrics or [],
        "packages": _versions(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limits": "Trained on five seasons. About 334 NCAA games in cross-validation, so differences smaller than about 0.02 log loss are within noise.",
    }
    return Predictor(fitted, manifest)
