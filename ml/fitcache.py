"""Fit a Phase 2 configuration on chosen seasons, once, and keep the result on disk."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.dataset import ROOT
from ml.experiment import fit_components
from ml.predictor import Predictor
from ml.sets import find

CACHE = ROOT / "artifacts" / "bracket" / "cache"


def fitted_predictor(name: str, df: pd.DataFrame, train_seasons, data_hash: str, cache_dir: Path = CACHE) -> Predictor:
    """The configuration called `name`, trained on rows of `train_seasons` only. Cached by config, data and seasons."""
    cfg = find(name)
    key = f"{cfg.run_id()}_{data_hash}_{'-'.join(map(str, train_seasons))}.joblib"
    path = cache_dir / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return Predictor(joblib.load(path), {})
    idx = np.flatnonzero(df["season"].isin(list(train_seasons)).to_numpy())
    fitted = fit_components(cfg, df, idx)
    joblib.dump(fitted, path)
    return Predictor(fitted, {})
