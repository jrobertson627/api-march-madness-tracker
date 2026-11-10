"""Win-probability and score models.

Every win model returns P(team A wins) and is symmetric: p(A, B) + p(B, A) = 1.
Linear models get that from having no intercept and only signed inputs, and
tree models from training on both orderings and averaging a prediction with
the swapped one. Imputation and scaling are fit inside `fit`, so they only
ever see the training rows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml.calibration import sigmoid
from ml.dataset import augment, is_signed, swap_features

MONOTONE_UP = ("diff_adj_em", "diff_elo", "diff_adj_off", "diff_win_pct", "proj_margin")
TOTAL_COLUMNS = ("proj_total", "sum_adj_off", "sum_adj_def", "sum_adj_tempo", "sum_tempo", "is_ncaa", "is_conf_tourney")


def _keep(w):
    """Row positions with positive weight (zero-weight rows are dropped before fitting)."""
    return None if w is None else np.flatnonzero(np.asarray(w, float) > 0)


def _scale(X: pd.DataFrame) -> pd.Series:
    """Root-mean-square per column over the rows given. No centering, so antisymmetry survives."""
    rms = np.sqrt((X**2).mean(skipna=True))
    return rms.where(rms > 0, 1.0).fillna(1.0)


class WinModel:
    name = "base"

    def fit(self, X: pd.DataFrame, y, w=None) -> "WinModel":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class CoinFlip(WinModel):
    name = "coin"

    def fit(self, X, y, w=None):
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


class EloProbability(WinModel):
    """The Phase 1 Elo win probability, unfitted."""

    name = "elo_prob"

    def fit(self, X, y, w=None):
        return self

    def predict_proba(self, X):
        return X["elo_expected_a"].to_numpy(float)


class LogisticModel(WinModel):
    """L2 logistic regression on the signed columns, no intercept.

    Columns that do not flip sign when teams swap (for example game-type flags)
    are ignored, which is what keeps the model symmetric. Missing differences
    become 0, meaning no information.
    """

    name = "logistic"

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X, y, w=None):
        from sklearn.linear_model import LogisticRegression

        keep = _keep(w)
        if keep is not None:
            X, y, w = X.iloc[keep], np.asarray(y)[keep], np.asarray(w)[keep]
        self.cols_ = [c for c in X.columns if is_signed(c)]
        if not self.cols_:
            raise ValueError("LogisticModel needs at least one signed column")
        Xa, ya, wa = augment(X[self.cols_], y, w)
        self.scale_ = _scale(Xa)
        self.clf_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
        self.clf_.fit((Xa / self.scale_).fillna(0.0), ya, sample_weight=wa)
        return self

    def predict_proba(self, X):
        Z = (X[self.cols_] / self.scale_).fillna(0.0)
        return sigmoid(Z.to_numpy() @ self.clf_.coef_[0])

    def params(self) -> dict:
        return {"C": self.C, "columns": self.cols_, "coef": dict(zip(self.cols_, map(float, self.clf_.coef_[0])))}


class SeedPairModel(WinModel):
    """Win rate of the better seed in each seed pairing, smoothed toward a seed-gap logistic.

    Sparse, so each pairing is shrunk toward a logistic on the seed gap with
    `smoothing` pseudo-games. Equal seeds give 0.5. Symmetric by construction.
    """

    name = "seed_pair"

    def __init__(self, smoothing: float = 5.0):
        self.smoothing = smoothing

    def fit(self, X, y, w=None):
        keep = _keep(w)
        if keep is not None:
            X, y = X.iloc[keep], np.asarray(y)[keep]
        sa, sb = X["seed_a"].to_numpy(float), X["seed_b"].to_numpy(float)
        ok = ~np.isnan(sa) & ~np.isnan(sb) & (sa != sb)
        low_wins = np.where(sa < sb, np.asarray(y, float), 1 - np.asarray(y, float))[ok]
        gap = np.abs(sa - sb)[ok]
        self.k_ = self._fit_slope(gap, low_wins)  # P(better seed wins) = sigmoid(k * seed gap)
        self.table_: dict[tuple[float, float], tuple[float, int]] = {}
        for a, b, win in zip(np.minimum(sa, sb)[ok], np.maximum(sa, sb)[ok], low_wins):
            wins, n = self.table_.get((a, b), (0.0, 0))
            self.table_[(a, b)] = (wins + win, n + 1)
        return self

    @staticmethod
    def _fit_slope(gap, low_wins) -> float:
        if len(gap) == 0:
            return 0.1
        k = 0.1
        for _ in range(50):
            mu = sigmoid(k * gap)
            grad = (gap * (mu - low_wins)).sum() + 1e-6 * k
            hess = (gap**2 * mu * (1 - mu)).sum() + 1e-6
            step = grad / hess
            k -= step
            if abs(step) < 1e-9:
                break
        return float(k)

    def predict_proba(self, X):
        out = np.full(len(X), 0.5)
        for i, (a, b) in enumerate(zip(X["seed_a"].to_numpy(float), X["seed_b"].to_numpy(float))):
            if np.isnan(a) or np.isnan(b) or a == b:
                continue
            prior = float(sigmoid(self.k_ * abs(a - b)))
            wins, n = self.table_.get((min(a, b), max(a, b)), (0.0, 0))
            p_better = (wins + self.smoothing * prior) / (n + self.smoothing)
            out[i] = p_better if a < b else 1 - p_better
        return out


class HGBModel(WinModel):
    """Histogram gradient boosting, trained on both orderings, prediction averaged over the swap."""

    name = "hgb"

    def __init__(
        self, max_depth: int = 3, learning_rate: float = 0.05, max_iter: int = 150,
        l2: float = 1.0, min_samples_leaf: int = 50, monotone: bool = False, random_state: int = 0,
    ):
        self.max_depth, self.learning_rate, self.max_iter = max_depth, learning_rate, max_iter
        self.l2, self.min_samples_leaf, self.monotone, self.random_state = l2, min_samples_leaf, monotone, random_state

    def fit(self, X, y, w=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        keep = _keep(w)
        if keep is not None:
            X, y, w = X.iloc[keep], np.asarray(y)[keep], np.asarray(w)[keep]
        Xa, ya, wa = augment(X, y, w)
        # A column that is entirely missing in the training rows (roster features when only 2022 is
        # available) cannot be binned and carries no information, so it is left out of this fit.
        self.cols_ = [c for c in Xa.columns if Xa[c].notna().any()]
        Xa = Xa[self.cols_]
        cst = {c: 1 for c in self.cols_ if c in MONOTONE_UP} if self.monotone else None
        self.clf_ = HistGradientBoostingClassifier(
            max_depth=self.max_depth, learning_rate=self.learning_rate, max_iter=self.max_iter,
            l2_regularization=self.l2, min_samples_leaf=self.min_samples_leaf, early_stopping=False,
            random_state=self.random_state, monotonic_cst=cst,
        )
        self.clf_.fit(Xa, ya.astype(int), sample_weight=wa)
        return self

    def predict_proba(self, X):
        X = X[self.cols_]
        p = self.clf_.predict_proba(X)[:, 1]
        q = self.clf_.predict_proba(swap_features(X))[:, 1]
        return (p + 1 - q) / 2

    def params(self) -> dict:
        return {k: getattr(self, k) for k in ("max_depth", "learning_rate", "max_iter", "l2", "min_samples_leaf", "monotone")}


WIN_MODELS = {"coin": CoinFlip, "elo_prob": EloProbability, "logistic": LogisticModel, "seed_pair": SeedPairModel, "hgb": HGBModel}


def make_win_model(name: str, **params) -> WinModel:
    if name not in WIN_MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(WIN_MODELS)}")
    return WIN_MODELS[name](**params)


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #

_erf = np.frompyfunc(math.erf, 1, 1)


def normal_cdf(z) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z, float) / math.sqrt(2.0)).astype(float))


class ScoreModel:
    """Predicts margin and total, then derives both scores.

    Margin uses the signed columns and is antisymmetric (swap flips its sign).
    Total uses symmetric columns and predicts a 40-minute game. `sigma` is the
    spread of margin errors per game group, used for the margin-implied win
    probability Phi(margin / sigma).
    """

    def __init__(self, kind: str = "ridge", alpha: float = 10.0, clip: float = 30.0, hgb: dict | None = None):
        self.kind, self.alpha, self.clip, self.hgb = kind, alpha, clip, hgb or {}

    def _make(self):
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=self.alpha, fit_intercept=False), Ridge(alpha=self.alpha)
        if self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            kw = dict(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50, early_stopping=False, random_state=0) | self.hgb
            return HistGradientBoostingRegressor(**kw), HistGradientBoostingRegressor(**kw)
        raise KeyError(f"unknown score model kind {self.kind!r}")

    def fit(self, X: pd.DataFrame, margin, total_reg, w=None, group=None) -> "ScoreModel":
        keep = _keep(w)
        if keep is not None:
            X, margin, total_reg, w = X.iloc[keep], np.asarray(margin)[keep], np.asarray(total_reg)[keep], np.asarray(w)[keep]
            group = None if group is None else np.asarray(group)[keep]
        margin, total_reg = np.asarray(margin, float), np.asarray(total_reg, float)
        self.m_cols_ = [c for c in X.columns if is_signed(c)]
        self.t_cols_ = [c for c in TOTAL_COLUMNS if c in X.columns]
        wa = None if w is None else np.asarray(w, float)

        Xm = X[self.m_cols_]
        Xa = pd.concat([Xm, swap_features(Xm)], ignore_index=True)
        ya = np.concatenate([np.clip(margin, -self.clip, self.clip), -np.clip(margin, -self.clip, self.clip)])
        wm = None if wa is None else np.concatenate([wa, wa])
        self.m_scale_ = _scale(Xa) if self.kind == "ridge" else pd.Series(1.0, index=Xa.columns)
        self.m_model_, self.t_model_ = self._make()
        self.m_model_.fit((Xa / self.m_scale_).fillna(0.0) if self.kind == "ridge" else Xa, ya, sample_weight=wm)

        Xt = X[self.t_cols_]
        self.t_fill_ = Xt.median()
        self.t_mean_, self.t_std_ = Xt.fillna(self.t_fill_).mean(), Xt.fillna(self.t_fill_).std().replace(0, 1.0).fillna(1.0)
        self.t_model_.fit(self._t_matrix(Xt), total_reg, sample_weight=wa)

        resid = margin - self._margin(X)
        self.sigma_ = {"all": float(np.sqrt(np.average(resid**2, weights=wa)))}
        if group is not None:
            for g in np.unique(group):
                m = np.asarray(group) == g
                self.sigma_[str(g)] = float(np.sqrt(np.average(resid[m] ** 2, weights=None if wa is None else wa[m])))
        return self

    def _t_matrix(self, Xt: pd.DataFrame):
        Z = Xt.fillna(self.t_fill_)
        return (Z - self.t_mean_) / self.t_std_ if self.kind == "ridge" else Z

    def _margin_raw(self, Xm: pd.DataFrame) -> np.ndarray:
        if self.kind == "ridge":
            return self.m_model_.predict((Xm / self.m_scale_).fillna(0.0))
        return self.m_model_.predict(Xm)

    def _margin(self, X: pd.DataFrame) -> np.ndarray:
        Xm = X[self.m_cols_]
        return (self._margin_raw(Xm) - self._margin_raw(swap_features(Xm))) / 2  # antisymmetric

    def predict(self, X: pd.DataFrame, group=None) -> pd.DataFrame:
        margin = self._margin(X)
        total = self.t_model_.predict(self._t_matrix(X[self.t_cols_]))
        out = pd.DataFrame(
            {"margin_hat": margin, "total_reg_hat": total, "score_a_hat": (total + margin) / 2, "score_b_hat": (total - margin) / 2},
            index=X.index,
        )
        sigma = np.full(len(X), self.sigma_["all"]) if group is None else np.array([self.sigma_.get(str(g), self.sigma_["all"]) for g in group])
        out["p_from_margin"] = normal_cdf(margin / sigma)
        return out
