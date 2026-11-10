"""Probability calibration. Both calibrators keep p(A, B) + p(B, A) = 1.

Platt scaling here is slope-only (no intercept) by default: an intercept would
learn the alphabetical ordering of team A and B, and would break symmetry.
"""

from __future__ import annotations

import numpy as np

P_CLIP = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)


def logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class PlattCalibrator:
    """calibrated = sigmoid(slope * logit(p) + intercept), fit by Newton's method."""

    def __init__(self, fit_intercept: bool = False, ridge: float = 1e-6):
        self.fit_intercept = fit_intercept
        self.ridge = ridge
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, p, y, weight=None):
        z = logit(p)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
        cols = [z, np.ones_like(z)] if self.fit_intercept else [z]
        A = np.column_stack(cols)
        theta = np.zeros(A.shape[1])
        theta[0] = 1.0
        for _ in range(50):
            mu = sigmoid(A @ theta)
            grad = A.T @ (w * (mu - y)) + self.ridge * theta
            hess = (A * (w * mu * (1 - mu))[:, None]).T @ A + self.ridge * np.eye(len(theta))
            step = np.linalg.solve(hess, grad)
            theta -= step
            if np.abs(step).max() < 1e-9:
                break
        self.slope = float(theta[0])
        self.intercept = float(theta[1]) if self.fit_intercept else 0.0
        return self

    def transform(self, p) -> np.ndarray:
        return sigmoid(self.slope * logit(p) + self.intercept)

    def params(self) -> dict:
        return {"kind": "platt", "slope": self.slope, "intercept": self.intercept}


class IsotonicCalibrator:
    """Monotone calibration, fit on both orderings and averaged so it stays symmetric."""

    def __init__(self):
        self.model = None

    def fit(self, p, y, weight=None):
        from sklearn.isotonic import IsotonicRegression

        p, y = _clip(p), np.asarray(y, dtype=float)
        w = None if weight is None else np.concatenate([weight, weight])
        self.model = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        self.model.fit(np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y]), sample_weight=w)
        return self

    def transform(self, p) -> np.ndarray:
        p = _clip(p)
        return 0.5 * (self.model.predict(p) + 1 - self.model.predict(1 - p))

    def params(self) -> dict:
        return {"kind": "isotonic"}


def make_calibrator(kind: str | None):
    if kind in (None, "none"):
        return None
    if kind == "platt":
        return PlattCalibrator()
    if kind == "isotonic":
        return IsotonicCalibrator()
    raise KeyError(f"unknown calibration {kind!r}; choose none, platt or isotonic")
