"""Meta-learners for conditional average treatment effect estimation.

Contract:

    learner.fit(X, treatment, y)
    learner.predict_uplift(X)  ->  P(Y=1|do(T=1)) - P(Y=1|do(T=0))

All of these assume randomised assignment and none corrects for
confounding.

Künzel, Sekhon, Bickel and Yu (2019), "Metalearners for estimating
heterogeneous treatment effects using machine learning", PNAS.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

__all__ = ["SLearner", "TLearner", "XLearner", "ResponseModel"]


def _default_classifier():
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200, random_state=0
    )


def _default_regressor():
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=200, random_state=0
    )


def _as_array(X) -> np.ndarray:
    return np.asarray(X, dtype=float)


def _positive_proba(model, X: np.ndarray) -> np.ndarray:
    """P(Y=1) from a fitted classifier, tolerating a single-class fit."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


class BaseLearner:
    """Shared plumbing. Not useful on its own."""

    name = "base"

    def fit(self, X, treatment, y):  # pragma: no cover - interface only
        raise NotImplementedError

    def predict_uplift(self, X) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _check(X, treatment, y):
        X = _as_array(X)
        treatment = np.asarray(treatment).astype(int).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if not (len(X) == len(treatment) == len(y)):
            raise ValueError("X, treatment and y must have the same length")
        # An integer cast would truncate a continuous outcome into dozens of
        # classes without complaint, so refuse it instead.
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must be coded 0/1; these learners fit classifiers")
        y = y.astype(int)
        if set(np.unique(treatment)) - {0, 1}:
            raise ValueError("treatment must be coded 0/1")
        if treatment.sum() == 0 or (1 - treatment).sum() == 0:
            raise ValueError("need both treated and control observations")
        return X, treatment, y


class ResponseModel(BaseLearner):
    """Not an uplift model: predicts P(Y=1), ignoring treatment.

    The comparison baseline. This is what "who is likely to convert?"
    produces, and it cannot separate persuadables from sure things.
    """

    name = "response"

    def __init__(self, model=None):
        self.model = model or _default_classifier()

    def fit(self, X, treatment, y):
        X, treatment, y = self._check(X, treatment, y)
        self.model_ = clone(self.model).fit(X, y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return _positive_proba(self.model_, _as_array(X))


class SLearner(BaseLearner):
    """Single model with treatment as an extra feature.

    Failure mode: when the effect is small relative to the baseline signal,
    a regularised model can drop the treatment feature and predict near-zero
    uplift for everyone.
    """

    name = "s-learner"

    def __init__(self, model=None):
        self.model = model or _default_classifier()

    def fit(self, X, treatment, y):
        X, treatment, y = self._check(X, treatment, y)
        augmented = np.hstack([X, treatment.reshape(-1, 1)])
        self.model_ = clone(self.model).fit(augmented, y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        ones = np.hstack([X, np.ones((len(X), 1))])
        zeros = np.hstack([X, np.zeros((len(X), 1))])
        return _positive_proba(self.model_, ones) - _positive_proba(self.model_, zeros)


class TLearner(BaseLearner):
    """Separate models per arm, differenced.

    Each model sees only its own arm, so an unbalanced split fits the
    smaller arm poorly and that error carries into the difference.
    """

    name = "t-learner"

    def __init__(self, model=None):
        self.model = model or _default_classifier()

    def fit(self, X, treatment, y):
        X, treatment, y = self._check(X, treatment, y)
        self.model_treated_ = clone(self.model).fit(X[treatment == 1], y[treatment == 1])
        self.model_control_ = clone(self.model).fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        return _positive_proba(self.model_treated_, X) - _positive_proba(
            self.model_control_, X
        )


class XLearner(BaseLearner):
    """Two-stage learner that regresses imputed individual effects on X.

    Stage 1 fits outcome models per arm. Stage 2 imputes an effect for each
    observation against the opposite arm's model, then fits an effect model
    per arm and combines the two by propensity.

    Expected to lead on unbalanced arms: the larger arm yields the better
    effect model and the propensity weighting favours it.
    """

    name = "x-learner"

    def __init__(self, outcome_model=None, effect_model=None, propensity=None):
        self.outcome_model = outcome_model or _default_classifier()
        self.effect_model = effect_model or _default_regressor()
        self.propensity = propensity

    def fit(self, X, treatment, y):
        X, treatment, y = self._check(X, treatment, y)
        treated, control = treatment == 1, treatment == 0

        # Stage 1: outcome models per arm.
        self.mu1_ = clone(self.outcome_model).fit(X[treated], y[treated])
        self.mu0_ = clone(self.outcome_model).fit(X[control], y[control])

        # Stage 2: impute individual effects against the opposite arm's model.
        d1 = y[treated] - _positive_proba(self.mu0_, X[treated])
        d0 = _positive_proba(self.mu1_, X[control]) - y[control]

        self.tau1_ = clone(self.effect_model).fit(X[treated], d1)
        self.tau0_ = clone(self.effect_model).fit(X[control], d0)

        if self.propensity is None:
            self.propensity_ = float(treated.mean())
        else:
            self.propensity_ = float(self.propensity)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        g = self.propensity_
        # g weights toward the effect model fitted on the larger arm.
        return g * self.tau0_.predict(X) + (1.0 - g) * self.tau1_.predict(X)


LEARNERS = {
    cls.name: cls for cls in (ResponseModel, SLearner, TLearner, XLearner)
}
