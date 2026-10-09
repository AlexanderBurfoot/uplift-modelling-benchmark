"""Synthetic randomised experiment with known individual treatment effects.

Three choices worth recording:

- Assignment is an independent coin flip, so there is no confounding to
  correct for. Observational data is out of scope.
- tau is set on the probability scale, not the logit scale, and the stored
  value is recomputed after clipping. A logit shift would make the effect
  vary with the baseline rate even at heterogeneity=0.
- Response and effect features are disjoint, so a model that learns only
  "who converts" cannot recover the effect by proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["DGPConfig", "generate"]


@dataclass(frozen=True)
class DGPConfig:
    """Parameters controlling the data-generating process.

    Attributes:
        n: Number of individuals.
        n_response: Features driving baseline conversion probability.
        n_effect: Features driving the treatment effect. Disjoint from the
            response features.
        n_noise: Pure noise features, included to make feature selection a
            real part of the problem.
        treatment_share: Fraction assigned to treatment. Values away from
            0.5 are the regime where the X-learner is expected to help.
        base_rate: Target mean conversion probability without treatment.
        effect_size: Target mean treatment effect, in probability points.
        heterogeneity: 0 gives a constant effect for everyone. Larger values
            widen the spread of effects across individuals.
        sleeping_dog_share: Fraction forced to a negative effect, i.e. people
            the treatment actively puts off.
        seed: Seed for the random generator.
    """

    n: int = 20_000
    n_response: int = 5
    n_effect: int = 3
    n_noise: int = 10
    treatment_share: float = 0.5
    base_rate: float = 0.10
    effect_size: float = 0.05
    heterogeneity: float = 1.0
    sleeping_dog_share: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.treatment_share < 1.0:
            raise ValueError("treatment_share must lie strictly between 0 and 1")
        if not 0.0 < self.base_rate < 1.0:
            raise ValueError("base_rate must lie strictly between 0 and 1")
        if self.heterogeneity < 0:
            raise ValueError("heterogeneity must be non-negative")
        if not 0.0 <= self.sleeping_dog_share <= 1.0:
            raise ValueError("sleeping_dog_share must lie in [0, 1]")
        if min(self.n_response, self.n_effect) < 1:
            raise ValueError("need at least one response and one effect feature")

    @property
    def n_features(self) -> int:
        return self.n_response + self.n_effect + self.n_noise

    @property
    def feature_names(self) -> list[str]:
        return [f"x{i}" for i in range(self.n_features)]

    @property
    def response_features(self) -> list[str]:
        return [f"x{i}" for i in range(self.n_response)]

    @property
    def effect_features(self) -> list[str]:
        start = self.n_response
        return [f"x{i}" for i in range(start, start + self.n_effect)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _solve_intercept(logits: np.ndarray, target: float) -> float:
    """Intercept that makes the mean sigmoid equal ``target``.

    Monotone in the intercept, so bisection suffices and avoids a solver
    dependency. 200 halvings over +/-30 is past machine precision.
    """
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _sigmoid(logits + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _standardise(z: np.ndarray) -> np.ndarray:
    sd = z.std()
    if sd == 0:
        return np.zeros_like(z)
    return (z - z.mean()) / sd


def generate(config: DGPConfig | None = None, **overrides) -> pd.DataFrame:
    """Draw a synthetic randomised experiment.

    Columns: ``x0..xk`` features, ``treatment``, ``outcome``, and the oracle
    columns ``p0``, ``p1``, ``tau``. Use :func:`feature_columns` to select
    model inputs; passing the oracles to a learner leaks the answer.
    """
    if config is None:
        config = DGPConfig(**overrides)
    elif overrides:
        config = DGPConfig(**{**config.__dict__, **overrides})

    rng = np.random.default_rng(config.seed)
    X = rng.standard_normal((config.n, config.n_features))
    frame = pd.DataFrame(X, columns=config.feature_names)

    # Baseline response depends only on the response features.
    response_idx = list(range(config.n_response))
    beta = rng.normal(0.0, 1.0, size=config.n_response)
    logit = X[:, response_idx] @ beta
    logit = _standardise(logit)  # unit variance, so base_rate controls the level
    intercept = _solve_intercept(logit, config.base_rate)
    p0 = _sigmoid(logit + intercept)

    # Treatment effect depends only on the effect features. Non-linear so a
    # purely linear learner is at a genuine disadvantage.
    effect_idx = list(range(config.n_response, config.n_response + config.n_effect))
    gamma = rng.normal(0.0, 1.0, size=config.n_effect)
    raw = X[:, effect_idx] @ gamma + 0.5 * X[:, effect_idx[0]] * X[:, effect_idx[-1]]
    raw = _standardise(raw)
    tau = config.effect_size * (1.0 + config.heterogeneity * raw)

    if config.sleeping_dog_share > 0:
        n_dogs = int(round(config.sleeping_dog_share * config.n))
        dogs = rng.choice(config.n, size=n_dogs, replace=False)
        tau[dogs] = -np.abs(tau[dogs])

    p1 = np.clip(p0 + tau, 1e-3, 1 - 1e-3)
    tau = p1 - p0  # record what the sampler actually uses, post-clipping

    treatment = (rng.random(config.n) < config.treatment_share).astype(int)
    p_realised = np.where(treatment == 1, p1, p0)
    outcome = (rng.random(config.n) < p_realised).astype(int)

    frame["treatment"] = treatment
    frame["outcome"] = outcome
    frame["p0"] = p0
    frame["p1"] = p1
    frame["tau"] = tau
    return frame


ORACLE_COLUMNS = ("p0", "p1", "tau")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns only, with outcome, treatment and oracles removed."""
    excluded = {"treatment", "outcome", *ORACLE_COLUMNS}
    return [c for c in frame.columns if c not in excluded]
