"""Uplift modelling benchmark: learners, metrics and a synthetic oracle."""

from uplift.dgp import DGPConfig, generate
from uplift.learners import LEARNERS, ResponseModel, SLearner, TLearner, XLearner
from uplift.metrics import (
    effect_correlation,
    incremental_at_k,
    pehe,
    qini_auc,
    qini_coefficient,
    qini_curve,
)

__version__ = "0.1.0"

__all__ = [
    "DGPConfig",
    "generate",
    "LEARNERS",
    "ResponseModel",
    "SLearner",
    "TLearner",
    "XLearner",
    "qini_curve",
    "qini_auc",
    "qini_coefficient",
    "incremental_at_k",
    "pehe",
    "effect_correlation",
]
