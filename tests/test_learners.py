"""Tests for the meta-learners.

Behavioural, not numerical: an exact uplift estimate would pin the suite to
one sklearn version, whereas recovering the ordering of a known effect is
the property that has to hold.
"""

import numpy as np
import pytest

from uplift import metrics
from uplift.dgp import DGPConfig, feature_columns, generate
from uplift.learners import LEARNERS, ResponseModel, SLearner, TLearner, XLearner

UPLIFT_LEARNERS = (SLearner, TLearner, XLearner)


@pytest.fixture(scope="module")
def split():
    frame = generate(DGPConfig(n=12_000, effect_size=0.06, heterogeneity=1.5, seed=30))
    columns = feature_columns(frame)
    train = frame.sample(frac=0.7, random_state=1)
    test = frame.drop(train.index)
    return train, test, columns


def _fit(cls, train, columns):
    return cls().fit(train[columns].values, train["treatment"], train["outcome"])


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_predictions_have_one_value_per_row(cls, split):
    train, test, columns = split
    predictions = _fit(cls, train, columns).predict_uplift(test[columns].values)
    assert predictions.shape == (len(test),)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_learners_recover_the_ordering_of_the_true_effect(cls, split):
    train, test, columns = split
    predictions = _fit(cls, train, columns).predict_uplift(test[columns].values)
    correlation = metrics.effect_correlation(test["tau"], predictions)
    assert correlation > 0.3


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_learners_beat_random_targeting(cls, split):
    train, test, columns = split
    predictions = _fit(cls, train, columns).predict_uplift(test[columns].values)
    assert metrics.qini_auc(test["outcome"], test["treatment"], predictions) > 0


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_uplift_learners_rank_effects_better_than_a_response_model(cls, split):
    """The result the project exists to demonstrate.

    A response model ranks people by how likely they are to convert, which
    conflates persuadables with people who would have converted anyway.
    """
    train, test, columns = split
    uplift = _fit(cls, train, columns).predict_uplift(test[columns].values)
    response = _fit(ResponseModel, train, columns).predict_uplift(test[columns].values)
    assert metrics.effect_correlation(
        test["tau"], uplift
    ) > metrics.effect_correlation(test["tau"], response)


def test_response_model_predicts_conversion_not_uplift(split):
    """Its output should track the baseline rate, not the effect."""
    train, test, columns = split
    predictions = _fit(ResponseModel, train, columns).predict_uplift(
        test[columns].values
    )
    assert metrics.effect_correlation(test["p0"], predictions) > 0.5
    assert (predictions >= 0).all()


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_estimates_are_closer_to_the_truth_than_a_zero_effect_baseline(cls, split):
    train, test, columns = split
    predictions = _fit(cls, train, columns).predict_uplift(test[columns].values)
    assert metrics.pehe(test["tau"], predictions) < metrics.pehe(
        test["tau"], np.zeros(len(test))
    )


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_a_single_arm_is_rejected(cls, split):
    train, _, columns = split
    with pytest.raises(ValueError):
        cls().fit(train[columns].values, np.ones(len(train)), train["outcome"])


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_mismatched_lengths_are_rejected(cls, split):
    train, _, columns = split
    with pytest.raises(ValueError):
        cls().fit(train[columns].values, train["treatment"][:-5], train["outcome"])


@pytest.mark.parametrize("cls", UPLIFT_LEARNERS)
def test_non_binary_treatment_is_rejected(cls, split):
    train, _, columns = split
    treatment = train["treatment"].copy()
    treatment.iloc[:10] = 2
    with pytest.raises(ValueError):
        cls().fit(train[columns].values, treatment, train["outcome"])


def test_x_learner_uses_the_observed_propensity_when_none_is_given(split):
    train, _, columns = split
    model = _fit(XLearner, train, columns)
    assert model.propensity_ == pytest.approx(train["treatment"].mean())


def test_x_learner_accepts_a_known_propensity(split):
    train, test, columns = split
    model = XLearner(propensity=0.5).fit(
        train[columns].values, train["treatment"], train["outcome"]
    )
    assert model.propensity_ == 0.5
    assert np.isfinite(model.predict_uplift(test[columns].values)).all()


def test_learner_registry_is_complete():
    assert set(LEARNERS) == {"response", "s-learner", "t-learner", "x-learner"}
