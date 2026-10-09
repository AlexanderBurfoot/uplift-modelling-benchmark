"""Tests for the Qini machinery and the oracle metrics."""

import numpy as np
import pytest

from uplift import metrics
from uplift.dgp import DGPConfig, generate


@pytest.fixture(scope="module")
def experiment():
    frame = generate(DGPConfig(n=20_000, effect_size=0.05, heterogeneity=1.0, seed=20))
    return frame


def test_curve_starts_at_zero_and_has_one_point_per_observation(experiment):
    fraction, gains = metrics.qini_curve(
        experiment["outcome"], experiment["treatment"], experiment["tau"]
    )
    assert len(fraction) == len(gains) == len(experiment) + 1
    assert fraction[0] == 0.0 and gains[0] == 0.0
    assert fraction[-1] == pytest.approx(1.0)


def test_curve_endpoint_equals_the_overall_incremental_effect(experiment):
    """At full depth the Qini value is the scaled difference in conversions."""
    y, t = experiment["outcome"].values, experiment["treatment"].values
    expected = y[t == 1].sum() - y[t == 0].sum() * (t == 1).sum() / (t == 0).sum()
    _, gains = metrics.qini_curve(y, t, experiment["tau"])
    assert gains[-1] == pytest.approx(expected)


def test_endpoint_is_invariant_to_the_ranking(experiment):
    """Only the path differs between rankings; the destination is fixed."""
    y, t = experiment["outcome"], experiment["treatment"]
    rng = np.random.default_rng(0)
    _, oracle = metrics.qini_curve(y, t, experiment["tau"])
    _, random = metrics.qini_curve(y, t, rng.random(len(experiment)))
    assert oracle[-1] == pytest.approx(random[-1])


def test_oracle_ranking_beats_random_ranking(experiment):
    y, t = experiment["outcome"], experiment["treatment"]
    rng = np.random.default_rng(1)
    oracle = metrics.qini_auc(y, t, experiment["tau"])
    random = metrics.qini_auc(y, t, rng.random(len(experiment)))
    assert oracle > random > -np.inf
    assert oracle > 0


def test_random_rankings_average_to_roughly_zero_area(experiment):
    """A single random draw is noisy; the expectation is zero."""
    y, t = experiment["outcome"], experiment["treatment"]
    rng = np.random.default_rng(2)
    areas = [metrics.qini_auc(y, t, rng.random(len(experiment))) for _ in range(30)]
    oracle = metrics.qini_auc(y, t, experiment["tau"])
    assert abs(np.mean(areas)) < 0.15 * oracle


def test_reversed_ranking_mirrors_the_oracle(experiment):
    y, t = experiment["outcome"], experiment["treatment"]
    oracle = metrics.qini_auc(y, t, experiment["tau"])
    reversed_ = metrics.qini_auc(y, t, -experiment["tau"])
    assert reversed_ < 0 < oracle


def test_constant_score_is_indistinguishable_from_random(experiment):
    """A score carrying no information should sit inside the random spread.

    Asserting its area is near zero would be asserting about noise: a
    constant score preserves input order, which is one draw from the random
    distribution, and that distribution is wide. The testable claim is that
    the draw is not an outlier.
    """
    y, t = experiment["outcome"], experiment["treatment"]
    rng = np.random.default_rng(3)
    random_areas = [
        metrics.qini_auc(y, t, rng.random(len(experiment))) for _ in range(40)
    ]
    spread = np.std(random_areas)
    constant = metrics.qini_auc(y, t, np.ones(len(experiment)))
    assert abs(constant - np.mean(random_areas)) < 3 * spread


def test_qini_coefficient_is_one_against_itself(experiment):
    y, t = experiment["outcome"], experiment["treatment"]
    value = metrics.qini_coefficient(y, t, experiment["tau"], experiment["tau"])
    assert value == pytest.approx(1.0)


def test_incremental_at_full_depth_matches_the_curve_endpoint(experiment):
    y, t = experiment["outcome"], experiment["treatment"]
    _, gains = metrics.qini_curve(y, t, experiment["tau"])
    assert metrics.incremental_at_k(y, t, experiment["tau"], 1.0) == pytest.approx(
        gains[-1]
    )


def test_targeting_the_top_decile_by_true_effect_beats_targeting_everyone(experiment):
    """The core claim of uplift modelling, stated as a test.

    Treating only the top 10% by true effect should capture a
    disproportionate share of the total incremental conversions.
    """
    y, t = experiment["outcome"], experiment["treatment"]
    top_decile = metrics.incremental_at_k(y, t, experiment["tau"], 0.1)
    everyone = metrics.incremental_at_k(y, t, experiment["tau"], 1.0)
    assert top_decile > 0.1 * everyone


def test_pehe_is_zero_for_a_perfect_estimate(experiment):
    assert metrics.pehe(experiment["tau"], experiment["tau"]) == pytest.approx(0.0)


def test_pehe_grows_with_error(experiment):
    tau = experiment["tau"].values
    assert metrics.pehe(tau, tau + 0.02) > metrics.pehe(tau, tau + 0.01)


def test_effect_correlation_is_one_for_a_perfect_estimate(experiment):
    assert metrics.effect_correlation(experiment["tau"], experiment["tau"]) == (
        pytest.approx(1.0)
    )


def test_spearman_ignores_monotone_rescaling(experiment):
    """Targeting depends on order, so a miscalibrated estimate still ranks."""
    tau = experiment["tau"].values
    assert metrics.effect_correlation(tau, 3 * tau + 1) == pytest.approx(1.0)
    assert metrics.effect_correlation(tau, 3 * tau + 1, method="pearson") == (
        pytest.approx(1.0)
    )


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        metrics.qini_curve([1, 0], [1, 0, 1], [0.1, 0.2])
    with pytest.raises(ValueError):
        metrics.pehe([0.1, 0.2], [0.1])


@pytest.mark.parametrize("k", [0.0, -0.1, 1.5])
def test_invalid_depth_is_rejected(experiment, k):
    with pytest.raises(ValueError):
        metrics.incremental_at_k(
            experiment["outcome"], experiment["treatment"], experiment["tau"], k
        )
