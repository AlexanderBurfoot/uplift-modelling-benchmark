"""Tests for the synthetic data generator.

Every oracle metric rests on the recorded ``tau`` being the quantity the
sampler used. Nothing else in the suite would catch a mismatch.
"""

import numpy as np
import pytest

from uplift.dgp import DGPConfig, feature_columns, generate


def test_recorded_tau_matches_potential_outcomes():
    frame = generate(DGPConfig(n=5_000, seed=1))
    np.testing.assert_allclose(frame["tau"], frame["p1"] - frame["p0"], atol=1e-12)


def test_mean_effect_tracks_configured_effect_size():
    frame = generate(DGPConfig(n=50_000, effect_size=0.05, seed=2))
    assert abs(frame["tau"].mean() - 0.05) < 0.015


def test_base_rate_is_respected():
    frame = generate(DGPConfig(n=50_000, base_rate=0.10, seed=3))
    assert abs(frame["p0"].mean() - 0.10) < 0.005


def test_zero_heterogeneity_gives_a_constant_effect():
    frame = generate(DGPConfig(n=5_000, heterogeneity=0.0, seed=4))
    assert frame["tau"].std() < 1e-9


def test_heterogeneity_widens_the_spread_of_effects():
    narrow = generate(DGPConfig(n=10_000, heterogeneity=0.5, seed=5))
    wide = generate(DGPConfig(n=10_000, heterogeneity=2.0, seed=5))
    assert wide["tau"].std() > narrow["tau"].std()


def test_sleeping_dogs_produce_negative_effects():
    frame = generate(DGPConfig(n=10_000, sleeping_dog_share=0.2, seed=6))
    assert (frame["tau"] < 0).mean() >= 0.15


def test_observed_outcome_rates_match_the_assigned_probabilities():
    frame = generate(DGPConfig(n=100_000, seed=7))
    treated = frame[frame["treatment"] == 1]
    control = frame[frame["treatment"] == 0]
    assert abs(treated["outcome"].mean() - treated["p1"].mean()) < 0.01
    assert abs(control["outcome"].mean() - control["p0"].mean()) < 0.01


def test_assignment_is_independent_of_features():
    """Randomisation check: no feature should separate the arms."""
    frame = generate(DGPConfig(n=50_000, seed=8))
    for column in feature_columns(frame):
        treated_mean = frame.loc[frame["treatment"] == 1, column].mean()
        control_mean = frame.loc[frame["treatment"] == 0, column].mean()
        assert abs(treated_mean - control_mean) < 0.05


def test_treatment_share_is_respected():
    frame = generate(DGPConfig(n=50_000, treatment_share=0.2, seed=9))
    assert abs(frame["treatment"].mean() - 0.2) < 0.01


def test_response_and_effect_features_are_disjoint():
    config = DGPConfig()
    assert not set(config.response_features) & set(config.effect_features)


def test_feature_columns_excludes_oracles():
    frame = generate(DGPConfig(n=100, seed=10))
    columns = feature_columns(frame)
    for leaked in ("tau", "p0", "p1", "treatment", "outcome"):
        assert leaked not in columns


def test_same_seed_reproduces_the_dataset():
    a = generate(DGPConfig(n=1_000, seed=11))
    b = generate(DGPConfig(n=1_000, seed=11))
    assert a.equals(b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"treatment_share": 0.0},
        {"treatment_share": 1.0},
        {"base_rate": 1.5},
        {"heterogeneity": -1.0},
        {"sleeping_dog_share": 2.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DGPConfig(**kwargs)
