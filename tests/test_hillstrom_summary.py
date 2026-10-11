"""Tests for the summaries in the Hillstrom script.

The summaries are what the README will quote, so they are checked on a
small frame where the right answer can be worked out by hand.
"""

import pandas as pd
import pytest

from scripts.run_hillstrom import (
    UNRANKED,
    average_effect,
    paired_differences,
    summarise,
)

QINI_BY_SPLIT = {
    "response": [10.0, 20.0, 30.0],
    "s-learner": [50.0, 40.0, 60.0],
    "x-learner": [52.0, 39.0, 65.0],
    UNRANKED: [0.0, 0.0, 0.0],
}


@pytest.fixture
def results():
    rows = [
        {
            "seed": seed,
            "learner": learner,
            "qini_auc": value,
            "incremental_top_30pct": value * 2,
            "profit_top_30pct": value * 3,
        }
        for learner, values in QINI_BY_SPLIT.items()
        for seed, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_summary_reports_mean_and_spread_across_splits(results):
    summary = summarise(results)
    assert summary.loc["response", "qini_auc_mean"] == pytest.approx(20.0)
    assert summary.loc["response", "qini_auc_std"] == pytest.approx(10.0)
    assert summary.loc["s-learner", "profit_top_30pct_mean"] == pytest.approx(150.0)


def test_summary_keeps_the_unranked_row(results):
    assert summarise(results).loc[UNRANKED, "qini_auc_mean"] == 0.0


def test_differences_are_taken_within_each_split(results):
    differences = paired_differences(results)
    row = differences.loc["x-learner minus s-learner"]
    assert row["mean_difference"] == pytest.approx(2.0)
    assert row["splits_ahead"] == 2
    assert row["splits"] == 3


def test_a_consistent_gap_shows_as_ahead_on_every_split(results):
    row = paired_differences(results).loc["s-learner minus response"]
    assert row["splits_ahead"] == row["splits"]


def test_the_unranked_row_is_not_compared_as_a_learner(results):
    comparisons = paired_differences(results).index
    assert not any(UNRANKED in comparison for comparison in comparisons)


def test_average_effect_is_the_difference_in_outcome_rates():
    frame = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "outcome": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )
    effect, half_width = average_effect(frame)
    assert effect == pytest.approx(0.5)
    assert half_width > 0
