"""Run the learners on the Hillstrom email experiment.

Real randomised data, so the oracle metrics are unavailable and only the
ranking metrics apply. Downloads the dataset to data/raw on first run.

    python scripts/run_hillstrom.py

Repeats the train-test split over several seeds, because on one split the
ordering of the uplift learners is decided by noise. Writes one row per
split and learner to results/hillstrom_results.csv and prints the summary.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uplift import metrics  # noqa: E402
from uplift.data import feature_columns, load_hillstrom  # noqa: E402
from uplift.learners import LEARNERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Cost per email and value per visit, used to turn a ranking into a
# decision. Both are invented: Hillstrom publishes neither, and the point
# is the shape of the decision rule, not these particular numbers.
COST_PER_CONTACT = 0.10
VALUE_PER_VISIT = 2.00

TRAIN_FRACTION = 0.7
TARGET_FRACTION = 0.3
EVERYONE = 1.0
SPLIT_SEEDS = tuple(range(20))
Z_95 = 1.96

UNRANKED = "target everyone"
SUMMARY_METRICS = ["qini_auc", "incremental_top_30pct", "profit_top_30pct"]


def average_effect(frame: pd.DataFrame) -> tuple[float, float]:
    """Difference in outcome rate between the arms and its 95% half-width.

    Uses every customer. No model is involved, so holding part of the
    sample out would only widen the interval.
    """
    treated = frame[frame["treatment"] == 1]["outcome"]
    control = frame[frame["treatment"] == 0]["outcome"]
    effect = treated.mean() - control.mean()
    standard_error = (
        treated.var() / len(treated) + control.var() / len(control)
    ) ** 0.5
    return effect, Z_95 * standard_error


def targeting_row(
    name: str, seed: int, test: pd.DataFrame, score, depth: float, qini_auc: float
) -> dict:
    """Incremental visits and illustrative profit from targeting to ``depth``.

    Incremental visits are counted over the treated customers within the
    targeted group, so the contact cost is charged for those customers only.
    """
    incremental = metrics.incremental_at_k(
        test["outcome"], test["treatment"], score, depth
    )
    contacts = metrics.treated_at_k(test["treatment"], score, depth)
    return {
        "seed": seed,
        "learner": name,
        "qini_auc": qini_auc,
        "incremental_top_30pct": incremental,
        "profit_top_30pct": incremental * VALUE_PER_VISIT
        - contacts * COST_PER_CONTACT,
    }


def evaluate(frame: pd.DataFrame, columns: list[str], seed: int) -> list[dict]:
    """Fit every learner on one train-test split and score it on the test."""
    train = frame.sample(frac=TRAIN_FRACTION, random_state=seed)
    test = frame.drop(train.index)
    X_train, X_test = train[columns].values, test[columns].values

    rows = []
    for name, cls in LEARNERS.items():
        model = cls().fit(X_train, train["treatment"], train["outcome"])
        predicted = model.predict_uplift(X_test)
        qini_auc = metrics.qini_auc(test["outcome"], test["treatment"], predicted)
        rows.append(
            targeting_row(name, seed, test, predicted, TARGET_FRACTION, qini_auc)
        )

    # A constant score is deliberate: at full depth the Qini endpoint does
    # not depend on the ranking. Targeting everyone ranks nobody, so its
    # Qini area is zero by definition rather than computed.
    unranked = test["outcome"] * 0 + 1
    rows.append(targeting_row(UNRANKED, seed, test, unranked, EVERYONE, 0.0))
    return rows


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across splits.

    The standard deviation describes how far one split can sit from another.
    It is not a standard error: the splits reuse the same customers, so they
    are not independent draws.
    """
    grouped = results.groupby("learner", sort=False)[SUMMARY_METRICS]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    """Qini area of each learner against each other, split by split.

    Both learners in a pair are scored on the same test customers, so the
    difference removes the variation that comes from which customers landed
    in the test set and leaves the part that separates the learners.
    """
    ranked = results[results["learner"] != UNRANKED]
    by_split = ranked.pivot(index="seed", columns="learner", values="qini_auc")
    learners = [name for name in LEARNERS if name in by_split.columns]

    rows = []
    for earlier, later in combinations(learners, 2):
        difference = by_split[later] - by_split[earlier]
        rows.append(
            {
                "comparison": f"{later} minus {earlier}",
                "mean_difference": difference.mean(),
                "sd_difference": difference.std(),
                "splits_ahead": int((difference > 0).sum()),
                "splits": len(difference),
            }
        )
    return pd.DataFrame(rows).set_index("comparison")


def main(
    outcome: str = "visit",
    segment: str = "Womens E-Mail",
    seeds: tuple[int, ...] = SPLIT_SEEDS,
) -> pd.DataFrame:
    frame = load_hillstrom(treatment_segment=segment, outcome=outcome)
    columns = feature_columns(frame)
    print(f"{len(frame):,} customers, {frame['treatment'].mean():.1%} treated")
    print(f"outcome rate: {frame['outcome'].mean():.3%}")

    # The plain A/B result first. Every heterogeneity claim below has to be
    # read against this, and if the average effect is indistinguishable from
    # zero then ranking by predicted effect is ranking noise.
    effect, half_width = average_effect(frame)
    print(f"\naverage treatment effect: {effect:+.4f} (95% CI +/- {half_width:.4f})\n")

    rows: list[dict] = []
    for seed in seeds:
        rows.extend(evaluate(frame, columns, seed))
        print(f"  split seed={seed} done", flush=True)

    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "hillstrom_results.csv", index=False)

    print(f"\nAcross {len(seeds)} splits:\n")
    print(summarise(results).round(2).to_string())
    print("\nPaired differences in Qini area:\n")
    print(paired_differences(results).round(2).to_string())
    return results


if __name__ == "__main__":
    main()
