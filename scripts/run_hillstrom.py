"""Run the learners on the Hillstrom email experiment.

Real randomised data, so the oracle metrics are unavailable and only the
ranking metrics apply. Downloads the dataset to data/raw on first run.

    python scripts/run_hillstrom.py

Writes results/hillstrom_results.csv and a Qini curve plot if matplotlib is
installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uplift import metrics  # noqa: E402
from uplift.data import feature_columns, load_hillstrom  # noqa: E402
from uplift.learners import LEARNERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Cost per email and value per conversion, used to turn a ranking into a
# decision. Both are stand-ins: Hillstrom publishes neither, and the point
# is the shape of the decision rule, not these particular numbers.
COST_PER_CONTACT = 0.10
VALUE_PER_VISIT = 2.00


def main(outcome: str = "visit", segment: str = "Womens E-Mail") -> pd.DataFrame:
    frame = load_hillstrom(treatment_segment=segment, outcome=outcome)
    columns = feature_columns(frame)
    print(f"{len(frame):,} customers, {frame['treatment'].mean():.1%} treated")
    print(f"outcome rate: {frame['outcome'].mean():.3%}")

    train = frame.sample(frac=0.7, random_state=0)
    test = frame.drop(train.index)
    X_train, X_test = train[columns].values, test[columns].values

    # The plain A/B result first. Every heterogeneity claim below has to be
    # read against this, and if the average effect is indistinguishable from
    # zero then ranking by predicted effect is ranking noise.
    treated = train[train["treatment"] == 1]["outcome"]
    control = train[train["treatment"] == 0]["outcome"]
    ate = treated.mean() - control.mean()
    se = (treated.var() / len(treated) + control.var() / len(control)) ** 0.5
    print(f"\naverage treatment effect: {ate:+.4f} (95% CI +/- {1.96 * se:.4f})")

    rows = []
    for name, cls in LEARNERS.items():
        model = cls().fit(X_train, train["treatment"], train["outcome"])
        predicted = model.predict_uplift(X_test)
        incremental = metrics.incremental_at_k(
            test["outcome"], test["treatment"], predicted, 0.3
        )
        contacts = 0.3 * len(test)
        rows.append(
            {
                "learner": name,
                "qini_auc": metrics.qini_auc(
                    test["outcome"], test["treatment"], predicted
                ),
                "incremental_top_30pct": incremental,
                "profit_top_30pct": incremental * VALUE_PER_VISIT
                - contacts * COST_PER_CONTACT,
            }
        )

    everyone = metrics.incremental_at_k(
        test["outcome"], test["treatment"], test["outcome"] * 0 + 1, 1.0
    )
    rows.append(
        {
            "learner": "target everyone",
            "qini_auc": 0.0,
            "incremental_top_30pct": everyone,
            "profit_top_30pct": everyone * VALUE_PER_VISIT
            - len(test) * COST_PER_CONTACT,
        }
    )

    results = pd.DataFrame(rows).round(3)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "hillstrom_results.csv", index=False)
    print("\n" + results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
