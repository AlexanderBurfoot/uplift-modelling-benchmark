"""Benchmark the learners against known individual treatment effects.

This is the experiment the repo exists for. Because the true effect is known
per person, each learner can be scored on estimation error (PEHE) as well as
on ranking quality (Qini), and the scenarios are chosen to put each learner
in the regime where it is expected to struggle.

    python scripts/run_synthetic.py

Writes results/synthetic_results.csv and prints the same table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uplift import metrics  # noqa: E402
from uplift.dgp import DGPConfig, feature_columns, generate  # noqa: E402
from uplift.learners import LEARNERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

SCENARIOS = {
    "baseline": dict(effect_size=0.05, heterogeneity=1.0, treatment_share=0.5),
    "small_effect": dict(effect_size=0.01, heterogeneity=1.0, treatment_share=0.5),
    "constant_effect": dict(effect_size=0.05, heterogeneity=0.0, treatment_share=0.5),
    "unbalanced_arms": dict(effect_size=0.05, heterogeneity=1.0, treatment_share=0.15),
    "sleeping_dogs": dict(
        effect_size=0.05, heterogeneity=1.0, treatment_share=0.5, sleeping_dog_share=0.25
    ),
}


def evaluate(scenario: str, params: dict, n: int, seed: int) -> list[dict]:
    frame = generate(DGPConfig(n=n, seed=seed, **params))
    columns = feature_columns(frame)

    train = frame.sample(frac=0.7, random_state=seed)
    test = frame.drop(train.index)
    X_train, X_test = train[columns].values, test[columns].values

    rows = []
    for name, cls in LEARNERS.items():
        model = cls().fit(X_train, train["treatment"], train["outcome"])
        predicted = model.predict_uplift(X_test)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "learner": name,
                "spearman": metrics.effect_correlation(test["tau"], predicted),
                "pehe": metrics.pehe(test["tau"], predicted),
                "qini_auc": metrics.qini_auc(
                    test["outcome"], test["treatment"], predicted
                ),
                "qini_vs_oracle": metrics.qini_coefficient(
                    test["outcome"], test["treatment"], predicted, test["tau"]
                ),
                "incremental_top_30pct": metrics.incremental_at_k(
                    test["outcome"], test["treatment"], predicted, 0.3
                ),
            }
        )

    rows.append(
        {
            "scenario": scenario,
            "seed": seed,
            "learner": "oracle",
            "spearman": 1.0,
            "pehe": 0.0,
            "qini_auc": metrics.qini_auc(test["outcome"], test["treatment"], test["tau"]),
            "qini_vs_oracle": 1.0,
            "incremental_top_30pct": metrics.incremental_at_k(
                test["outcome"], test["treatment"], test["tau"], 0.3
            ),
        }
    )
    return rows


def main(n: int = 20_000, seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario, params in SCENARIOS.items():
        for seed in seeds:
            rows.extend(evaluate(scenario, params, n, seed))
            print(f"  {scenario} seed={seed} done", flush=True)

    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "synthetic_results.csv", index=False)

    print("\nAveraged over seeds:\n")
    print(summarise(results).to_string())
    return results


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average the raw metrics, then form the oracle ratio from the averages.

    Averaging per-seed ratios instead would be wrong. In the low-signal
    scenarios the oracle's own Qini area is close to zero, so individual
    ratios are divisions by noise and their mean is dominated by whichever
    seed happened to have the smallest denominator. Pooling first keeps the
    denominator stable.
    """
    averaged = results.groupby(["scenario", "learner"])[
        ["spearman", "pehe", "qini_auc"]
    ].mean()
    oracle = averaged.xs("oracle", level="learner")["qini_auc"]
    scenario_index = averaged.index.get_level_values("scenario")
    averaged["qini_vs_oracle"] = (
        averaged["qini_auc"].values / oracle.reindex(scenario_index).values
    )
    return averaged.round(3)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
