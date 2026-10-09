"""Verify the README results table against results/synthetic_results.csv.

The most common way a portfolio repo embarrasses its author is a README
quoting numbers from a run that no longer matches the code. This recomputes
the summary from the committed results and compares it with the table parsed
out of the README, cell by cell.

Exits non-zero on any mismatch, so it can gate a push.

    python scripts/check_readme_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_synthetic import summarise  # noqa: E402

# The README quotes two decimal places, so a true value can legitimately sit
# half a unit of the last digit away. Slightly over half absorbs that plus
# float representation slack, while still catching any real drift between the
# prose and the committed results.
TOLERANCE = 0.0055

# README row label -> scenario key in the results file.
ROW_TO_SCENARIO = {
    "Baseline": "baseline",
    "Constant effect": "constant_effect",
    "Unbalanced arms": "unbalanced_arms",
    "sleeping dogs": "sleeping_dogs",
    "Small effect": "small_effect",
}

COLUMN_ORDER = ["response", "s-learner", "t-learner", "x-learner"]


def parse_readme_table(text: str) -> dict[str, list[float]]:
    """Pull the results table out of the README as {scenario: [values]}."""
    rows: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        label = cells[0]
        scenario = next(
            (key for phrase, key in ROW_TO_SCENARIO.items() if phrase in label), None
        )
        if scenario is None:
            continue
        try:
            # README uses a typographic minus sign for negatives.
            values = [float(c.replace("−", "-").strip("*")) for c in cells[1:]]
        except ValueError:
            continue
        rows[scenario] = values
    return rows


def main() -> int:
    results_path = ROOT / "results" / "synthetic_results.csv"
    readme_path = ROOT / "README.md"

    if not results_path.exists():
        print(f"FAIL  {results_path.name} missing; run scripts/run_synthetic.py")
        return 1

    computed = summarise(pd.read_csv(results_path))["qini_vs_oracle"]
    claimed = parse_readme_table(readme_path.read_text())

    missing = set(ROW_TO_SCENARIO.values()) - set(claimed)
    if missing:
        print(f"FAIL  README table is missing rows: {sorted(missing)}")
        return 1

    failures = 0
    for scenario, values in sorted(claimed.items()):
        for learner, stated in zip(COLUMN_ORDER, values, strict=True):
            actual = computed.loc[(scenario, learner)]
            if abs(actual - stated) > TOLERANCE:
                print(
                    f"FAIL  {scenario}/{learner}: README says {stated:+.2f}, "
                    f"results give {actual:+.2f}"
                )
                failures += 1

    if failures:
        print(f"\n{failures} mismatch(es) between README and results.")
        return 1

    cells = sum(len(v) for v in claimed.values())
    print(f"  README table matches results.csv ({cells} cells checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
