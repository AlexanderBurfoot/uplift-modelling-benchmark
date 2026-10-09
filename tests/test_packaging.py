"""Checks that the packaging metadata tells the truth.

A ``requires-python`` floor nobody tests is a number someone typed once.
These tests make the declared floor a property the suite enforces, so the
repo cannot quietly start using newer syntax while still claiming to run on
older interpreters.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = (ROOT / "pyproject.toml").read_text()


def declared_floor() -> tuple[int, int]:
    """Parse ``requires-python = ">=3.x"`` without needing tomllib.

    tomllib is 3.11+, and this test has to run on the floor itself.
    """
    match = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', PYPROJECT)
    assert match, "pyproject must declare requires-python as >=X.Y"
    return int(match.group(1)), int(match.group(2))


def source_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.py")
        if not any(part in {".venv", "venv", "build", ".git"} for part in path.parts)
    ]


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_source_parses_under_the_declared_floor(path):
    """Every file must be valid syntax on the oldest supported interpreter."""
    try:
        ast.parse(path.read_text(), feature_version=declared_floor())
    except SyntaxError as error:
        pytest.fail(
            f"{path.relative_to(ROOT)} uses syntax newer than "
            f"{'.'.join(map(str, declared_floor()))}: {error}"
        )


def test_numpy_floor_covers_the_api_actually_used():
    """metrics.py calls np.trapezoid, which arrived in numpy 2.0.

    The 1.x spelling was np.trapz, so a 1.x floor would install a numpy that
    cannot run this code.
    """
    import numpy as np

    assert hasattr(np, "trapezoid")
    match = re.search(r'"numpy>=(\d+)\.', PYPROJECT)
    assert match and int(match.group(1)) >= 2


def test_scipy_floor_covers_the_api_actually_used():
    """metrics.py reads .statistic off spearmanr's result object."""
    from scipy.stats import spearmanr

    assert hasattr(spearmanr([1, 2, 3], [1, 2, 3]), "statistic")


def test_every_tested_python_version_is_declared_in_classifiers():
    """CI matrix and the published classifiers must not drift apart."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    tested = set(re.findall(r'"(\d+\.\d+)"', workflow.split("python-version:")[1]))
    classified = set(re.findall(r"Python :: (\d+\.\d+)", PYPROJECT))
    assert tested <= classified, f"CI tests {sorted(tested - classified)}, not declared"
