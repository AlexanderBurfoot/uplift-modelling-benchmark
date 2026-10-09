"""Loader for the Hillstrom email dataset.

Kevin Hillstrom's MineThatData challenge data: 64,000 recent purchasers
randomly assigned to a men's email, a women's email, or no email, with
visit, conversion and spend measured over the following two weeks.

Used here because assignment is randomised. Downloaded on first use and
cached under ``data/raw``; not committed, since redistribution terms are
the author's to set.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

__all__ = ["load_hillstrom", "HILLSTROM_URL", "SEGMENTS"]

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

SEGMENTS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")

_CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "hillstrom.csv"

_CATEGORICAL = ("history_segment", "zip_code", "channel")


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(HILLSTROM_URL, timeout=60) as response:
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - surface the cause plainly
        raise RuntimeError(
            f"Could not download the Hillstrom dataset from {HILLSTROM_URL}. "
            f"Download it manually and save it to {destination}. Cause: {exc}"
        ) from exc
    frame = pd.read_csv(io.BytesIO(payload))
    frame.to_csv(destination, index=False)


def load_hillstrom(
    treatment_segment: str = "Womens E-Mail",
    outcome: str = "visit",
    path: Path | None = None,
    encode: bool = True,
) -> pd.DataFrame:
    """Load Hillstrom as a two-arm experiment.

    Args:
        treatment_segment: Which email arm to treat as the treatment. The
            third arm is dropped rather than pooled, because pooling two
            different creatives into one "treated" group estimates the
            average of two different effects and is hard to interpret.
        outcome: One of ``visit``, ``conversion`` or ``spend``. ``visit`` is
            the usual choice: conversion fires on roughly 0.9% of customers,
            which leaves very little signal for a heterogeneity model.
        path: Override the cache location.
        encode: One-hot encode the categorical columns. Set False to inspect
            the raw frame.

    Returns:
        A frame with feature columns plus ``treatment`` and ``outcome``.
    """
    if treatment_segment not in SEGMENTS[:2]:
        raise ValueError(f"treatment_segment must be one of {SEGMENTS[:2]}")
    if outcome not in {"visit", "conversion", "spend"}:
        raise ValueError("outcome must be 'visit', 'conversion' or 'spend'")

    destination = Path(path) if path else _CACHE
    if not destination.exists():
        _download(destination)

    raw = pd.read_csv(destination)
    keep = raw["segment"].isin([treatment_segment, "No E-Mail"])
    raw = raw.loc[keep].reset_index(drop=True)

    frame = pd.DataFrame(index=raw.index)
    frame["treatment"] = (raw["segment"] == treatment_segment).astype(int)
    frame["outcome"] = raw[outcome].astype(float)

    features = raw.drop(columns=["segment", "visit", "conversion", "spend"])
    if encode:
        features = pd.get_dummies(
            features, columns=list(_CATEGORICAL), drop_first=True, dtype=float
        )
    return pd.concat([features, frame], axis=1)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"treatment", "outcome"}]
