"""Tests for the Hillstrom loader.

None of these touch the network: the loader is pointed at a small frame
with the real column layout, and the download is replaced where it matters.
"""

import urllib.request

import pandas as pd
import pytest

from uplift import data
from uplift.data import feature_columns, load_hillstrom

ROWS_PER_SEGMENT = 4


@pytest.fixture
def raw_frame():
    rows = [
        {
            "recency": index + 1,
            "history_segment": "1) $0 - $100" if index % 2 else "2) $100 - $200",
            "history": 50.0 + index,
            "mens": index % 2,
            "womens": 1 - index % 2,
            "zip_code": "Urban" if index % 2 else "Rural",
            "newbie": index % 2,
            "channel": "Web" if index % 2 else "Phone",
            "segment": segment,
            "visit": index % 2,
            "conversion": 0,
            "spend": 12.5 * (index % 2),
        }
        for segment in data.SEGMENTS
        for index in range(ROWS_PER_SEGMENT)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def cached(raw_frame, tmp_path):
    path = tmp_path / "hillstrom.csv"
    raw_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_the_other_email_arm_is_dropped_not_pooled(cached):
    frame = load_hillstrom(path=cached)
    assert len(frame) == 2 * ROWS_PER_SEGMENT
    assert frame["treatment"].sum() == ROWS_PER_SEGMENT


def test_no_outcome_column_is_offered_as_a_feature(cached):
    columns = feature_columns(load_hillstrom(path=cached))
    assert not {"segment", "visit", "conversion", "spend"} & set(columns)
    assert not {"treatment", "outcome"} & set(columns)


def test_spend_is_rejected_because_the_learners_are_classifiers(cached):
    with pytest.raises(ValueError, match="spend"):
        load_hillstrom(path=cached, outcome="spend")


def test_a_file_placed_by_hand_is_used_without_downloading(cached, no_network):
    assert len(load_hillstrom(path=cached)) == 2 * ROWS_PER_SEGMENT


def test_a_failed_download_says_where_to_put_the_file(tmp_path, no_network):
    destination = tmp_path / "raw" / "hillstrom.csv"
    with pytest.raises(RuntimeError, match=str(destination)):
        load_hillstrom(path=destination)
    assert not destination.exists()


def test_a_response_that_is_not_the_dataset_is_not_cached(tmp_path, monkeypatch):
    class NotTheDataset:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return b"title,body\nMoved,This page has moved\n"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: NotTheDataset())
    destination = tmp_path / "hillstrom.csv"
    with pytest.raises(RuntimeError, match="Download it manually"):
        load_hillstrom(path=destination)
    assert not destination.exists()
