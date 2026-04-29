from __future__ import annotations

import numpy as np
import pytest

from slicemap.metrics import get_metric, known_metrics, regression_amount
from slicemap.slicing import slices_for


def test_known_metrics():
    assert "accuracy" in known_metrics()
    assert "error" in known_metrics()


def test_accuracy_metric():
    m = get_metric("accuracy")
    assert m.higher_is_better
    assert m.score(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) == 0.75


def test_error_metric():
    m = get_metric("error")
    assert not m.higher_is_better
    assert m.score(np.array([1, 1, 0]), np.array([1, 0, 0])) == 1 / 3


def test_regression_amount_higher_is_better():
    m = get_metric("accuracy")
    assert regression_amount(m, 0.9, 0.8) == pytest.approx(0.1)  # dropped
    assert regression_amount(m, 0.8, 0.9) == 0.0  # improved


def test_regression_amount_lower_is_better():
    m = get_metric("error")
    assert regression_amount(m, 0.1, 0.2) == pytest.approx(0.1)  # error rose
    assert regression_amount(m, 0.2, 0.1) == 0.0  # error fell


def test_categorical_slices():
    values = np.array(["a", "b", "a", "c"])
    slices = slices_for("g", values)
    labels = {s.label for s in slices}
    assert labels == {"a", "b", "c"}
    a_slice = next(s for s in slices if s.label == "a")
    assert a_slice.mask.tolist() == [True, False, True, False]


def test_numeric_slices_partition_all_rows():
    values = np.arange(100.0)
    slices = slices_for("x", values, bins=4)
    covered = np.zeros(100, dtype=int)
    for s in slices:
        covered += s.mask.astype(int)
    assert np.all(covered == 1)  # every row in exactly one slice


def test_high_cardinality_categorical_skipped():
    values = np.array([str(i) for i in range(200)])
    assert slices_for("id", values) == []
