from __future__ import annotations

import numpy as np

from slicemap.analyze import analyze, has_regression, overall_scores
from slicemap.metrics import get_metric


def _data():
    # 200 rows, group A and group B; the new model breaks on group B.
    rng = np.random.default_rng(0)
    group = np.array(["A"] * 100 + ["B"] * 100)
    true = rng.integers(0, 2, size=200)
    old = true.copy()  # old is perfect
    new = true.copy()
    # corrupt half of group B's predictions in the new model
    new[150:] = 1 - new[150:]
    return true, old, new, {"group": group}


def test_analyze_finds_regressed_slice():
    true, old, new, feats = _data()
    findings = analyze(true, old, new, feats, get_metric("accuracy"), min_slice=10)
    assert findings
    worst = findings[0]
    assert worst.feature == "group"
    assert worst.label == "B"
    assert worst.new_score < worst.old_score


def test_analyze_no_regression_when_models_equal():
    true, old, _, feats = _data()
    findings = analyze(true, old, old, feats, get_metric("accuracy"), min_slice=10)
    assert findings == []
    assert not has_regression(findings)


def test_min_slice_filters_small_groups():
    true = np.array([0, 1, 0, 1])
    old = true.copy()
    new = 1 - true
    feats = {"g": np.array(["a", "a", "b", "b"])}
    findings = analyze(true, old, new, feats, get_metric("accuracy"), min_slice=10)
    assert findings == []  # every slice too small


def test_findings_sorted_by_impact():
    rng = np.random.default_rng(1)
    group = np.array(["A"] * 100 + ["B"] * 100)
    true = rng.integers(0, 2, size=200)
    old = true.copy()
    new = true.copy()
    new[180:] = 1 - new[180:]  # small break in B
    new[0:50] = 1 - new[0:50]  # bigger break in A
    findings = analyze(true, old, new, {"group": group}, get_metric("accuracy"), min_slice=10)
    assert findings[0].impact >= findings[-1].impact


def test_overall_scores():
    true = np.array([1, 1, 0, 0])
    old = np.array([1, 1, 0, 0])
    new = np.array([1, 0, 0, 0])
    old_s, new_s = overall_scores(true, old, new, get_metric("accuracy"))
    assert old_s == 1.0
    assert new_s == 0.75
