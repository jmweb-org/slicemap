"""Find the slices where the new predictions are worse than the old ones."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from slicemap.metrics import Metric, regression_amount
from slicemap.slicing import slices_for


@dataclass(frozen=True, slots=True)
class SliceRegression:
    feature: str
    label: str
    size: int
    old_score: float
    new_score: float
    regression: float

    @property
    def impact(self) -> float:
        return self.regression * self.size

    @property
    def sort_key(self) -> tuple[float, str, str]:
        return (-self.impact, self.feature, self.label)


def analyze(
    true: np.ndarray,
    old_pred: np.ndarray,
    new_pred: np.ndarray,
    features: Mapping[str, np.ndarray],
    metric: Metric,
    *,
    min_slice: int = 20,
    min_regression: float = 0.0,
) -> list[SliceRegression]:
    """Return slices where the metric got worse from old to new, worst first."""

    findings: list[SliceRegression] = []
    for name, values in features.items():
        for sl in slices_for(name, values):
            size = int(np.count_nonzero(sl.mask))
            if size < min_slice:
                continue
            old_score = metric.score(true[sl.mask], old_pred[sl.mask])
            new_score = metric.score(true[sl.mask], new_pred[sl.mask])
            amount = regression_amount(metric, old_score, new_score)
            if amount > min_regression:
                findings.append(
                    SliceRegression(
                        feature=name,
                        label=sl.label,
                        size=size,
                        old_score=old_score,
                        new_score=new_score,
                        regression=amount,
                    )
                )
    findings.sort(key=lambda f: f.sort_key)
    return findings


def overall_scores(
    true: np.ndarray, old_pred: np.ndarray, new_pred: np.ndarray, metric: Metric
) -> tuple[float, float]:
    return metric.score(true, old_pred), metric.score(true, new_pred)


def has_regression(findings: list[SliceRegression]) -> bool:
    return len(findings) > 0
