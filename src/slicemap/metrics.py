"""Scoring metrics for comparing predictions against ground truth.

Each metric reports a score and whether higher is better, so the analysis can
decide what "got worse" means without special-casing each metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]

    def score(self, true: np.ndarray, pred: np.ndarray) -> float:
        return float(self.fn(true, pred))


def _accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size == 0:
        return 0.0
    return float(np.mean(true == pred))


def _error_rate(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size == 0:
        return 0.0
    return float(np.mean(true != pred))


def _mae(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size == 0:
        return 0.0
    return float(np.mean(np.abs(true.astype(float) - pred.astype(float))))


_METRICS: dict[str, Metric] = {
    "accuracy": Metric("accuracy", True, _accuracy),
    "error": Metric("error", False, _error_rate),
    "mae": Metric("mae", False, _mae),
}


def known_metrics() -> list[str]:
    return sorted(_METRICS)


def get_metric(name: str) -> Metric:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"unknown metric: {name}") from exc


def regression_amount(metric: Metric, old: float, new: float) -> float:
    """How much worse ``new`` is than ``old``; zero if it did not get worse."""

    delta = new - old if metric.higher_is_better else old - new
    return max(0.0, -delta)


def as_array(values: Sequence) -> np.ndarray:
    return np.asarray(values)
