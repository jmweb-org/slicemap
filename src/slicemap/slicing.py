"""Turn a feature column into named slices (boolean masks over the rows).

Categorical columns slice by value; numeric columns slice by quantile bins.
Slicing is pure and works on NumPy arrays, so the analysis can be tested
without a dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BINS = 5
MAX_CATEGORIES = 50


@dataclass(frozen=True, slots=True)
class Slice:
    feature: str
    label: str
    mask: np.ndarray


def _is_numeric(values: np.ndarray) -> bool:
    return np.issubdtype(values.dtype, np.number)


def slices_for(feature: str, values: np.ndarray, *, bins: int = DEFAULT_BINS) -> list[Slice]:
    if _is_numeric(values):
        return _numeric_slices(feature, values, bins)
    return _categorical_slices(feature, values)


def _categorical_slices(feature: str, values: np.ndarray) -> list[Slice]:
    uniques = np.unique(values)
    if uniques.size > MAX_CATEGORIES:
        return []
    out: list[Slice] = []
    for value in uniques:
        out.append(Slice(feature, str(value), values == value))
    return out


def _numeric_slices(feature: str, values: np.ndarray, bins: int) -> list[Slice]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return [Slice(feature, f"== {edges[0]:g}", values == edges[0])]
    out: list[Slice] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
            label = f"[{lo:g}, {hi:g}]"
        else:
            mask = (values >= lo) & (values < hi)
            label = f"[{lo:g}, {hi:g})"
        out.append(Slice(feature, label, mask))
    return out
