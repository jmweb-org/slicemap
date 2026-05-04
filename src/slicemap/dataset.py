"""Read a predictions file and pull out the columns the analysis needs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl


def read_frame(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pl.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pl.read_parquet(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pl.read_ndjson(path)
        if suffix == ".json":
            return pl.read_json(path)
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    raise ValueError(f"unsupported file type: {path.suffix or '(none)'}")


def column(frame: pl.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"column not found: {name}")
    return frame.get_column(name).to_numpy()


def feature_columns(
    frame: pl.DataFrame,
    names: Sequence[str] | None,
    exclude: Sequence[str],
) -> dict[str, np.ndarray]:
    if names:
        chosen = list(names)
    else:
        chosen = [c for c in frame.columns if c not in set(exclude)]
    out: dict[str, np.ndarray] = {}
    for name in chosen:
        if name not in frame.columns:
            raise ValueError(f"column not found: {name}")
        out[name] = frame.get_column(name).to_numpy()
    return out
