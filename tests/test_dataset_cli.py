from __future__ import annotations

import json

import polars as pl
import pytest
from typer.testing import CliRunner

from slicemap import __version__
from slicemap import cli as cli_module
from slicemap.dataset import column, feature_columns, read_frame

runner = CliRunner()


def _frame():
    return pl.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4,
            "old": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4,
            "new": [0, 1, 0, 1] + [1, 0] * 18,
            "group": (["A"] * 20) + (["B"] * 20),
        }
    )


def _csv(tmp_path, frame):
    path = tmp_path / "preds.csv"
    frame.write_csv(path)
    return path


def test_read_unsupported(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError):
        read_frame(path)


def test_column_and_features():
    frame = _frame()
    assert column(frame, "y").shape[0] == 40
    feats = feature_columns(frame, None, exclude=["y", "old", "new"])
    assert set(feats) == {"group"}


def test_feature_columns_unknown_raises():
    with pytest.raises(ValueError):
        feature_columns(_frame(), ["nope"], exclude=[])


def test_version():
    result = runner.invoke(cli_module.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_compare_reports_regression_json(tmp_path):
    path = _csv(tmp_path, _frame())
    result = runner.invoke(
        cli_module.app,
        [
            "compare",
            str(path),
            "--true",
            "y",
            "--old",
            "old",
            "--new",
            "new",
            "--min-slice",
            "5",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metric"] == "accuracy"
    assert any(r["feature"] == "group" for r in payload["regressions"])


def test_compare_check_fails_on_regression(tmp_path):
    path = _csv(tmp_path, _frame())
    result = runner.invoke(
        cli_module.app,
        [
            "compare",
            str(path),
            "--true",
            "y",
            "--old",
            "old",
            "--new",
            "new",
            "--min-slice",
            "5",
            "--check",
        ],
    )
    assert result.exit_code == cli_module.EXIT_REGRESSION


def test_compare_no_regression_passes(tmp_path):
    frame = _frame().with_columns(pl.col("old").alias("new"))
    path = _csv(tmp_path, frame)
    result = runner.invoke(
        cli_module.app,
        [
            "compare",
            str(path),
            "--true",
            "y",
            "--old",
            "old",
            "--new",
            "new",
            "--min-slice",
            "5",
            "--check",
        ],
    )
    assert result.exit_code == 0


def test_compare_unknown_metric_is_bad_input(tmp_path):
    path = _csv(tmp_path, _frame())
    result = runner.invoke(
        cli_module.app,
        ["compare", str(path), "--true", "y", "--old", "old", "--new", "new", "--metric", "nope"],
    )
    assert result.exit_code == cli_module.EXIT_BAD_INPUT
