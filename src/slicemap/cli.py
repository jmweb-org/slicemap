"""Command-line interface for slicemap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from slicemap import __version__
from slicemap.analyze import analyze, has_regression, overall_scores
from slicemap.dataset import column, feature_columns, read_frame
from slicemap.metrics import get_metric, known_metrics
from slicemap.render import findings_to_json, render_table

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Find the data slices where a new model regressed against an old one.",
)
_out = Console()
_err = Console(stderr=True)

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_BAD_INPUT = 2


def _version_callback(value: bool) -> None:
    if value:
        _out.print(f"slicemap {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """slicemap command-line interface."""


@app.command("compare")
def compare(
    data: Path = typer.Argument(..., help="Predictions file with truth, old, new and features."),
    true: str = typer.Option(..., "--true", help="Ground-truth column."),
    old: str = typer.Option(..., "--old", help="Old model prediction column."),
    new: str = typer.Option(..., "--new", help="New model prediction column."),
    metric: str = typer.Option(
        "accuracy", "--metric", help=f"One of: {', '.join(known_metrics())}."
    ),
    features: str = typer.Option(None, "--features", help="Comma-separated feature columns."),
    min_slice: int = typer.Option(20, "--min-slice", help="Ignore slices smaller than this."),
    as_json: bool = typer.Option(False, "--json", help="Emit findings as JSON."),
    check: bool = typer.Option(False, "--check", help="Exit non-zero if any slice regressed."),
) -> None:
    """Report the slices where the new model is worse than the old one."""

    try:
        frame = read_frame(data)
        metric_obj = get_metric(metric)
        true_arr = column(frame, true)
        old_arr = column(frame, old)
        new_arr = column(frame, new)
        feature_names = [f.strip() for f in features.split(",")] if features else None
        feats = feature_columns(frame, feature_names, exclude=[true, old, new])
    except (OSError, ValueError) as exc:
        _err.print(f"slicemap: {exc}")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    findings = analyze(true_arr, old_arr, new_arr, feats, metric_obj, min_slice=min_slice)
    old_overall, new_overall = overall_scores(true_arr, old_arr, new_arr, metric_obj)

    if as_json:
        _out.print_json(
            json.dumps(
                {
                    "metric": metric_obj.name,
                    "old_overall": round(old_overall, 6),
                    "new_overall": round(new_overall, 6),
                    "regressions": findings_to_json(findings),
                }
            )
        )
    else:
        _err.print(f"slicemap: {metric_obj.name} overall {old_overall:.3f} -> {new_overall:.3f}")
        _out.print(render_table(findings))

    if check and has_regression(findings):
        raise typer.Exit(EXIT_REGRESSION)


def entrypoint() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("slicemap: interrupted", file=sys.stderr)
        raise SystemExit(130) from None
