"""Render slice regressions for the terminal and as JSON."""

from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text

from slicemap.analyze import SliceRegression


def findings_to_json(findings: list[SliceRegression]) -> list[dict]:
    return [
        {
            "feature": f.feature,
            "slice": f.label,
            "size": f.size,
            "old_score": round(f.old_score, 6),
            "new_score": round(f.new_score, 6),
            "regression": round(f.regression, 6),
            "impact": round(f.impact, 6),
        }
        for f in findings
    ]


def render_table(findings: list[SliceRegression], limit: int = 20) -> Group:
    if not findings:
        return Group(Text("no slice regressed", style="green"))
    table = Table(box=None, pad_edge=False)
    table.add_column("feature", style="cyan")
    table.add_column("slice")
    table.add_column("size", justify="right")
    table.add_column("old", justify="right")
    table.add_column("new", justify="right")
    table.add_column("regression", justify="right")
    for f in findings[:limit]:
        table.add_row(
            f.feature,
            f.label,
            f"{f.size}",
            f"{f.old_score:.3f}",
            f"{f.new_score:.3f}",
            Text(f"-{f.regression:.3f}", style="bold red"),
        )
    if len(findings) > limit:
        table.add_row("...", f"{len(findings) - limit} more", "", "", "", "")
    return Group(table)
