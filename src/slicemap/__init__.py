"""slicemap: find the data slices where a new model regressed against an old one."""

from slicemap.analyze import SliceRegression, analyze, overall_scores
from slicemap.metrics import Metric, get_metric, known_metrics
from slicemap.slicing import Slice, slices_for

__version__ = "0.1.0"

__all__ = [
    "Metric",
    "Slice",
    "SliceRegression",
    "__version__",
    "analyze",
    "get_metric",
    "known_metrics",
    "overall_scores",
    "slices_for",
]
