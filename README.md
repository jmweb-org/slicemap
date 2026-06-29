# slicemap

Find the data slices where a new model regressed against an old one. Headless,
file in, table out.

A new model can lift the overall metric while quietly getting worse on a
segment that matters: one country, one age band, one product category. An
aggregate number hides it. `slicemap` takes a predictions file with both
models' outputs and the features, scores every slice, and lists the ones where
the new model lost ground, ranked by how many rows are affected.

```console
$ slicemap compare preds.parquet --true label --old pred_v1 --new pred_v2
slicemap: accuracy overall 0.910 -> 0.918
feature   slice        size   old     new    regression
country   BR            842   0.904   0.731   -0.173
device    tablet        311   0.880   0.795   -0.085
age       [55, 70)      540   0.901   0.860   -0.041
```

## Install

```console
$ pip install slicemap
```

Reads one CSV, Parquet or JSON Lines file containing the truth column, both
prediction columns, and the feature columns.

## Usage

```console
$ slicemap compare preds.parquet --true y --old pred_a --new pred_b
$ slicemap compare preds.csv --true y --old a --new b --features country,age
$ slicemap compare preds.csv --true y --old a --new b --metric error
$ slicemap compare preds.csv --true y --old a --new b --min-slice 50
$ slicemap compare preds.csv --true y --old a --new b --json
$ slicemap compare preds.csv --true y --old a --new b --check
```

If `--features` is omitted, every column except the truth and prediction columns
is treated as a feature.

### JSON output schema

`--json` writes a single object to stdout:

```json
{
  "metric": "accuracy",
  "old_overall": 0.910,
  "new_overall": 0.918,
  "regressions": [
    {
      "feature": "country",
      "slice": "BR",
      "size": 842,
      "old_score": 0.904,
      "new_score": 0.731,
      "regression": 0.173,
      "impact": 145.766
    }
  ]
}
```

| Field | Type | Description |
| --- | --- | --- |
| `metric` | string | Metric name used for scoring (e.g. `"accuracy"`) |
| `old_overall` | number | Overall metric score for the old model |
| `new_overall` | number | Overall metric score for the new model |
| `regressions` | array | Slices where the new model is worse, sorted by `impact` descending |
| `regressions[].feature` | string | Column name the slice is drawn from |
| `regressions[].slice` | string | Slice label (a category value or a quantile bin like `"[55, 70)"`) |
| `regressions[].size` | integer | Number of rows in the slice |
| `regressions[].old_score` | number | Old model's metric score on this slice |
| `regressions[].new_score` | number | New model's metric score on this slice |
| `regressions[].regression` | number | Absolute degradation (always positive) |
| `regressions[].impact` | number | `regression × size` — used for ranking |

All numeric values are rounded to six decimal places.

### In CI

Fail a model update when any slice regresses:

```yaml
- run: slicemap compare preds.parquet --true y --old champion --new challenger --check
```

## How slicing works

Categorical features slice by value; numeric features slice by quantile bins.
For each slice the metric is computed for both models, and the slice is flagged
when the new model is worse. Slices smaller than `--min-slice` are skipped to
avoid noise, and findings are ranked by **impact** (regression size times the
number of rows), so the segments worth fixing first come first.

## Metrics

| Metric | Direction |
| --- | --- |
| `accuracy` | higher is better |
| `error` | lower is better |
| `mae` | lower is better |

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Compared; no slice regressed (or `--check` not set) |
| 1 | `--check` found at least one regressed slice |
| 2 | A column was missing, the metric is unknown, or the file is unsupported |

## License

MIT. See [LICENSE](LICENSE).
