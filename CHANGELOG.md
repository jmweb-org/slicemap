# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-08

### Added
- Docker image and a published container entry point.
- Continuous integration across Python 3.10, 3.11 and 3.12.
- Expanded documentation and a metrics reference.

## [0.1.0] - 2026-05-04

### Added
- `compare` command: score every data slice for two models and report the ones
  where the new model regressed, ranked by impact, with a `--check` CI gate.
- Categorical and quantile-binned numeric slicing.
- `accuracy`, `error` and `mae` metrics, with `--min-slice` to drop small
  segments and `--features` to restrict the columns considered.
- CSV, Parquet and JSON Lines input through polars.

[0.2.0]: https://github.com/jmweb-org/slicemap/releases/tag/v0.2.0
[0.1.0]: https://github.com/jmweb-org/slicemap/releases/tag/v0.1.0
