# Compare predictions from inside a container by mounting the file:
#
#   docker build -t slicemap .
#   docker run --rm -v "$PWD:/w" -w /w slicemap compare preds.parquet \
#     --true y --old a --new b
#
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/jmweb-org/slicemap"
LABEL org.opencontainers.image.description="Find the data slices where a new model regressed against an old one."
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

ENTRYPOINT ["slicemap"]
CMD ["--help"]
