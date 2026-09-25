# ---------------------------------------------------------------------------
# TabrizOTC-ML — container image (CLI study tool)
# ---------------------------------------------------------------------------
# Build:  docker build -t tabriz-otc-ml .
# Run the full study on the synthetic benchmark (results mounted to ./results):
#   docker run --rm -v "$PWD/results:/app/results" tabriz-otc-ml
# Fast demo:
#   docker run --rm -v "$PWD/results:/app/results" tabriz-otc-ml \
#       --n-trials 3 --out results/
# With your own field-data CSV:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#       tabriz-otc-ml --csv data/my_measurements.csv --out results/
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Dependency layer (cached until requirements.txt changes) ---------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- Application layer ------------------------------------------------------
COPY otc_ml ./otc_ml
COPY scripts ./scripts
COPY tests ./tests
COPY README.md LICENSE CITATION.cff pyproject.toml ./

# --- Non-root user; results volume writable ---------------------------------
RUN useradd --create-home --uid 1000 otcuser \
    && mkdir -p /app/results /app/data \
    && chown -R otcuser:otcuser /app
USER otcuser

VOLUME ["/app/results", "/app/data"]

# Health check = the package imports and the CLI responds
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=2 \
    CMD python -c "import otc_ml" || exit 1

ENTRYPOINT ["python", "scripts/run_study.py"]
CMD ["--out", "results/"]
