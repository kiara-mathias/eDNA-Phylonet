# CPU-only image for data ingestion / cleaning (Steps 1-2 of the build plan).
# No trainable model exists yet, so this has no fixed entrypoint command --
# use it as a reproducible shell to run the ingest/preprocess CLIs in.
#
# NOTE: this Dockerfile has not been built/validated locally (no Docker
# installed on the authoring machine). It is validated instead by
# .github/workflows/docker-build.yml on every push, once this repo is
# pushed to GitHub.
#
# Pinned to 3.12 (not just "3.11+") to match the pinned numpy/scipy/
# scikit-learn versions in requirements.txt, which only ship prebuilt
# wheels for Python 3.12+ -- python:3.11-slim has no compiler toolchain,
# so pip would otherwise try (and fail) to build them from source.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app

WORKDIR /app

# ncbi-blast+ provides makeblastdb/blastn for the Step 6 BLAST baseline
# (src/baselines/blast_baseline.py). Not a Python dependency, so it can't
# go in requirements.txt -- installed here instead.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
