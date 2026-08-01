# CPU-only image for data ingestion / cleaning (Steps 1-2 of the build plan).
# No trainable model exists yet, so this has no fixed entrypoint command --
# use it as a reproducible shell to run the ingest/preprocess CLIs in.
#
# NOTE: this Dockerfile has not been built/validated locally (no Docker
# installed on the authoring machine). It is validated instead by
# .github/workflows/docker-build.yml on every push, once this repo is
# pushed to GitHub.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
