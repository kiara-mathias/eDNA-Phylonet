"""CLI: fetch raw BOLD Systems barcode records into ``data/raw/`` and record
the exact pull in ``data/manifest.json``.

Usage::

    python -m src.ingest.fetch_bold --config configs/ingestion.yaml

No paths are hardcoded: everything is read from the YAML config and
resolved against the repo root (see ``src/common.py``), so this runs
identically on any machine or inside the Docker image.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `python src/ingest/fetch_bold.py`

from src.common import load_config, project_root, resolve_path  # noqa: E402
from src.ingest.bold_client import BoldClient  # noqa: E402

logger = logging.getLogger(__name__)

MARKER_FIELD_CANDIDATES = ("marker_code", "marker")


def _record_marker(record: dict[str, Any]) -> str | None:
    for field in MARKER_FIELD_CANDIDATES:
        value = record.get(field)
        if value:
            return value
    return None


def fetch(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full fetch and return the manifest dict that was written."""
    query_cfg = config["query"]
    api_cfg = config.get("api", {})
    output_cfg = config["output"]
    marker = config.get("marker")
    max_records = config.get("max_records")

    client = BoldClient(
        base_url=api_cfg.get("base_url", "https://portal.boldsystems.org/api"),
        timeout_seconds=api_cfg.get("timeout_seconds", 60),
    )

    query = query_cfg["scope"]
    extent = query_cfg.get("extent", "full")
    batch_size = api_cfg.get("batch_size", 1000)
    fmt = api_cfg.get("format", "json")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = resolve_path(output_cfg["raw_dir"]) / f"bold_{timestamp}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preprocessing + submitting query: %s (extent=%s)", query, extent)
    query_id, records = client.fetch_all(query, extent=extent, fmt=fmt)
    logger.info("Query accepted, query_id=%s", query_id)

    # The BOLD download endpoint streams the full matching result set as a
    # single NDJSON response with no server-side pagination (see
    # bold_client.py). We chunk it into fixed-size batch files client-side
    # here, both to keep individual files manageable and to preserve the
    # planned data/raw/bold_<timestamp>/batch_NNN.json layout.
    batch_files: list[str] = []
    total_kept = 0
    batch_index = 0
    current_batch: list[dict[str, Any]] = []
    stopped_early = False

    def _flush(batch: list[dict[str, Any]]) -> None:
        nonlocal batch_index
        if not batch:
            return
        batch_path = raw_dir / f"batch_{batch_index:04d}.json"
        with open(batch_path, "w", encoding="utf-8") as fh:
            json.dump(batch, fh)
        batch_files.append(str(batch_path.relative_to(project_root())))
        logger.info("Wrote %s (%d records)", batch_path.name, len(batch))
        batch_index += 1

    for record in records:
        if marker and _record_marker(record) != marker:
            continue
        current_batch.append(record)
        total_kept += 1

        if len(current_batch) >= batch_size:
            _flush(current_batch)
            current_batch = []

        if max_records is not None and total_kept >= max_records:
            stopped_early = True
            logger.info("Reached max_records=%d, stopping stream early.", max_records)
            break

    _flush(current_batch)

    manifest = {
        "source": "BOLD Systems Portal API",
        "api_base_url": api_cfg.get("base_url", "https://portal.boldsystems.org/api"),
        "query": query,
        "extent": extent,
        "marker_filter": marker,
        "query_id": query_id,
        "fetched_at_utc": timestamp,
        "batch_size": batch_size,
        "format": fmt,
        "max_records": max_records,
        "stopped_early_due_to_max_records": stopped_early,
        "total_records_kept": total_kept,
        "batch_files": batch_files,
    }

    manifest_path = resolve_path(output_cfg["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Wrote manifest: %s (%d records total)", manifest_path, total_kept)
    return manifest


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ingestion.yaml",
        help="Path to the ingestion YAML config (default: configs/ingestion.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    fetch(config)


if __name__ == "__main__":
    main()
