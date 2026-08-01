"""CLI: clean raw BOLD batches (from ``data/manifest.json``) into the single
source-of-truth table, ``data/processed/sequences.parquet``.

Usage::

    python -m src.preprocess.clean --config configs/ingestion.yaml

Reads every batch file listed in the manifest written by
``src/ingest/fetch_bold.py``, then:

1. Drops records missing species/genus/family/order or sequence.
2. Drops duplicate ``(species, sequence)`` pairs.
3. Drops sequences outside the configured length band or with too high a
   fraction of ambiguous (non-ACGT) bases.
4. Writes ``species, genus, family, order, sequence, lat, lon, process_id,
   bin_uri`` to a Parquet file, and logs a summary for sanity-checking
   against the project's target dataset size.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `python src/preprocess/clean.py`

import pandas as pd  # noqa: E402

from src.common import load_config, project_root, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)

_AMBIGUOUS_BASE_RE = re.compile(r"[^ACGT]", re.IGNORECASE)

# BCDM field names, confirmed against live BOLD Portal API responses.
# `lat`/`lon` are handled separately below since BOLD returns them as a
# single two-element `coord` array rather than separate fields.
_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "species": ("species", "species_name"),
    "genus": ("genus",),
    "family": ("family",),
    "order": ("order",),
    "sequence": ("nuc", "nucleotides", "sequence"),
    "process_id": ("processid", "process_id"),
    "bin_uri": ("bin_uri", "bin"),
}


def _get_field(record: dict[str, Any], field: str) -> Any:
    for key in _FIELD_CANDIDATES[field]:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _get_coords(record: dict[str, Any]) -> tuple[Any, Any]:
    coord = record.get("coord")
    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        return coord[0], coord[1]
    return record.get("lat"), record.get("lon")


def _load_raw_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in manifest["batch_files"]:
        batch_path = project_root() / rel_path
        with open(batch_path, "r", encoding="utf-8") as fh:
            batch = json.load(fh)
        records.extend(batch)
    return records


def _to_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        lat, lon = _get_coords(record)
        rows.append(
            {
                "species": _get_field(record, "species"),
                "genus": _get_field(record, "genus"),
                "family": _get_field(record, "family"),
                "order": _get_field(record, "order"),
                "sequence": _get_field(record, "sequence"),
                "lat": lat,
                "lon": lon,
                "process_id": _get_field(record, "process_id"),
                "bin_uri": _get_field(record, "bin_uri"),
            }
        )
    return rows


def _ambiguous_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    ambiguous = len(_AMBIGUOUS_BASE_RE.findall(sequence))
    return ambiguous / len(sequence)


def clean(config: dict[str, Any]) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the resulting DataFrame."""
    output_cfg = config["output"]
    cleaning_cfg = config["cleaning"]

    manifest_path = resolve_path(output_cfg["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Run "
            f"`python -m src.ingest.fetch_bold --config <config>` first."
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    records = _load_raw_records(manifest)
    logger.info("Loaded %d raw records from %d batch file(s)", len(records), len(manifest["batch_files"]))

    df = pd.DataFrame(_to_rows(records))
    n_start = len(df)

    required = ["species", "genus", "family", "order", "sequence"]
    df = df.dropna(subset=required)
    df = df[(df[required] != "").all(axis=1)]
    n_after_required = len(df)

    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    df = df.drop_duplicates(subset=["species", "sequence"])
    n_after_dedupe = len(df)

    min_len = cleaning_cfg["min_length"]
    max_len = cleaning_cfg["max_length"]
    seq_len = df["sequence"].str.len()
    df = df[(seq_len >= min_len) & (seq_len <= max_len)]
    n_after_length = len(df)

    max_ambig = cleaning_cfg["max_ambiguous_fraction"]
    ambig_frac = df["sequence"].map(_ambiguous_fraction)
    df = df[ambig_frac <= max_ambig]
    n_after_ambiguous = len(df)

    df = df.reset_index(drop=True)

    logger.info(
        "Cleaning summary: %d raw -> %d with required fields -> %d after dedupe -> "
        "%d after length filter [%d, %d] -> %d after ambiguous-base filter (<= %.2f%%)",
        n_start,
        n_after_required,
        n_after_dedupe,
        n_after_length,
        min_len,
        max_len,
        n_after_ambiguous,
        max_ambig * 100,
    )
    logger.info(
        "Final dataset: %d sequences, %d species, %d genera, %d families, %d orders",
        len(df),
        df["species"].nunique(),
        df["genus"].nunique(),
        df["family"].nunique(),
        df["order"].nunique(),
    )

    output_path = resolve_path(cleaning_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %s", output_path)

    return df


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
    clean(config)


if __name__ == "__main__":
    main()
