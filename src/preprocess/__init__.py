"""Cleaning / QC package.

Step 2 of the build plan: turn raw BOLD batches into the single source of
truth table, ``data/processed/sequences.parquet``, with columns
``species, genus, family, order, sequence, lat, lon, process_id, bin_uri``.

Implemented:
- ``clean.py``: dedupes, filters ambiguous-base/length outliers, and writes
  the parquet table.

Not yet implemented: QIIME2/DADA2-based amplicon cleaning. Per the project
spec this is only needed for raw multi-specimen amplicon data; since BOLD
records are already assembled single-specimen barcodes, that stage is
skipped for this dataset.
"""
