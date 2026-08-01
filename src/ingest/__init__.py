"""Data ingestion package.

Step 1 of the build plan: pull raw DNA barcode records (specimen + sequence +
taxonomy metadata) from public sources.

Implemented:
- ``bold_client.py``: thin client for the BOLD Systems Portal API
  (preprocessor -> query -> paginated document download).
- ``fetch_bold.py``: CLI entrypoint that drives ``bold_client`` from a YAML
  config, writes raw batches to ``data/raw/`` and records the exact query in
  ``data/manifest.json`` for reproducibility.

Not yet implemented: an NCBI GenBank ingestion path. The current scope
targets BOLD only (COI-5P barcodes for Actinopterygii), since BOLD alone
comfortably covers the ~500-1000 species / 10K-50K sequence target.
"""
