"""Clade-exclusion splitting + benchmarking package (Steps 3 and 6 of the
build plan).

Implemented:
- ``splitter.py``: holds out entire genera (see its module docstring for
  why genus rather than family, given this dataset's taxonomy) at
  30%/50%/70% levels to simulate genuinely unseen species, saving the
  resulting splits as JSON (not regenerated with an unseeded RNG each run).
- ``validate_baseline.py``: runs the Step 4 encoder+classifier against
  each saved split and reports per-rank (species/genus/family/order)
  accuracy -- the "validate this alone before adding fallback" checkpoint.

Not yet implemented: the full benchmarking harness (Step 6) comparing this
system against a BLAST + QIIME2 naive-Bayes baseline and a no-phylogeny
nearest-neighbor baseline, reporting coverage/confidence curves.
"""
