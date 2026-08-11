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
- ``validate_fallback.py``: runs the Step 5 encoder + ``HierarchicalFallback``
  against each saved split and reports novelty-detection recall, false-flag
  rate, abstention rate, and resolved-rank accuracy.
- ``validate_calibration.py``: dumps per-query confidence+correctness on
  held-out genera, builds reliability diagrams, reports ECE per rank, and
  fits Platt/isotonic recalibrators when ECE exceeds the configured threshold.
- ``calibration.py``: ECE / reliability-bin helpers and post-hoc calibrators.
- ``benchmark.py``: runs the full Step 6 comparison -- our system vs. the
  three ``src/baselines`` -- reporting per-rank coverage + accuracy-among-
  answered across all three holdout levels.

Not yet implemented: coverage/confidence *curves* (varying calibration
thresholds and plotting the tradeoff) -- this pass reports point estimates
at the currently-configured thresholds only.
"""
