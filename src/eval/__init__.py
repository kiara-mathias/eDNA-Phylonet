"""Clade-exclusion splitting + benchmarking package (Steps 3 and 6 of the
build plan).

Not yet implemented. Planned contents:
- A splitter that holds out entire families at 30%/50%/70% levels to
  simulate genuinely unseen species, saving the resulting splits as JSON
  (not regenerated with an unseeded RNG each run).
- A benchmarking harness comparing this system against a BLAST + QIIME2
  naive-Bayes baseline, and a no-phylogeny nearest-neighbor baseline (to
  isolate the contribution of the phylogeny/co-occurrence signal), all on
  the same held-out splits, reporting genus/family-level accuracy plus
  coverage/confidence curves.
"""
