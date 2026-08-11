"""Dashboard package (Step 7 of the build plan).

- ``pipeline.py``: builds the deployed inference pipeline (encoder +
  ``HierarchicalFallback``, fit on nearly all known reference data, with a
  small random -- not clade-excluded -- validation slice for threshold
  calibration) and cleans pasted sequence/FASTA text.
- ``dashboard.py``: Identify / Evidence / Method tabs (live classification,
  calibration + BLAST-lie gallery, FCW + clade-exclusion tables).
  ``gallery_examples.py`` holds the hardcoded held-out contrasts;
  ``dashboard_charts.py`` is the testable chart math.

There's no separately-trained model to pull from external storage (unlike
a typical deep learning deployment) -- the k-mer+PCA encoder and
centroid-based classifier are cheap enough to fit directly from
``data/processed/sequences.parquet`` at app startup (cached via
``st.cache_resource`` so it only happens once per process). Served by the
lightweight ``Dockerfile.inference`` image.
"""
