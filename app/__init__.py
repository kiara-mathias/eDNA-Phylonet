"""Dashboard package (Step 7 of the build plan).

- ``pipeline.py``: builds the deployed inference pipeline (encoder +
  ``HierarchicalFallback``, fit on nearly all known reference data, with a
  small random -- not clade-excluded -- validation slice for threshold
  calibration) and cleans pasted sequence/FASTA text.
- ``dashboard.py``: the Streamlit app itself -- a Classify tab for
  interactive inference (predicted taxonomy, per-rank confidence, novelty
  flag with top-k nearest known relatives) and a Benchmark Results tab
  rendering the Step 6 comparison against the BLAST/Naive Bayes/1-NN
  baselines.

There's no separately-trained model to pull from external storage (unlike
a typical deep learning deployment) -- the k-mer+PCA encoder and
centroid-based classifier are cheap enough to fit directly from
``data/processed/sequences.parquet`` at app startup (cached via
``st.cache_resource`` so it only happens once per process). Served by the
lightweight ``Dockerfile.inference`` image.
"""
