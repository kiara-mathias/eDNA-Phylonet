"""Sequence embedding / feature-encoding package (Step 4 of the build plan).

Implemented:
- ``kmer_pca.py``: k-mer frequency + PCA (default, no training).
- ``conv_triplet.py``: small 1D CNN + triplet loss (numpy, CPU).
- ``dnabert.py``: optional frozen Hugging Face DNA embeddings (no fine-tune).
- ``encoder.py``: ``build_encoder(config)`` so eval/dashboard swap encoders
  without touching ``HierarchicalFallback``.
"""
