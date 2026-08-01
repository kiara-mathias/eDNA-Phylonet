"""Sequence embedding / feature-encoding package (Step 4 of the build plan).

Not yet implemented. Planned approach, per project spec:
1. Start with a k-mer frequency vector + PCA encoder to validate the rest of
   the pipeline (classifier, splitter, benchmarking) end-to-end quickly on
   CPU, with no training required.
2. Swap in a learned encoder (1D CNN or small Transformer over
   nucleotide/k-mer tokens) once the pipeline is validated, still targeting
   CPU-feasible model sizes per the current compute constraints.

Expected interface (to be added): an ``Encoder`` with ``fit(sequences)`` and
``transform(sequences) -> np.ndarray`` methods, so the k-mer+PCA and learned
encoders are interchangeable behind the same call site in ``src/model``.
"""
