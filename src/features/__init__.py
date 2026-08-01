"""Sequence embedding / feature-encoding package (Step 4 of the build plan).

Implemented: ``kmer_pca.py``'s ``KmerPCAEncoder`` -- a k-mer frequency
vector + PCA encoder, fast to fit/validate the rest of the pipeline
(classifier, splitter, benchmarking) end-to-end on CPU with no training
required.

Not yet implemented: a learned encoder (1D CNN or small Transformer over
nucleotide/k-mer tokens), to be swapped in later behind the same
``fit(sequences)`` / ``transform(sequences) -> np.ndarray`` interface, still
targeting CPU-feasible model sizes per the current compute constraints.
"""
