"""Base classifier package (Step 4 of the build plan).

Implemented: ``classifier.py``'s ``Paper2Classifier`` -- the base algorithm
from Paper 2 (PLOS Computational Biology, Dec 2025): zero-shot annotation
of eDNA reads using phylogeny structure (taxonomic hierarchy, since we
don't have branch-length phylogenies) plus species co-occurrence (here,
geographic proximity, the real per-record signal available from BOLD).
Consumes embeddings from ``src/features`` and the taxonomy table produced
by ``src/preprocess/clean.py``, and outputs per-rank (species/genus/family/
order) predictions plus a full distance vector, which ``src/fallback``
will consume once built.

Not yet implemented: incorporating Paper 1's (IEEE Access 2024, DOI
10.1109/ACCESS.2024.3450016) robustness insight for messy/incomplete input
data, and swapping the k-mer+PCA embedding for a learned encoder.
"""
