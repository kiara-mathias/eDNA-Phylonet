"""Base classifier package (Step 4 of the build plan).

Not yet implemented. Implements the base algorithm from Paper 2 (PLOS
Computational Biology, Dec 2025): zero-shot annotation of eDNA reads using
phylogeny structure (taxonomic hierarchy) plus species co-occurrence,
classifying reads directly from raw sequences -- including species never
seen during training.

Also incorporates the robustness insight from Paper 1 (IEEE Access 2024,
DOI 10.1109/ACCESS.2024.3450016) for handling messy/incomplete input data.

Expected interface (to be added): a classifier that consumes embeddings
from ``src/features`` and the taxonomy table produced by
``src/preprocess/clean.py``, and outputs per-rank (species/genus/family/
order) logits or distances, which ``src/fallback`` then consumes.
"""
