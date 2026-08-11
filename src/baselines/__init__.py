"""Benchmarking baselines (Step 6 of the build plan).

Three comparison points for ``src/eval/benchmark.py``, alongside our own
encoder + ``Paper2Classifier`` + ``HierarchicalFallback`` pipeline:

- ``naive_bayes_baseline.py``: a scikit-learn ``MultinomialNB`` trained on
  raw k-mer frequency vectors, standing in for QIIME2's actual classifier
  algorithm (installing the full multi-GB QIIME2 conda stack was judged
  infeasible for this project's portability goals).
- ``nearest_neighbor_baseline.py``: a plain single-nearest-neighbor lookup
  over the *same* k-mer+PCA embeddings our system uses, with no taxonomy-
  hierarchy reasoning and no geography -- isolating exactly what Paper 2's
  phylogeny+co-occurrence signal adds.
- ``blast_baseline.py``: real NCBI BLAST+ (``makeblastdb``/``blastn`` via
  subprocess), the canonical exact-match tool this project's fallback
  module is designed to improve on (BLAST returns no answer at all when no
  sufficiently similar reference exists, rather than a ranked, hierarchical
  guess).

None of these baselines abstain gracefully the way our system's fallback
does (except BLAST, which instead fails outright below a %identity cutoff)
-- that contrast is the point of the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaselinePrediction:
    """One query's prediction from a baseline system.

    All three baselines share this shape so ``src/eval/benchmark.py`` and
    ``src/eval/head_to_head.py`` can score them uniformly. ``None`` at a
    given rank means "no answer" (only ever produced by ``BlastBaseline``
    at its operating-point cutoff); the two ML baselines always fill every
    rank.

    ``confidence`` is a per-rank score in ``[0, 1]`` used for
    false-confident-wrong-call curves. ``nearest`` is the top/nearest label
    at each rank even when the operating point abstained (BLAST below
    ``min_pident``) -- that is what the threshold sweep scores.
    """

    species: str | None
    genus: str | None
    family: str | None
    order: str | None
    confidence: dict[str, float] = field(default_factory=dict)
    nearest: dict[str, str] = field(default_factory=dict)
