"""Hierarchical fallback + novelty-flagging package (Step 5 of the build plan).

Implemented: ``novelty.py``'s ``HierarchicalFallback`` -- this is the
project's novel component, adapted from Paper 3's (Algorithms/MDPI, Feb
2025) genus-level graceful-fallback idea, but implemented as an add-on
layer sitting after Paper 2's classifier (``src/model``) rather than as a
standalone hybrid ensemble.

Behavior: per-sample distance-to-centroid scoring at each taxonomic rank
(species -> genus -> family -> order), with calibrated per-rank thresholds
(set from the val split's in-distribution distance distribution), deciding
whether to emit a confident species label or a "novel taxon" flag with
the top-k nearest known species and genera -- always returning a full
hierarchical confidence vector rather than a single label.
"""
