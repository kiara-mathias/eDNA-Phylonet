"""Hierarchical fallback + novelty-flagging package (Step 5 of the build plan).

Not yet implemented. This is the project's novel component: adapted from
Paper 3's (Algorithms/MDPI, Feb 2025) genus-level graceful-fallback idea,
but implemented as an add-on layer sitting after Paper 2's classifier
(``src/model``) rather than as a standalone hybrid ensemble.

Planned behavior: per-sample distance-to-centroid scoring at each
taxonomic rank (species -> genus -> family -> order), with calibrated
per-rank thresholds, deciding whether to emit a confident species label or
a "novel taxon, closest relative: X" flag -- always returning a full
hierarchical confidence vector rather than a single label.
"""
