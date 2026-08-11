# eDNA Biodiversity Classifier

Zero-shot eDNA species annotation with an honest fallback: instead of
failing outright when a read doesn't match any known reference sequence
(the failure mode of exact-match tools like BLAST/QIIME2), this system
always returns a usable, ranked answer -- either a confident species call,
or a "novel taxon" flag with the top-k nearest known species/genera and a
hierarchical confidence score at each taxonomic rank (species -> genus ->
family -> order).

**Current status: all 8 steps of the build plan are implemented and
validated end-to-end** -- ingestion, cleaning, clade-exclusion splitting,
the base classifier, the hierarchical fallback/novelty-flagging layer, the
BLAST+Naive-Bayes+1-NN benchmarking harness, and the Streamlit dashboard --
see [Roadmap](#roadmap).

## Approach

- **Base algorithm** (Paper 2: PLOS Computational Biology, Dec 2025):
  zero-shot annotation using phylogeny structure + species co-occurrence to
  classify eDNA reads directly from raw sequences, including species never
  seen in training.
- **Novel component** (adapted from Paper 3: Algorithms/MDPI, Feb 2025):
  a hierarchical fallback + novelty-flagging layer sitting after
  classification, deciding per-sample whether to output a confident label
  or a flagged novel taxon with the top-k nearest known relatives and
  per-rank confidence.
- **Robustness insight** (Paper 1: IEEE Access 2024,
  [DOI 10.1109/ACCESS.2024.3450016](https://doi.org/10.1109/ACCESS.2024.3450016)):
  informs handling of messy/incomplete input data throughout.

Validated via clade-exclusion benchmarking (30%/50%/70% **genus** holdout --
see [`configs/eval.yaml`](configs/eval.yaml) for why genus rather than
family, given this dataset only spans 5 families -- with family-level
accuracy reported as the benchmark metric) on marine fish COI-5P barcodes
from BOLD Systems (5 families: Gadidae, Scombridae, Pleuronectidae,
Serranidae, Carangidae). Benchmarked against a real BLAST+ baseline, a
scikit-learn Naive Bayes stand-in for QIIME2's classifier, and a
no-phylogeny/no-geo 1-NN ablation of our own encoder -- see
[`src/eval/benchmark.py`](src/eval/benchmark.py) and the
[Benchmarking](#benchmarking) section below.

"Phylogeny structure + species co-occurrence" is currently operationalized
as taxonomic hierarchy (order/family/genus/species, from BOLD metadata) +
geographic proximity (lat/lon per record, as a proxy for co-occurrence) --
see [`src/model/classifier.py`](src/model/classifier.py).

The hierarchical fallback (`src/fallback/novelty.py`) sits after that base
classifier: it cascades species -> genus -> family -> order, committing to
the deepest rank whose nearest-centroid distance is within a threshold
calibrated (per rank) from the validation split, and otherwise flags the
query as novel -- while always reporting the top-k nearest known species
and genera (by distance) so the "closest relative" claim is inspectable,
with a confidence score at every rank.

## Repo layout

```
edna-classifier/
  data/                # data/raw, data/processed, data/eval_results gitignored;
                       # manifest.json + splits/ committed for reproducibility
  src/
    ingest/            # BOLD pull scripts (implemented)
    preprocess/        # cleaning, QC (implemented)
    features/          # KmerPCAEncoder (default) + ConvTripletEncoder (1D CNN);
                       # optional frozen DNABERT via HuggingFaceDNAEncoder
    model/             # Paper2Classifier: taxonomy+geo nearest-centroid (implemented)
    fallback/          # HierarchicalFallback: cascading novelty flagging (implemented)
    eval/              # clade-exclusion splitter, baseline/fallback validation,
                       # confidence calibration (ECE/reliability), BLAST/Naive-
                       # Bayes/1-NN benchmark, and head-to-head accuracy/FCW
                       # table+plot (all implemented)
    baselines/         # NaiveBayesBaseline, NearestNeighborBaseline, BlastBaseline
  app/                 # Streamlit dashboard: pipeline.py + dashboard.py (implemented)
  configs/             # YAML config per experiment
  tests/               # offline unit tests (no network required)
  Dockerfile           # CPU-only image for ingest/preprocess (+ ncbi-blast+)
  Dockerfile.inference # lightweight dashboard-only image
  requirements.txt     # pinned exact versions
```

## Setup

Requires Python 3.12+ (the pinned `numpy`/`scipy`/`scikit-learn` versions in
`requirements.txt` only ship prebuilt wheels for 3.12+; developed against
3.12.4).

```bash
cd edna-classifier
pip install -r requirements.txt
```

## Running the pipeline (clean clone -> processed dataset)

```bash
# 1. Fetch raw BOLD records for marine fish (Actinopterygii) COI-5P barcodes.
#    Writes data/raw/bold_<timestamp>/*.json and data/manifest.json.
python -m src.ingest.fetch_bold --config configs/ingestion.yaml

# 2. Clean + dedupe into the single source-of-truth table.
#    Writes data/processed/sequences.parquet.
python -m src.preprocess.clean --config configs/ingestion.yaml
```

Both commands take `--config <path>` (default `configs/ingestion.yaml`) --
no paths are hardcoded; everything is resolved against the repo root (see
`src/common.py`, `PROJECT_ROOT` env var support included for container use).

```bash
# 3. Build the genus-level clade-exclusion splits (30/50/70% holdout).
#    Writes data/splits/holdout_{30,50,70}.json.
python -m src.eval.splitter --config configs/eval.yaml

# 4. Fit the k-mer+PCA encoder + Paper2Classifier on each split's train set
#    and report species/genus/family/order accuracy on its held-out test set.
#    Writes data/eval_results/baseline.json.
python -m src.eval.validate_baseline --config configs/eval.yaml
```

`validate_baseline` prints per-holdout accuracy. On the current dataset,
species/genus accuracy on held-out genera is (expectedly) 0% -- the exact
species/genus can never be right when its whole genus was excluded from
training, which is precisely the "zero-shot floor problem" Step 5's
fallback module addresses -- while family/order accuracy degrades sensibly
as more genera are held out (~88% at 30% holdout, ~70% at 50%, ~68% at
70%), confirming the base classifier's taxonomy+geography signal is doing
real work.

```bash
# 5. Fit the encoder + HierarchicalFallback on each split's train set,
#    calibrate confidence thresholds on val, and report novelty-detection
#    recall / abstention / resolved-rank accuracy on the held-out test set.
#    Writes data/eval_results/fallback.json.
python -m src.eval.validate_fallback --config configs/eval.yaml
```

`validate_fallback` prints, per holdout level: **novelty-detection recall**
on held-out-genus queries (~98-99% -- the fallback correctly recognizes
these as unseen almost every time), **abstention rate** (~1-2%, fully
unresolved even at order level), and **family-resolved accuracy** among
queries the model *did* commit to at family rank (~63-70%, in the same
ballpark as `validate_baseline`'s unconditional family accuracy -- a good
cross-check between the two scripts). On the val split (in-distribution,
seen species), ~90% still get a confident species call, consistent with
the 90th-percentile calibration, with ~88-90% of those species calls
actually correct.

```bash
# 5b. Prove reported confidence is calibrated on the same held-out genera.
#     Dumps confidence+correctness per query/rank, writes a reliability
#     diagram, reports ECE per rank, and (if ECE > 0.1) fits isotonic /
#     Platt recalibrators leave-one-holdout-out then re-scores. Writes
#     data/eval_results/calibration/.
python -m src.eval.validate_calibration --config configs/eval.yaml
```

```bash
# 5c. Learned encoder: same HierarchicalFallback, k-mer+PCA vs a small 1D
#     CNN trained with triplet loss (and optionally a frozen DNABERT-style
#     Hugging Face model -- extra torch/transformers deps, no fine-tuning).
#     Re-runs Step 1 ECE and Step 2 accuracy / false-confident-wrong-call
#     rate on identical holdout splits. Writes
#     data/eval_results/encoder_comparison/.
python -m src.eval.compare_encoders --config configs/eval.yaml
```

Set `encoder.name: conv_triplet` in [`configs/eval.yaml`](configs/eval.yaml)
(or `configs/app.yaml` for the dashboard) to run any single script --
`validate_calibration`, `validate_fallback`, `benchmark` -- on the CNN
instead of k-mer+PCA. The fallback hyperparameters do not change.

## Benchmarking

```bash
# 6. Compare our system (encoder + Paper2Classifier + HierarchicalFallback)
#    against a real BLAST+ baseline, a scikit-learn Naive Bayes stand-in
#    for QIIME2's classifier, and a no-phylogeny/no-geo 1-NN ablation of
#    our own encoder -- across all three holdout levels. Reports per-rank
#    coverage (fraction of queries given any answer) and accuracy-among-
#    answered (of those, fraction correct). Writes
#    data/eval_results/benchmark.json.
python -m src.eval.benchmark --config configs/eval.yaml
```

On held-out genera, our system mostly abstains at species/genus level
(coverage ~1-2%) rather than guessing wrong, while Naive Bayes and 1-NN
always answer (100% coverage) but get ~0% species/genus accuracy there --
the exact failure mode this project's fallback exists to avoid. At
family/order, our system's accuracy-among-answered (~0.71-0.84) beats
Naive Bayes (~0.10-0.16) and is comparable to the 1-NN ablation, showing
where the phylogeny/geo reasoning helps vs. where the embedding itself is
doing the work. Each coverage/accuracy figure is also reported with a
bootstrap percentile CI (see `benchmark.n_bootstrap` / `benchmark.ci_level`
in [`configs/eval.yaml`](configs/eval.yaml)). The BLAST row requires
`blastn`/`makeblastdb` on `PATH` (installed in Docker/CI; not required for
local development, skipped with a logged warning if absent -- see
`src/baselines/blast_baseline.py`).

```bash
# 6b. Standardized head-to-head on the same held-out splits: BLAST, Naive
#     Bayes, 1-NN, and our system. Per method per rank: accuracy (abstention
#     counts as wrong) and false-confident-wrong-call rate, also restricted
#     to the novel/held-out-genera subset. Writes one table (method × rank
#     × metric) and one plot (FCW rate vs. confidence threshold, one line
#     per method). BLAST uses bitscore (e-value fallback) as its score.
#     Writes data/eval_results/head_to_head/.
python -m src.eval.head_to_head --config configs/eval.yaml
```

Reported confidence in the fallback (and dashboard) is distance-based
confidence multiplied by sample-count reliability
`n / (n + sample_count_prior)`, so thinly sampled centroids cannot look as
trustworthy as well-backed ones at the same distance. Cascade commit
decisions still use raw distance vs. the calibrated threshold.

## Dashboard

```bash
# 7. Launch the interactive specimen record: paste a DNA sequence and get
#    a hierarchical, novelty-aware prediction. Calibration, BLAST-lie
#    gallery, and false-confident-wrong curves render below when eval
#    artifacts exist.
streamlit run app/dashboard.py
```

The page is a single vertical scroll (not a sidebar of metric cards). Color
is taxonomic confidence as ocean depth: species-level calls sit in shallow
seafoam, order-level fallback in deep navy -- the same cascade
`HierarchicalFallback` uses, made visible. The taxonomic depth tree is the
only element that animates (rank-by-rank reveal). Reliability, nearest
relatives (when the read is flagged novel), a gallery of held-out queries
where a species-level identity call would have been confidently wrong, and
an interactive false-confident-wrong-vs-threshold chart follow underneath
when `data/eval_results/calibration/` and `benchmark.json` are present.

Requires `data/processed/sequences.parquet` to already exist (steps 1-2
above) -- the dashboard fits the encoder + classifier + fallback once at
startup (cached via `st.cache_resource`, no GPU/training needed) and shows
a friendly "run ingestion first" message instead of crashing if the data
isn't there yet. See [`configs/app.yaml`](configs/app.yaml) for why this
uses a random validation split rather than the genus-exclusion splits in
`configs/eval.yaml` -- the deployed model should use every known genus.

Evidence panels (matched to the calibration / head-to-head / nearest-
relative work, not a substitute for running those scripts):

- **Calibration** reads `data/eval_results/calibration/calibration.json`
  and, when `heldout_predictions.parquet` is present, re-bins the
  reliability diagram live.
- **BLAST would have lied** is five hardcoded 50% genus-holdout contrasts
  (wrong-confident BLAST vs. honest fallback + top-k relatives).
- **False-confident wrong** reads `data/eval_results/head_to_head/head_to_head.json`,
  toggles methods, and slides the confidence threshold along the saved curve.

## Running tests

```bash
pytest tests/ -v
```

All tests are offline (BOLD API calls are mocked) and require no network
access or real dataset.

## Docker

```bash
docker build -t edna-classifier .
docker run -it --rm -v "$(pwd)/data:/app/data" edna-classifier bash

# Lightweight dashboard-only image:
docker build -t edna-classifier-inference -f Dockerfile.inference .
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" edna-classifier-inference
# then open http://localhost:8501
```

Neither Dockerfile has been built/validated on the authoring machine (no
local Docker install) -- both are validated on every push via
[`.github/workflows/docker-build.yml`](.github/workflows/docker-build.yml),
which builds both images on GitHub's own runners and additionally
smoke-tests that the inference image's container starts and reports
healthy (`/_stcore/health`) -- CI has no reference data available, so this
only proves the container boots cleanly and the app's own "no data found"
message renders, not a real inference request. If you have Docker
locally, running the commands above (with real data mounted) is a good
first sanity check on a fresh clone.

## Roadmap

| Step | Description | Status |
|---|---|---|
| 0 | Repo, environment, Dockerfile, pinned deps | Done |
| 1 | Data ingestion (BOLD fetch script + manifest) | Done |
| 2 | Cleaning + taxonomy table (`sequences.parquet`) | Done |
| 3 | Clade-exclusion splitter (30/50/70% genus holdout, saved as JSON) | Done |
| 4 | Base embedding + classifier (k-mer+PCA; taxonomy+geo nearest-centroid) | Done |
| 5 | Novel hierarchical fallback module (distance-to-centroid, novelty flagging) | Done |
| 6 | Benchmark vs. BLAST+Naive-Bayes and a no-phylogeny nearest-neighbor baseline | Done |
| 7 | Dashboard + deployment (Streamlit, inference-only Docker image) | Done |

## Portability checklist

- [x] No absolute paths in source (`PROJECT_ROOT`-relative throughout)
- [x] Dependencies pinned exactly (not `>=`)
- [x] Data fetch script runs standalone and reproduces the manifest
- [ ] `docker build .` succeeds from a clean clone -- both Dockerfiles
      written but not locally validated; see
      `.github/workflows/docker-build.yml` for CI validation on push
- [x] Splits saved as files, not regenerated with unseeded RNG -- see
      `data/splits/holdout_*.json`, written by `src/eval/splitter.py`
- [x] README has exact commands: clean clone -> running dashboard (see
      [Dashboard](#dashboard) above)

## Data source & licensing

Sequence and taxonomy data comes from the
[BOLD Systems Portal API](https://boldsystems.org/data/api/) (free, public).
Taxonomic labels (order -> family -> genus -> species) arrive as metadata
already attached to each record -- this project does not build phylogenetic
trees or sequence DNA itself.
