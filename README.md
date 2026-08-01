# eDNA Biodiversity Classifier

Zero-shot eDNA species annotation with an honest fallback: instead of
failing outright when a read doesn't match any known reference sequence
(the failure mode of exact-match tools like BLAST/QIIME2), this system
always returns a usable, ranked answer -- either a confident species call,
or a "novel taxon, closest relative: X" flag with a hierarchical confidence
score at each taxonomic rank (species -> genus -> family -> order).

**Current status: ingestion, cleaning, clade-exclusion splitting, the base
classifier, and the hierarchical fallback/novelty-flagging layer are all
implemented and validated end-to-end.** The BLAST/QIIME2 benchmarking
harness and the dashboard described below are designed but not yet
implemented -- see [Roadmap](#roadmap).

## Approach

- **Base algorithm** (Paper 2: PLOS Computational Biology, Dec 2025):
  zero-shot annotation using phylogeny structure + species co-occurrence to
  classify eDNA reads directly from raw sequences, including species never
  seen in training.
- **Novel component** (adapted from Paper 3: Algorithms/MDPI, Feb 2025):
  a hierarchical fallback + novelty-flagging layer sitting after
  classification, deciding per-sample whether to output a confident label
  or a flagged "closest relative" with per-rank confidence.
- **Robustness insight** (Paper 1: IEEE Access 2024,
  [DOI 10.1109/ACCESS.2024.3450016](https://doi.org/10.1109/ACCESS.2024.3450016)):
  informs handling of messy/incomplete input data throughout.

Validated via clade-exclusion benchmarking (30%/50%/70% **genus** holdout --
see [`configs/eval.yaml`](configs/eval.yaml) for why genus rather than
family, given this dataset only spans 5 families -- with family-level
accuracy reported as the benchmark metric) on marine fish COI-5P barcodes
from BOLD Systems (5 families: Gadidae, Scombridae, Pleuronectidae,
Serranidae, Carangidae). A BLAST + QIIME2 naive-Bayes baseline comparison
is planned for Step 6 (not yet implemented).

"Phylogeny structure + species co-occurrence" is currently operationalized
as taxonomic hierarchy (order/family/genus/species, from BOLD metadata) +
geographic proximity (lat/lon per record, as a proxy for co-occurrence) --
see [`src/model/classifier.py`](src/model/classifier.py).

The hierarchical fallback (`src/fallback/novelty.py`) sits after that base
classifier: it cascades species -> genus -> family -> order, committing to
the deepest rank whose nearest-centroid distance is within a threshold
calibrated (per rank) from the validation split, and otherwise flags the
query as novel -- while always reporting the single nearest known species
as a "closest relative" hint, with a confidence score at every rank.

## Repo layout

```
edna-classifier/
  data/                # data/raw, data/processed, data/eval_results gitignored;
                       # manifest.json + splits/ committed for reproducibility
  src/
    ingest/            # BOLD pull scripts (implemented)
    preprocess/        # cleaning, QC (implemented)
    features/          # KmerPCAEncoder (implemented); learned encoder not yet
    model/             # Paper2Classifier: taxonomy+geo nearest-centroid (implemented)
    fallback/          # HierarchicalFallback: cascading novelty flagging (implemented)
    eval/              # clade-exclusion splitter + baseline/fallback validation
                       # (implemented); BLAST/QIIME2 benchmarking harness not yet
  app/                 # dashboard (not yet implemented)
  configs/             # YAML config per experiment
  tests/               # offline unit tests (no network required)
  Dockerfile           # CPU-only image for ingest/preprocess
  Dockerfile.inference # placeholder for the future dashboard-only image
  requirements.txt     # pinned exact versions
```

## Setup

Requires Python 3.11+ (developed against 3.12).

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
```

The Dockerfile has **not been built/validated on the authoring machine**
(no local Docker install) -- it is validated on every push via
[`.github/workflows/docker-build.yml`](.github/workflows/docker-build.yml),
which runs `docker build .` on GitHub's own runners. If you have Docker
locally, running the command above is a good first sanity check on a fresh
clone.

`Dockerfile.inference` is an intentional placeholder (no `FROM` instruction)
until the dashboard exists (see Roadmap) -- it will fail to build if
attempted today.

## Roadmap

| Step | Description | Status |
|---|---|---|
| 0 | Repo, environment, Dockerfile, pinned deps | Done |
| 1 | Data ingestion (BOLD fetch script + manifest) | Done |
| 2 | Cleaning + taxonomy table (`sequences.parquet`) | Done |
| 3 | Clade-exclusion splitter (30/50/70% genus holdout, saved as JSON) | Done |
| 4 | Base embedding + classifier (k-mer+PCA; taxonomy+geo nearest-centroid) | Done |
| 5 | Novel hierarchical fallback module (distance-to-centroid, novelty flagging) | Done |
| 6 | Benchmark vs. BLAST+QIIME2 and a no-phylogeny nearest-neighbor baseline | Not started |
| 7 | Dashboard + deployment (Streamlit, inference-only Docker image, docker-compose) | Not started |

## Portability checklist

- [x] No absolute paths in source (`PROJECT_ROOT`-relative throughout)
- [x] Dependencies pinned exactly (not `>=`)
- [x] Data fetch script runs standalone and reproduces the manifest
- [ ] `docker build .` succeeds from a clean clone -- Dockerfile written but
      not locally validated; see `.github/workflows/docker-build.yml` for
      CI validation on push
- [x] Splits saved as files, not regenerated with unseeded RNG -- see
      `data/splits/holdout_*.json`, written by `src/eval/splitter.py`
- [x] README has exact commands: clean clone -> running dashboard (dashboard
      commands will be added once Step 7 lands)

## Data source & licensing

Sequence and taxonomy data comes from the
[BOLD Systems Portal API](https://boldsystems.org/data/api/) (free, public).
Taxonomic labels (order -> family -> genus -> species) arrive as metadata
already attached to each record -- this project does not build phylogenetic
trees or sequence DNA itself.
