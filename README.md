# eDNA Biodiversity Classifier

Zero-shot eDNA species annotation with an honest fallback: instead of
failing outright when a read doesn't match any known reference sequence
(the failure mode of exact-match tools like BLAST/QIIME2), this system
always returns a usable, ranked answer -- either a confident species call,
or a "novel taxon, closest relative: X" flag with a hierarchical confidence
score at each taxonomic rank (species -> genus -> family -> order).

**Current status: data ingestion + cleaning only.** The base classifier,
fallback module, benchmarking harness, and dashboard described below are
designed but not yet implemented -- see [Roadmap](#roadmap).

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

Validated via clade-exclusion benchmarking (30%/50%/70% family holdout)
against a BLAST + QIIME2 naive-Bayes baseline, on marine fish (Actinopterygii)
COI-5P barcodes from BOLD Systems.

## Repo layout

```
edna-classifier/
  data/                # gitignored raw/processed data; manifest.json tracks provenance
  src/
    ingest/            # BOLD pull scripts (implemented)
    preprocess/        # cleaning, QC (implemented)
    features/          # embedding/encoder (not yet implemented)
    model/             # base classifier, Paper 2 logic (not yet implemented)
    fallback/          # novel hierarchical fallback module (not yet implemented)
    eval/              # clade-exclusion splitter + benchmarking (not yet implemented)
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
| 3 | Clade-exclusion splitter (30/50/70% family holdout, saved as JSON) | Not started |
| 4 | Base embedding + classifier (k-mer+PCA first, then CNN; Paper 2 logic) | Not started |
| 5 | Novel hierarchical fallback module (distance-to-centroid, novelty flagging) | Not started |
| 6 | Benchmark vs. BLAST+QIIME2 and a no-phylogeny nearest-neighbor baseline | Not started |
| 7 | Dashboard + deployment (Streamlit, inference-only Docker image, docker-compose) | Not started |

## Portability checklist

- [x] No absolute paths in source (`PROJECT_ROOT`-relative throughout)
- [x] Dependencies pinned exactly (not `>=`)
- [x] Data fetch script runs standalone and reproduces the manifest
- [ ] `docker build .` succeeds from a clean clone -- Dockerfile written but
      not locally validated; see `.github/workflows/docker-build.yml` for
      CI validation on push
- [ ] Splits saved as files, not regenerated with unseeded RNG -- N/A until
      Step 3 (clade-exclusion splitter) is implemented
- [x] README has exact commands: clean clone -> running dashboard (dashboard
      commands will be added once Step 7 lands)

## Data source & licensing

Sequence and taxonomy data comes from the
[BOLD Systems Portal API](https://boldsystems.org/data/api/) (free, public).
Taxonomic labels (order -> family -> genus -> species) arrive as metadata
already attached to each record -- this project does not build phylogenetic
trees or sequence DNA itself.
