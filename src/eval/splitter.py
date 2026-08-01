"""Clade-exclusion splitter (Step 3 of the build plan).

Simulates genuinely unseen species by holding out entire **genera** (not
families -- see the module-level note below) at configurable fractions,
and saves the resulting splits as JSON files rather than regenerating them
with an unseeded RNG each run.

Usage::

    python -m src.eval.splitter --config configs/eval.yaml

Why genus, not family: the current dataset spans only 5 families but 151
genera (see ``data/manifest.json``). Holding out 30/50/70% of 5 families
rounds to 1-4 families -- too coarse for a meaningful accuracy curve.
Holding out 30/50/70% of 151 genera gives real granularity, while
downstream evaluation (``src/eval/validate_baseline.py``) still reports
family-level accuracy as the benchmark metric: a held-out genus's family is
still "seen" via sibling genera, which is exactly the generalization
scenario Paper 2/3 target.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `python src/eval/splitter.py`

import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)


def make_split(
    df: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    val_fraction: float,
    unit: str = "genus",
) -> dict[str, Any]:
    """Build one clade-exclusion split.

    Returns a dict with the held-out taxa and the ``process_id`` lists for
    train/val/test, suitable for ``json.dump``.
    """
    if unit != "genus":
        raise NotImplementedError(f"Only genus-level holdout is implemented, got unit={unit!r}")

    rng = random.Random(seed)

    all_genera = sorted(df["genus"].unique())
    n_held_out = round(len(all_genera) * holdout_fraction)
    held_out_genera = set(rng.sample(all_genera, n_held_out))

    test_mask = df["genus"].isin(held_out_genera)
    test_df = df[test_mask]
    seen_df = df[~test_mask]

    train_ids: list[str] = []
    val_ids: list[str] = []

    # Stratify the val split by species so val also exercises ordinary
    # (non-zero-shot) generalization on species the classifier has *some*
    # training examples for. Species with fewer than 2 rows can't be split
    # without leaving val or train empty for that species, so they stay in
    # train entirely.
    for _, species_rows in seen_df.groupby("species", sort=False):
        ids = list(species_rows["process_id"])
        rng.shuffle(ids)
        n_val = int(len(ids) * val_fraction) if len(ids) >= 2 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return {
        "unit": unit,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        "val_fraction": val_fraction,
        "n_genera_total": len(all_genera),
        "n_genera_held_out": len(held_out_genera),
        "held_out_genera": sorted(held_out_genera),
        "train_process_ids": sorted(train_ids),
        "val_process_ids": sorted(val_ids),
        "test_process_ids": sorted(test_df["process_id"].tolist()),
    }


def make_all_splits(config: dict[str, Any]) -> list[Path]:
    """Build and write every configured holdout split. Returns written paths."""
    data_cfg = config["data"]
    splitter_cfg = config["splitter"]

    sequences_path = resolve_path(data_cfg["sequences_path"])
    df = pd.read_parquet(sequences_path)

    splits_dir = resolve_path(splitter_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fraction in splitter_cfg["holdout_fractions"]:
        split = make_split(
            df,
            holdout_fraction=fraction,
            seed=splitter_cfg["seed"],
            val_fraction=splitter_cfg["val_fraction"],
            unit=splitter_cfg.get("unit", "genus"),
        )
        pct = round(fraction * 100)
        out_path = splits_dir / f"holdout_{pct}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(split, fh, indent=2)
        logger.info(
            "Wrote %s: %d/%d genera held out -> train=%d val=%d test=%d",
            out_path,
            split["n_genera_held_out"],
            split["n_genera_total"],
            len(split["train_process_ids"]),
            len(split["val_process_ids"]),
            len(split["test_process_ids"]),
        )
        written.append(out_path)

    return written


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Path to the eval YAML config (default: configs/eval.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    make_all_splits(config)


if __name__ == "__main__":
    main()
