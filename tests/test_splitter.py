"""Offline unit tests for src.eval.splitter.

Uses a small synthetic DataFrame (no real dataset or network needed).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.splitter import make_split


def _synthetic_df(n_genera: int = 10, species_per_genus: int = 3, rows_per_species: int = 4) -> pd.DataFrame:
    rows = []
    pid = 0
    for g in range(n_genera):
        genus = f"Genus{g}"
        for s in range(species_per_genus):
            species = f"{genus} species{s}"
            for _r in range(rows_per_species):
                rows.append(
                    {
                        "process_id": f"P{pid}",
                        "species": species,
                        "genus": genus,
                        "family": f"Family{g % 3}",
                        "order": "OrderA",
                    }
                )
                pid += 1
    return pd.DataFrame(rows)


def test_make_split_holds_out_correct_fraction_of_genera():
    df = _synthetic_df(n_genera=10)

    split = make_split(df, holdout_fraction=0.3, seed=42, val_fraction=0.15)

    assert split["n_genera_total"] == 10
    assert split["n_genera_held_out"] == 3
    assert len(split["held_out_genera"]) == 3


def test_make_split_is_deterministic_given_seed():
    df = _synthetic_df(n_genera=10)

    split_a = make_split(df, holdout_fraction=0.3, seed=42, val_fraction=0.15)
    split_b = make_split(df, holdout_fraction=0.3, seed=42, val_fraction=0.15)

    assert split_a["held_out_genera"] == split_b["held_out_genera"]
    assert split_a["train_process_ids"] == split_b["train_process_ids"]
    assert split_a["test_process_ids"] == split_b["test_process_ids"]


def test_make_split_different_seed_gives_different_holdout():
    df = _synthetic_df(n_genera=10)

    split_a = make_split(df, holdout_fraction=0.3, seed=1, val_fraction=0.15)
    split_b = make_split(df, holdout_fraction=0.3, seed=2, val_fraction=0.15)

    assert split_a["held_out_genera"] != split_b["held_out_genera"]


def test_held_out_genera_fully_absent_from_train_and_val():
    df = _synthetic_df(n_genera=10)

    split = make_split(df, holdout_fraction=0.3, seed=42, val_fraction=0.15)

    by_id = df.set_index("process_id")
    held_out = set(split["held_out_genera"])

    train_genera = set(by_id.loc[split["train_process_ids"], "genus"])
    val_genera = set(by_id.loc[split["val_process_ids"], "genus"]) if split["val_process_ids"] else set()

    assert train_genera.isdisjoint(held_out)
    assert val_genera.isdisjoint(held_out)


def test_test_set_exactly_matches_held_out_genera_rows():
    df = _synthetic_df(n_genera=10)

    split = make_split(df, holdout_fraction=0.3, seed=42, val_fraction=0.15)

    held_out = set(split["held_out_genera"])
    expected_test_ids = set(df.loc[df["genus"].isin(held_out), "process_id"])

    assert set(split["test_process_ids"]) == expected_test_ids


def test_train_val_test_partition_all_rows_exactly_once():
    df = _synthetic_df(n_genera=10)

    split = make_split(df, holdout_fraction=0.5, seed=7, val_fraction=0.2)

    all_ids = set(split["train_process_ids"]) | set(split["val_process_ids"]) | set(split["test_process_ids"])
    assert all_ids == set(df["process_id"])

    # No overlap between the three sets.
    train_set = set(split["train_process_ids"])
    val_set = set(split["val_process_ids"])
    test_set = set(split["test_process_ids"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_species_with_single_row_stays_entirely_in_train():
    df = _synthetic_df(n_genera=4, species_per_genus=2, rows_per_species=1)

    split = make_split(df, holdout_fraction=0.25, seed=42, val_fraction=0.5)

    # With only 1 row per species, none of them can be split into val
    # without leaving train or val empty for that species.
    assert len(split["val_process_ids"]) == 0


def test_unsupported_unit_raises():
    df = _synthetic_df(n_genera=4)

    with pytest.raises(NotImplementedError):
        make_split(df, holdout_fraction=0.3, seed=42, val_fraction=0.15, unit="family")
