"""
actor_split.py

Splits dataset.csv into train/val/test by ACTOR (config.TRAIN_ACTORS /
VAL_ACTORS / TEST_ACTORS), never by random row, so no speaker leaks across
splits.

Usage:
    python src/actor_split.py --csv data/dataset.csv --out_dir data/splits
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config


def split_by_actor(df, train_actors=None, val_actors=None, test_actors=None):
    train_actors = train_actors or config.TRAIN_ACTORS
    val_actors = val_actors or config.VAL_ACTORS
    test_actors = test_actors or config.TEST_ACTORS

    overlap = (
        set(train_actors) & set(val_actors)
        | set(train_actors) & set(test_actors)
        | set(val_actors) & set(test_actors)
    )
    assert not overlap, f"Actor overlap between splits: {overlap}"

    dataset_actors = set(df["actor"].unique())
    all_assigned = set(train_actors) | set(val_actors) | set(test_actors)
    missing = dataset_actors - all_assigned
    if missing:
        print(f"Warning: actors {missing} not assigned to any split, dropped.")

    train_df = df[df["actor"].isin(train_actors)].reset_index(drop=True)
    val_df = df[df["actor"].isin(val_actors)].reset_index(drop=True)
    test_df = df[df["actor"].isin(test_actors)].reset_index(drop=True)
    return train_df, val_df, test_df


def print_summary(name, split_df):
    print(f"\n{name}: {len(split_df)} files, {split_df['actor'].nunique()} actors")
    print(split_df["emotion"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/splits")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = split_by_actor(df)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    print_summary("Train", train_df)
    print_summary("Val", val_df)
    print_summary("Test", test_df)
    print(f"\nSplits saved to {args.out_dir}/")
