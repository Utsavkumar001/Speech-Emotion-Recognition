"""
error_analysis.py

Reads predictions.csv (produced by evaluate.py) and surfaces:
    - top confused label pairs, ranked by count (e.g. sad -> fearful: 8)
    - the most-confident WRONG predictions (model was confidently wrong --
      these are the most interesting failure cases to inspect)
    - per-class recall ranked worst-to-best

Usage:
    python src/error_analysis.py --predictions reports/predictions.csv --out_dir reports
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config


def top_confused_pairs(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    wrong = df[df["true_label"] != df["pred_label"]]
    pair_counts = (
        wrong.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return pair_counts.head(top_n)


def worst_confident_mistakes(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    wrong = df[df["true_label"] != df["pred_label"]].copy()
    wrong = wrong.sort_values("confidence", ascending=False)
    return wrong[["path", "true_label", "pred_label", "confidence"]].head(top_n)


def per_class_recall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in config.LABELS:
        subset = df[df["true_label"] == label]
        if len(subset) == 0:
            continue
        correct = (subset["true_label"] == subset["pred_label"]).sum()
        rows.append({
            "label": label,
            "n_samples": len(subset),
            "correct": correct,
            "recall": correct / len(subset),
        })
    result = pd.DataFrame(rows).sort_values("recall")
    return result


def run(predictions_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(predictions_csv)

    pairs = top_confused_pairs(df)
    mistakes = worst_confident_mistakes(df)
    recall = per_class_recall(df)

    print("=== Top confused pairs (true -> predicted) ===")
    print(pairs.to_string(index=False))

    print("\n=== Per-class recall (worst first) ===")
    print(recall.to_string(index=False))

    print("\n=== Most confident WRONG predictions (inspect these audio files) ===")
    print(mistakes.to_string(index=False))

    pairs.to_csv(os.path.join(out_dir, "confused_pairs.csv"), index=False)
    mistakes.to_csv(os.path.join(out_dir, "worst_mistakes.csv"), index=False)
    recall.to_csv(os.path.join(out_dir, "per_class_recall.csv"), index=False)
    print(f"\nSaved confused_pairs.csv, worst_mistakes.csv, per_class_recall.csv to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()
    run(args.predictions, args.out_dir)
