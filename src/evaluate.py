"""
evaluate.py

Runs a trained model on the test split and produces:
    - classification_report.txt (per-class precision/recall/F1)
    - confusion_matrix.png
    - metrics.json (accuracy, macro F1, per-class F1)
    - predictions.csv (path, true label, predicted label, confidence per class)
      -- this feeds directly into error_analysis.py

Usage:
    python src/evaluate.py --model_dir outputs/ser-model/final --test_csv data/splits/test.csv --out_dir reports
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import load_waveform


def run_inference(model_dir: str, test_csv: str):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    df = pd.read_csv(test_csv)
    all_preds, all_labels, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for _, row in df.iterrows():
            waveform = load_waveform(row["path"])
            inputs = feature_extractor(
                waveform, sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
            ).to(device)

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))

            all_preds.append(pred_id)
            all_labels.append(config.LABEL2ID[row["emotion"]])
            all_probs.append(probs)
            all_paths.append(row["path"])

    return {
        "paths": all_paths,
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "probs": np.array(all_probs),
    }


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_LABELS)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.LABELS, yticklabels=config.LABELS,
    )
    plt.title("Confusion Matrix — Speech Emotion Recognition (wav2vec2 + RAVDESS)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm


def evaluate(model_dir: str, test_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    results = run_inference(model_dir, test_csv)
    y_true, y_pred, probs, paths = (
        results["y_true"], results["y_pred"], results["probs"], results["paths"]
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=config.LABELS, zero_division=0
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")
    print(report)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}")

    cm = save_confusion_matrix(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"))

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {
            config.ID2LABEL[i]: float(v)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None, zero_division=0))
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "path": paths,
        "true_label": [config.ID2LABEL[i] for i in y_true],
        "pred_label": [config.ID2LABEL[i] for i in y_pred],
        "confidence": probs.max(axis=1),
    })
    for i, label in config.ID2LABEL.items():
        pred_df[f"prob_{label}"] = probs[:, i]
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print(f"\nSaved report, confusion matrix, metrics.json, predictions.csv to {out_dir}/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()
    evaluate(args.model_dir, args.test_csv, args.out_dir)
