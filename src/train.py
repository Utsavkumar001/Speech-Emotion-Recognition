"""
train.py

Fine-tunes Wav2Vec2ForSequenceClassification on RAVDESS with:
    - CNN feature extractor frozen (only transformer + classification head trained)
    - class-weighted loss (inverse frequency + manual boost for weak classes)
    - dynamic padding via DataCollatorSER
    - early stopping on macro F1

Usage:
    python src/train.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import RAVDESSDataset, DataCollatorSER


def compute_class_weights(train_csv: str) -> torch.Tensor:
    """Inverse-frequency weights from the training split, with manual boosts
    applied on top for classes with known weak recall (see config.py)."""
    import pandas as pd
    df = pd.read_csv(train_csv)
    counts = df["emotion"].value_counts()

    weights = torch.zeros(config.NUM_LABELS)
    total = len(df)
    for label, idx in config.LABEL2ID.items():
        count = counts.get(label, 1)
        base_weight = total / (config.NUM_LABELS * count)
        boost = config.EXTRA_CLASS_WEIGHT_BOOST.get(label, 1.0)
        weights[idx] = base_weight * boost

    print("Class weights:")
    for label, idx in config.LABEL2ID.items():
        print(f"  {label}: {weights[idx]:.3f}")
    return weights


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted + label-smoothed
    cross-entropy loss instead of the default unweighted loss."""

    def __init__(self, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss = loss_fct(logits.view(-1, config.NUM_LABELS), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {"accuracy": acc, "macro_f1": macro_f1}
    for idx, label in config.ID2LABEL.items():
        metrics[f"f1_{label}"] = per_class_f1[idx] if idx < len(per_class_f1) else 0.0
    return metrics


def main(train_csv: str, val_csv: str, output_dir: str = None):
    cfg = config.TRAIN_CONFIG
    output_dir = output_dir or cfg.output_dir
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Loading feature extractor + model: {config.MODEL_NAME}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        label2id=config.LABEL2ID,
        id2label=config.ID2LABEL,
    )

    if config.FREEZE_FEATURE_EXTRACTOR:
        model.freeze_feature_encoder()
        print("CNN feature encoder frozen. Training transformer + classification head only.")

    train_dataset = RAVDESSDataset(train_csv, augment=True)
    val_dataset = RAVDESSDataset(val_csv, augment=False)
    collator = DataCollatorSER(feature_extractor=feature_extractor)

    class_weights = compute_class_weights(train_csv)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing_factor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()

    print(f"\nSaving best model + feature extractor to {output_dir}/final")
    trainer.save_model(os.path.join(output_dir, "final"))
    feature_extractor.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(args.train_csv, args.val_csv, args.output_dir)
