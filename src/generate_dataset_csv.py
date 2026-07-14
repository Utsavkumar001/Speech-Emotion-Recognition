"""
generate_dataset_csv.py

Scans the RAVDESS 'Speech audio-only' dataset folder (Actor_01 ... Actor_24)
and builds a clean dataset.csv with parsed labels.

RAVDESS filename format (7 parts, dash-separated):
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    e.g. 03-01-06-01-02-01-12.wav

Usage:
    python src/generate_dataset_csv.py --data_dir data/RAVDESS --out data/dataset.csv
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config

RAW_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear" if config.MERGE_CALM_INTO_NEUTRAL else "fearful",
    "07": "disgust",
    "08": "surprise" if config.MERGE_CALM_INTO_NEUTRAL else "surprised",
}

INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {"01": "kids_door", "02": "dogs_door"}


def parse_filename(filename: str) -> dict:
    name = filename.replace(".wav", "")
    parts = name.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected filename format: {filename}")

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    actor_id = int(actor)
    gender = "male" if actor_id % 2 != 0 else "female"

    raw_emotion = RAW_EMOTION_MAP[emotion]
    if config.MERGE_CALM_INTO_NEUTRAL and raw_emotion == "calm":
        final_emotion = "neutral"
    else:
        final_emotion = raw_emotion

    return {
        "filename": filename,
        "modality": modality,
        "vocal_channel": vocal_channel,
        "raw_emotion": raw_emotion,
        "emotion": final_emotion,
        "intensity": INTENSITY_MAP[intensity],
        "statement": STATEMENT_MAP[statement],
        "repetition": int(repetition),
        "actor": actor_id,
        "gender": gender,
    }


def build_dataset_csv(data_dir: str, out_path: str) -> pd.DataFrame:
    rows, skipped = [], []

    for actor_folder in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if not fname.lower().endswith(".wav"):
                continue
            try:
                meta = parse_filename(fname)
            except ValueError as e:
                skipped.append(str(e))
                continue
            if meta["modality"] != "03" or meta["vocal_channel"] != "01":
                skipped.append(f"Skipped non speech/audio-only file: {fname}")
                continue
            meta["path"] = os.path.join(actor_path, fname)
            rows.append(meta)

    if not rows:
        raise RuntimeError(f"No valid .wav files found under {data_dir}.")

    df = pd.DataFrame(rows)
    cols = ["path", "filename", "emotion", "raw_emotion", "intensity",
            "statement", "repetition", "actor", "gender", "modality", "vocal_channel"]
    df = df[cols]

    unknown = set(df["emotion"].unique()) - set(config.LABELS)
    if unknown:
        raise RuntimeError(
            f"Parsed emotions {unknown} not in config.LABELS {config.LABELS}. "
            f"Check config.MERGE_CALM_INTO_NEUTRAL setting."
        )

    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}  (skipped {len(skipped)})")
    print(f"Taxonomy: {config.LABELS}  ({config.NUM_LABELS} classes)")
    print("\nClass distribution:")
    print(df["emotion"].value_counts())
    print(f"\nActors: {df['actor'].nunique()}  |  Gender split:")
    print(df["gender"].value_counts())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/dataset.csv")
    args = parser.parse_args()
    build_dataset_csv(args.data_dir, args.out)
