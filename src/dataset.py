"""
dataset.py

PyTorch Dataset for RAVDESS + a data collator that pads each batch
dynamically (NOT to a fixed length baked into preprocessing -- padding
happens here, at batch-construction time, via the Wav2Vec2FeatureExtractor).
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
import config
from augment import apply_augmentation

TARGET_SR = config.SAMPLE_RATE


def load_waveform(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read an audio file, convert to mono, resample to target_sr, normalize."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(waveform).squeeze(0).numpy()

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


class RAVDESSDataset(Dataset):
    """
    Wraps a split CSV (path, emotion columns) and returns raw waveforms +
    integer label ids. Feature extraction (padding, attention masks) happens
    later in the collator, not here -- this keeps augmentation and I/O
    independent of batch composition.
    """

    def __init__(self, csv_path: str, augment: bool = False):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        unknown = set(self.df["emotion"].unique()) - set(config.LABEL2ID.keys())
        if unknown:
            raise ValueError(f"CSV contains labels not in config.LABELS: {unknown}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Union[np.ndarray, int]]:
        row = self.df.iloc[idx]
        waveform = load_waveform(row["path"])

        if self.augment and np.random.rand() < config.AUGMENT_PROBABILITY:
            waveform = apply_augmentation(waveform)

        label_id = config.LABEL2ID[row["emotion"]]
        return {"input_values": waveform, "label": label_id}


@dataclass
class DataCollatorSER:
    """
    Pads a batch of variable-length waveforms using the model's feature
    extractor, and stacks labels into a tensor. This is where dynamic
    padding happens -- individual .pt files are never pre-padded to a
    fixed length.
    """
    feature_extractor: "transformers.Wav2Vec2FeatureExtractor"
    padding: Union[bool, str] = True

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_values = [item["input_values"] for item in batch]
        labels = [item["label"] for item in batch]

        features = self.feature_extractor(
            input_values,
            sampling_rate=TARGET_SR,
            padding=self.padding,
            return_tensors="pt",
        )
        features["labels"] = torch.tensor(labels, dtype=torch.long)
        return features
