"""
augment.py

Waveform-level augmentation for training data. RAVDESS is small (~1400
files), so augmentation is applied from the start of training, not as a
later optional improvement.

Applies (each with its own internal probability, wrapped by an outer
config.AUGMENT_PROBABILITY gate in dataset.py):
    - Gaussian noise
    - Pitch shift
    - Time stretch
    - Gain (volume) change
"""

import numpy as np
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Gain

SAMPLE_RATE = 16000

train_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.4),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.4),
])


def apply_augmentation(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply the augmentation pipeline to a 1D float32 waveform array."""
    return train_augment(samples=waveform.astype(np.float32), sample_rate=sample_rate)
