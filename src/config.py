"""
config.py
Central configuration for the SER project. Edit values here rather than
scattering magic numbers across scripts.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# RAVDESS has 8 original emotion classes. The assignment brief merges
# neutral+calm into a single "neutral" class (7-class taxonomy) to resolve
# label ambiguity between the two (they sound very similar acoustically).
#
# Set this to False if you want 8 separate classes instead (e.g. to directly
# compare against a reference model trained on all 8 original classes).
MERGE_CALM_INTO_NEUTRAL = True

if MERGE_CALM_INTO_NEUTRAL:
    LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
else:
    LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ---------------------------------------------------------------------------
# Model / audio
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/wav2vec2-base"   # plain pretrained backbone, NOT an ASR head
SAMPLE_RATE = 16000
FREEZE_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------
# Actor split (speaker-independent) — 24 actors total
# ---------------------------------------------------------------------------
TRAIN_ACTORS = list(range(1, 20))   # 1-19  (19 actors)
VAL_ACTORS = [20, 21]                # 2 actors
TEST_ACTORS = [22, 23, 24]           # 3 actors

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    output_dir: str = "outputs/ser-model"
    num_train_epochs: int = 25
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    label_smoothing_factor: float = 0.08
    fp16: bool = False              # fp16 has caused NaN loss with wav2vec2 fine-tuning; keep fp32
    logging_steps: int = 10
    save_total_limit: int = 2
    early_stopping_patience: int = 10
    seed: int = 22


TRAIN_CONFIG = TrainConfig()

# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------
# Base weights are computed automatically at train time (inverse frequency
# from the training split). Use this dict to apply an EXTRA multiplier on
# top of that for classes with known weak recall.
#
# From the reference confusion matrix (8-class run):
#   - sad:   recall 0.292 (7/24) -- confused mostly with fearful (8) and happy (5)
#   - angry: recall 0.458 (11/24) -- confused mostly with happy (7)
# Both get boosted. If you switch MERGE_CALM_INTO_NEUTRAL to False (8-class),
# these keys still apply directly; in 7-class mode they still work since
# both "sad" and "angry" exist in both taxonomies.
EXTRA_CLASS_WEIGHT_BOOST = {
    "sad": 1.35,
    "angry": 1.2,
}

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
AUGMENT_PROBABILITY = 0.7   # probability of applying augmentation to a training sample
