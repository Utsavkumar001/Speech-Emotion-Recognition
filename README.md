# Speech Emotion Recognition (SER) — wav2vec2 + RAVDESS

Fine-tunes `facebook/wav2vec2-base` for 7-class emotion classification on
the RAVDESS speech dataset, with class-weighted loss targeting known weak
classes (sad, angry), actor-independent evaluation, error analysis, and a
Gradio app ready for HuggingFace Spaces deployment.

## Project structure

```
SER-Project/
├── src/
│   ├── config.py              # all hyperparameters + label taxonomy live here
│   ├── generate_dataset_csv.py
│   ├── actor_split.py
│   ├── dataset.py             # PyTorch Dataset + dynamic-padding collator
│   ├── augment.py             # noise/pitch/time-stretch/gain augmentation
│   ├── train.py                # class-weighted fine-tuning
│   ├── evaluate.py             # test-set metrics, confusion matrix, predictions.csv
│   └── error_analysis.py       # confused pairs, worst mistakes, per-class recall
├── app/
│   ├── app.py                  # Gradio app (local or HF Spaces)
│   └── requirements.txt        # minimal deps for Spaces (lighter than root requirements.txt)
├── data/                        # not committed — see setup below
├── outputs/                     # trained model checkpoints (not committed)
├── reports/                     # evaluation outputs (not committed)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download the RAVDESS "Speech audio-only" dataset from Kaggle. You'll get
`Actor_01` ... `Actor_24` folders. Place them at `data/RAVDESS/`.

## Pipeline

### 1. Build dataset.csv
```bash
python src/generate_dataset_csv.py --data_dir data/RAVDESS --out data/dataset.csv
```
Merges neutral+calm into a single "neutral" class by default (7-class
taxonomy). To keep all 8 original classes instead, set
`MERGE_CALM_INTO_NEUTRAL = False` in `src/config.py` before running this.

Check the printed class distribution — "neutral" will look larger than
other classes per-actor. This is expected: RAVDESS has no "strong"
intensity version of neutral, and merging calm into it compounds this.

### 2. Actor-wise split
```bash
python src/actor_split.py --csv data/dataset.csv --out_dir data/splits
```
Splits by ACTOR (config: 19 train / 2 val / 3 test), never by row, so no
speaker's voice appears in more than one split.

### 3. Train
```bash
python src/train.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv
```
- CNN feature extractor is frozen; only the transformer + classification
  head are fine-tuned.
- Loss is class-weighted (inverse frequency, with an extra manual boost on
  "sad" and "angry" — see `EXTRA_CLASS_WEIGHT_BOOST` in `config.py`). This
  targets a known failure mode: sad is frequently confused with fearful/
  happy, and angry with happy, in baseline runs.
- fp16 is disabled by default — mixed precision has produced NaN losses in
  prior wav2vec2 fine-tuning runs. Only re-enable if you've verified it's
  stable on your setup.
- Model saved to `outputs/ser-model/final/`.

### 4. Evaluate
```bash
python src/evaluate.py --model_dir outputs/ser-model/final --test_csv data/splits/test.csv --out_dir reports
```
Produces `classification_report.txt`, `confusion_matrix.png`,
`metrics.json`, and `predictions.csv`.

### 5. Error analysis
```bash
python src/error_analysis.py --predictions reports/predictions.csv --out_dir reports
```
Surfaces the top confused label pairs, per-class recall (worst first), and
the most confident WRONG predictions — the most useful cases to listen to
manually and understand what's going wrong.

## Deploying to HuggingFace

### Push the model
```python
from huggingface_hub import HfApi
model.push_to_hub("your-username/wav2vec2-ser-ravdess")
feature_extractor.push_to_hub("your-username/wav2vec2-ser-ravdess")
```
(Run this after training, from a Python session with the trained `model`
and `feature_extractor` objects, or load them from `outputs/ser-model/final`
first.)

### Deploy the app on HF Spaces
1. Create a new Space → SDK: Gradio.
2. Set `MODEL_ID` in `app/app.py` (or as a Space secret/env var) to your
   pushed model repo id.
3. Upload `app/app.py` as the Space's `app.py`, and `app/requirements.txt`
   as the Space's `requirements.txt`.

## Known limitations / things worth improving further

- **Neutral class imbalance**: no "strong" intensity version exists for
  neutral in RAVDESS. Confirm the actual skew in your EDA before assuming
  the class weights fully compensate for it.
- **Small dataset**: ~1000-1100 training files after actor split. A 95M
  param transformer can overfit even with the CNN frozen — augmentation is
  applied from the start of training, not added later.
- **fp16 instability**: disabled by default in `config.py`. If you want
  faster training, test fp16 carefully on a short run first and watch for
  NaN losses before committing to a long run.
- **Emotion timeline / chunked long-audio prediction** is not implemented
  here. RAVDESS clips are themselves only 3-5 seconds, so any chunking
  demo would only be exercised on external audio, not evaluated against
  RAVDESS test labels — flag this clearly if you add it, rather than
  presenting timeline accuracy as validated.
