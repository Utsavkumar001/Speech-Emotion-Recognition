"""
app.py

Gradio web app for the Speech Emotion Recognition model. Designed to run
either locally or as a HuggingFace Space.

Deployment notes (HF Spaces):
    1. Create a new Space, SDK = Gradio.
    2. Set the MODEL_ID environment variable / edit MODEL_ID below to point
       to your pushed model repo, e.g. "yourusername/wav2vec2-ser-ravdess".
    3. Copy this file to the Space as app.py, and copy requirements.txt too.

Usage (local):
    python app/app.py
"""

import os
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Change this to your own HF model repo once trained + pushed.
MODEL_ID = os.environ.get("MODEL_ID", "utsav05/wav2vec2-ser-ravdess")
SAMPLE_RATE = 16000

print(f"Loading model: {MODEL_ID}")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

LABELS = [model.config.id2label[i] for i in range(len(model.config.id2label))]


def preprocess_audio(audio_input):
    """Gradio audio input arrives as (sample_rate, numpy_array)."""
    sr, waveform = audio_input

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)

    # Normalize integer PCM formats to [-1, 1] if needed
    if np.abs(waveform).max() > 1.0:
        waveform = waveform / np.abs(waveform).max()

    if sr != SAMPLE_RATE:
        import torchaudio
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(wav_tensor).squeeze(0).numpy()

    return waveform


def make_waveform_plot(waveform: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 2))
    times = np.linspace(0, len(waveform) / SAMPLE_RATE, len(waveform))
    ax.plot(times, waveform, linewidth=0.5, color="#4C72B0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    return fig


def predict_emotion(audio_input):
    if audio_input is None:
        return None, None, "Please upload or record an audio clip."

    waveform = preprocess_audio(audio_input)

    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    confidence_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top_label = LABELS[int(np.argmax(probs))]
    top_confidence = float(np.max(probs))

    waveform_plot = make_waveform_plot(waveform)
    summary = f"**Predicted emotion: {top_label}** ({top_confidence*100:.1f}% confidence)"

    return confidence_dict, waveform_plot, summary


with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown("# Speech Emotion Recognition")
    gr.Markdown(
        "Upload an audio clip or record from your microphone. "
        "The model predicts the speaker's emotional state from acoustic "
        "features alone (wav2vec2 fine-tuned on RAVDESS)."
    )

    with gr.Row():
        audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Speech input")

    predict_btn = gr.Button("Predict Emotion", variant="primary")

    with gr.Row():
        confidence_output = gr.Label(num_top_classes=len(LABELS), label="Emotion probabilities")
        waveform_output = gr.Plot(label="Waveform")

    summary_output = gr.Markdown()

    predict_btn.click(
        fn=predict_emotion,
        inputs=audio_input,
        outputs=[confidence_output, waveform_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
