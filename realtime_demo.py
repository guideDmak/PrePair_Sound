"""Realtime RakForest CNN demo for a Mac microphone.

Run from the project root with:
    python realtime_demo.py

The script only loads the trained checkpoint. It does not train or modify
the model, raw dataset, or processed dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

try:
    import sounddevice as sd
except ModuleNotFoundError:
    print("ไม่พบ sounddevice — ติดตั้งด้วยคำสั่ง: pip install sounddevice", file=sys.stderr)
    raise SystemExit(1)

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio


SAMPLE_RATE = 16_000
CLIP_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.50
MODEL_PATH = Path(__file__).resolve().parent / "models" / "rakforest_cnn_best.pt"


class SmallAudioCNN(nn.Module):
    """Architecture must match notebooks/03_train_cnn.ipynb exactly."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ไม่พบ checkpoint: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = SmallAudioCNN(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def make_mel_transform(device: torch.device):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=20,
        f_max=SAMPLE_RATE // 2,
    ).to(device)


def to_log_mel(
    waveforms: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    device: torch.device,
) -> torch.Tensor:
    """Match the on-the-fly preprocessing in notebooks/03_train_cnn.ipynb."""
    mel = mel_transform(waveforms.to(device))
    mel = torch.log(mel.clamp_min(1e-6))
    mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / (
        mel.std(dim=(-2, -1), keepdim=True) + 1e-6
    )
    # torchaudio output: (batch, mel, time); CNN input: (batch, 1, mel, time).
    return mel.unsqueeze(1)


def resample_to_16k(audio: np.ndarray, original_rate: int) -> np.ndarray:
    """Convert a microphone chunk to mono 16 kHz."""
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if original_rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=original_rate, target_sr=SAMPLE_RATE)
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size < NUM_SAMPLES:
        mono = np.pad(mono, (0, NUM_SAMPLES - mono.size))
    return mono[:NUM_SAMPLES]


def classify_chunk(
    audio: np.ndarray,
    original_rate: int,
    model: nn.Module,
    class_names: list[str],
    mel_transform: torchaudio.transforms.MelSpectrogram,
    device: torch.device,
):
    waveform = torch.from_numpy(resample_to_16k(audio, original_rate)).unsqueeze(0)
    with torch.no_grad():
        logits = model(to_log_mel(waveform, mel_transform, device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    if confidence >= CONFIDENCE_HIGH:
        level = "สูง"
        message = f"ตรวจพบเสียงต้องสงสัย: {predicted_class}"
    elif confidence >= CONFIDENCE_MEDIUM:
        level = "ปานกลาง"
        message = f"อาจเป็นเสียง {predicted_class} — ควรตรวจสอบเพิ่มเติม"
    else:
        level = "ต่ำ"
        message = "ผลยังไม่แน่ชัด"

    return predicted_class, confidence, level, message, probabilities


def print_result(
    predicted_class: str,
    confidence: float,
    level: str,
    message: str,
    probabilities: np.ndarray,
    class_names: list[str],
) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    probability_text = " | ".join(
        f"{name}={prob:.2%}" for name, prob in zip(class_names, probabilities)
    )
    print(
        f"[{timestamp}] class={predicted_class} | "
        f"confidence={confidence:.2%} | level={level}"
    )
    print(f"  {message}")
    print(f"  probabilities: {probability_text}")


def main() -> None:
    device = get_device()
    model, class_names = load_model(device)
    mel_transform = make_mel_transform(device)

    input_device = sd.query_devices(kind="input")
    microphone_rate = int(round(input_device["default_samplerate"]))
    if microphone_rate <= 0:
        raise RuntimeError("ไม่พบ sample rate ของ default microphone")

    print("RakForest realtime demo")
    print(f"device={device} | microphone={input_device['name']}")
    print(f"microphone sample rate={microphone_rate} Hz | chunk={CLIP_SECONDS} seconds")
    print("กด Ctrl+C เพื่อหยุดอย่างปลอดภัย\n")

    try:
        while True:
            print("กำลังรับเสียง...", end="", flush=True)
            audio = sd.rec(
                int(microphone_rate * CLIP_SECONDS),
                samplerate=microphone_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            )
            print(" ประมวลผล")
            result = classify_chunk(
                audio,
                microphone_rate,
                model,
                class_names,
                mel_transform,
                device,
            )
            print_result(*result, class_names)
            print()
    except KeyboardInterrupt:
        print("\nหยุดการรับเสียงแล้ว")
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาดระหว่างรับเสียง: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
