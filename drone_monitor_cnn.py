#!/usr/bin/env python3
"""
drone_monitor_cnn.py
Lightweight 1-bit CNN classifier for quadcopter drone acoustic signatures.

This module is designed to run side-by-side with a classical DSP early-warning
system. The CNN provides precise classification once triggered. 

Pipeline:
    audio → mel-spectrogram → normalize → 1-bit → CNN → P(drone)

The binarized 1-bit representation drastically reduces model size and improves
robustness in noisy environments with low SNR conditions.

Author: Connor Lafitte Audio | github.com/CLafitte
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# Optional ONNX runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# -----------------------------------------------------------
# 1-BIT CONVOLUTION LAYER
# -----------------------------------------------------------

class BinaryConv2d(nn.Conv2d):
    """
    A convolution layer that binarizes weights and inputs to 1-bit
    during the forward pass. Stored weights remain float32.
    """

    def binarize(self, x):
        return torch.sign(x)

    def forward(self, x):
        x_b = self.binarize(x)
        w_b = self.binarize(self.weight)
        return F.conv2d(x_b, w_b, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# -----------------------------------------------------------
# Tiny 1-Bit CNN Architecture (≈7k parameters)
# -----------------------------------------------------------

class Drone1BitCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = BinaryConv2d(1, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = BinaryConv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = BinaryConv2d(16, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)

        self.fc    = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        # x: [batch, 1, 64, 64] (64x64 mel-bins)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 32x32

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16x16

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 8x8

        x = torch.flatten(x, 1)

        logits = self.fc(x)
        return F.softmax(logits, dim=1)


# -----------------------------------------------------------
# Mel + Binarization Preprocessor
# -----------------------------------------------------------

def preprocess_audio(audio, fs, target_len=2.0):
    """
    Convert audio into a binarized 64×64 mel spectrogram.
    Includes:
      - AGC-style scaling for low-volume drone recordings
      - clipping/padding to target_len seconds
      - mel → z-score normalize → 1-bit sign()

    Returns: float32 tensor of shape [1, 1, 64, 64]
    """

    # Pad / clip to fixed duration
    N = int(target_len * fs)
    if len(audio) < N:
        pad = np.zeros(N - len(audio))
        audio = np.concatenate([audio, pad])
    else:
        audio = audio[:N]

    # AGC-like normalization (helps weak drone signatures)
    rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
    if rms < 0.01:
        audio = audio * (0.01 / rms)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=fs,
        n_fft=2048,
        hop_length=512,
        n_mels=64,
        fmin=20,
        fmax=12000
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-9)

    # Resize to 64×64 using librosa
    mel_resized = librosa.util.fix_length(mel, 64, axis=1)

    # Binarize
    mel_bin = np.sign(mel_resized)

    # Shape → Torch tensor
    x = torch.tensor(mel_bin, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x


# -----------------------------------------------------------
# PyTorch model loader
# -----------------------------------------------------------

def load_pytorch_model(path):
    model = Drone1BitCNN(num_classes=2)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------------------------------------
# ONNX model loader
# -----------------------------------------------------------

def load_onnx_model(path):
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime is not installed on this system.")
    return ort.InferenceSession(path)


# -----------------------------------------------------------
# Main inference entry point
# -----------------------------------------------------------

class DroneCNNInferencer:
    """
    Unified inference wrapper. Supports:
       - PyTorch models (.pt, .pth)
       - ONNX models (.onnx)

    Usage:
        infer = DroneCNNInferencer(path="model.onnx")
        p = infer.predict_is_drone(audio, fs)
    """

    def __init__(self, path, backend="auto"):
        self.path = path
        self.backend = backend
        self.model = None

        if backend == "pytorch" or (backend == "auto" and path.endswith((".pt", ".pth"))):
            self.backend = "pytorch"
            self.model = load_pytorch_model(path)

        elif backend == "onnx" or (backend == "auto" and path.endswith(".onnx")):
            self.backend = "onnx"
            self.model = load_onnx_model(path)

        else:
            raise ValueError("Could not infer model backend from filename. "
                             "Use backend='pytorch' or backend='onnx' explicitly.")

    # -------------------------------------------------------

    def predict_is_drone(self, audio, fs):
        """
        Returns: probability (0.0–1.0) that the signal contains a drone.
        """

        x = preprocess_audio(audio, fs).numpy()

        if self.backend == "pytorch":
            with torch.no_grad():
                p = self.model(torch.tensor(x)).numpy()[0][1]
            return float(p)

        elif self.backend == "onnx":
            inputs = {self.model.get_inputs()[0].name: x}
            outputs = self.model.run(None, inputs)[0]
            return float(outputs[0][1])


# -----------------------------------------------------------
# Training Stub (for future dataset work)
# -----------------------------------------------------------

def training_stub():
    """
    Placeholder for future training.
    This function will eventually:
      - Load labeled drone/ambient dataset
      - Perform data augmentation
      - Train 1-bit CNN
      - Export PyTorch and ONNX versions

    Not implemented yet.
    """
    raise NotImplementedError("Training pipeline will be added after dataset assembly.")
 
