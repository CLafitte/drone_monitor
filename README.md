# Acoustic Drone Monitor

This repository contains an very-alpha modular acoustic quadcopter drone-detection pipeline designed
for remote, low-power edge devices (e.g., Raspberry Pi). It combines a lightweight classical DSP early-
warning system with a compact 1-bit CNN classifier for more precise confirmation.The goal is a fast, 
explainable, field-deployable quadcopter detector that improves over time with new data and model refinement.

---

## System Architecture
### Stage One: DSP Early-Warning Stage (fast, low-power)
- Captures 1–2 second audio windows from a USB microphone  
- Computes lightweight acoustic features:
  - harmonic spacing consistency  
  - spectral centroid  
  - AM-depth via analytic envelope  
  - broadband energy  
- Produces an **early suspicion score** from 0–1  
  - If suspicion is low → **skip CNN** (saves compute)

### Stage Two: 1-Bit CNN Classifier (precise, trainable)
- Operates only when DSP suspicion crosses a threshold  
- Uses 1-bit quantized convolutions for extremely efficient inference  
  - Input: log-mel (or STFT-based) spectrogram  
  - Output: **cnn_prob** = probability of “drone-like sound”

### Stage Three: Warning Logic
```bash
mic → DSP → (low?) → no alert
            (high?) → CNN → final drone_result
```

The system logs DSP metrics, CNN probabilities, and final decisions. Model accuracy will improve through 
ongoing dataset collection and training.

---

## Files in This Repository

| File | Purpose |
|------|---------|
| `drone_monitor.py` | Main orchestrator. Runs DSP and, if needed, the CNN. |
| `early_warning_dsp.py` | Feature extractor + classical DSP suspicion estimator. |
| `drone_monitor_cnn.py` | ONNX-based 1-bit CNN inference module. |

---

## Quick Start

You must have:
- Python 3.9+
- A USB microphone recognized by ALSA
- `pip install sounddevice soundfile numpy scipy onnxruntime`

Run:
python3 drone_monitor.py

Optional:
- Replace `drone_1bit.onnx` with your trained model
- Adjust DSP and CNN thresholds in `drone_monitor.py`

---

## Roadmap

- Model training pipeline (PyTorch + 1-bit quantization)
- Automated dataset collection
- Per-device calibration
- Integration with InfluxDB + Grafana dashboards
- Real-time web front-end
- Adaptive multi-mic support (upcoming research)

---

## Project Status

This is a very-alpha **proof-of-concept** intended for internal review.  
Expect rough edges and active changes as the pipeline evolves. Please keep
in mind that pre-release versions won't contain the trained CNN.

Contributions and critiques from are strongly encouraged.
