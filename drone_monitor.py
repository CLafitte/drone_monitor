#!/usr/bin/env python3
"""
drone_monitor.py
A quadcopter drone detection pipeline using:

1) Classical DSP early-warning system (fast, low-power)
2) 1-bit CNN classifier (precise, heavy)

Pipeline:
    mic → DSP early-warning → 
            LOW suspicion  → no alert
            HIGH suspicion → run CNN → final decision

This system is optimized for remote Raspberry Pi deployment 
with a generic 20Hz-20KHz vocal-range USB 2.0 microphone.
"""

import time
import numpy as np
import sounddevice as sd

from early_warning_dsp import (
    acquire_audio,
    compute_suspicion     # returns suspicion score in [0, 1]
)

from drone_monitor_cnn import DroneCNNInferencer

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------

FS = 48000
CAPTURE_SECONDS = 2.0

DSP_THRESHOLD = 0.55       # above this → run CNN
CNN_ALERT_THRESHOLD = 0.65 # above this → final drone detection

MODEL_PATH = "drone_1bit.onnx"  # change to your model
cnn = DroneCNNInferencer(MODEL_PATH)

# -----------------------------------------------------------
# Main Pipeline Function
# -----------------------------------------------------------

def analyze_one_window():
    """
    Capture audio, run DSP early-warning, and (if needed) the CNN.

    Returns dictionary:
        {
          "suspicion": float,
          "cnn_prob": float or None,
          "is_drone": 0/1
        }
    """
    # --- 1) Capture audio from mic ---
    audio = acquire_audio(FS, CAPTURE_SECONDS)
    if audio is None:
        return {"suspicion": 0.0, "cnn_prob": None, "is_drone": 0}

    # --- 2) Run DSP early-warning ---
    suspicion = compute_suspicion(audio, FS)

    if suspicion < DSP_THRESHOLD:
        # Low suspicion → skip CNN
        return {
            "suspicion": suspicion,
            "cnn_prob": None,
            "is_drone": 0
        }

    # --- 3) High suspicion → run CNN ---
    cnn_prob = cnn.predict_is_drone(audio, FS)

    is_drone = 1 if cnn_prob >= CNN_ALERT_THRESHOLD else 0

    return {
        "suspicion": suspicion,
        "cnn_prob": cnn_prob,
        "is_drone": is_drone
    }


# -----------------------------------------------------------
# Main loop (optional)
# -----------------------------------------------------------

def main_loop():
    print("[INFO] Drone Monitor Started.")
    print("[INFO] DSP threshold =", DSP_THRESHOLD)
    print("[INFO] CNN threshold =", CNN_ALERT_THRESHOLD)

    while True:
        result = analyze_one_window()

        suspicion = result["suspicion"]
        cnn_prob = result["cnn_prob"]
        is_drone = result["is_drone"]

        # ---- Logging for humans ----
        if cnn_prob is None:
            print(f"[DSP] suspicion={suspicion:.3f} → no CNN")
        else:
            print(f"[DSP={suspicion:.3f}] [CNN={cnn_prob:.3f}] → DRONE={is_drone}")

        # ---- (future) Save logs, audio clips, alerts, etc. ----
        # if is_drone:
        #     save_audio_clip(...)
        #     send_alert(...)

        time.sleep(0.1)


if __name__ == "__main__":
    main_loop()
