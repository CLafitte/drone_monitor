#!/usr/bin/env python3
"""
drone_monitor.py — MVP acoustic drone monitor

- Captures short audio windows (default 1.0s) using sounddevice.
- Computes lightweight features:
    - power spectrum (FFT)
    - spectral centroid
    - harmonic peak spacing heuristic
    - amplitude-modulation (AM) depth via analytic envelope
- Produces a suspicion_score (0..1) using a tunable heuristic.
- Atomic log burst can save windows above threshold
- Designed for lightweight remote ARM monitor (Raspberry Pi)
"""

import argparse
import os
import time
from datetime import datetime
import json

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import find_peaks, butter, sosfilt, hilbert

# ---------------- CONFIG ----------------
DEFAULT_RATE = 48000
DEFAULT_WINDOW = 1.0  # seconds
DEFAULT_THRESHOLD = 0.65
DEFAULT_DEVICE_KEYWORDS = ("usb", "mic")

# ---------------- UTIL ----------------
def log(tag, msg):
    ts = datetime.utcnow().isoformat() + "Z"
    print(f"[{ts}] [{tag}] {msg}")

# ---------------- DEVICE HELPERS ----------------
def find_input_device(keywords=DEFAULT_DEVICE_KEYWORDS):
    """Return an input-capable device matching keywords, with safe fallbacks."""
    devices = sd.query_devices()

    # 1. Keyword match
    for idx, dev in enumerate(devices):
        name = dev.get("name", "").lower()
        if dev.get("max_input_channels", 0) > 0 and any(k in name for k in keywords):
            log("INIT", f"device_selected name={dev['name']} index={idx}")
            return idx

    # 2. Default input device
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            log("INIT", f"no keyword match; using default input index={default_in}")
            return default_in
    except Exception:
        pass

    # 3. First available device
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            log("INIT", f"fallback first input device index={idx}")
            return idx

    raise RuntimeError("No input devices found")

# ---------------- DSP FEATURES ----------------
def highpass(x, fs, cut=20.0, order=4):
    """Stable high-pass using SOS filtering."""
    sos = butter(order, cut / (fs / 2), btype="highpass", output="sos")
    return sosfilt(sos, x)

def power_spectrum(x, fs):
    N = len(x)
    X = np.fft.rfft(x * np.hanning(N))
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    return freqs, P

def spectral_centroid(freqs, P):
    num = np.sum(freqs * P)
    den = np.sum(P) + 1e-12
    return num / den

def harmonic_spacing_score(freqs, P, low_hz=40, high_hz=2000, top_peaks=6):
    """Heuristic for harmonic spacing regularity."""
    # Low-energy guard
    if np.sum(P) < 1e-6:
        return 0.0

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    f = freqs[mask]
    p = P[mask]
    if len(p) == 0:
        return 0.0

    peaks, props = find_peaks(p, height=np.max(p) * 0.05, distance=5)
    if len(peaks) < 2:
        return 0.0

    peak_freqs = f[peaks]
    heights = props["peak_heights"]

    order = np.argsort(heights)[::-1][:top_peaks]
    selected = np.sort(peak_freqs[order])
    if len(selected) < 2:
        return 0.0

    diffs = np.diff(selected)
    if len(diffs) == 0:
        return 0.0

    median = np.median(diffs)
    std = np.std(diffs)

    consistency = max(0.0, 1.0 - (std / (median + 1e-6)))
    spacing_ok = 1.0 if (5.0 < median < 800.0) else 0.4

    return float(np.clip(consistency * spacing_ok, 0.0, 1.0))

def am_depth(x):
    """AM depth using analytic envelope, robust for low-SNR or quiet recordings."""
    rms = np.sqrt(np.mean(x**2))
    if rms < 1e-5:
        return 0.0

    analytic = hilbert(x)
    env = np.abs(analytic)

    # Normalize envelope by RMS for scale-independent AM detection
    env_norm = env / (rms + 1e-12)

    depth = np.std(env_norm) / (np.mean(env_norm) + 1e-12)
    return float(np.tanh(depth * 2.5))

# ---------------- SUSPICION SCORING ----------------
def compute_suspicion(x, fs):
    x = x.astype(np.float64)
    x = highpass(x, fs, cut=20.0)

    freqs, P = power_spectrum(x, fs)
    total_energy = np.sum(P) + 1e-12
    centroid = spectral_centroid(freqs, P)

    harmonic_score = harmonic_spacing_score(freqs, P, low_hz=40, high_hz=2000)
    am = am_depth(x)

    # energy gating
    energy_db = 10.0 * np.log10(total_energy + 1e-12)
    energy_score = 1.0 if energy_db > -60 else np.clip((energy_db + 80) / 40.0, 0.0, 1.0)

    score = (
        0.45 * harmonic_score +
        0.35 * am +
        0.20 * energy_score
    )
    score = float(np.clip(score, 0.0, 1.0))

    features = {
        "centroid_hz": float(np.round(centroid, 2)),
        "harmonic_score": float(np.round(harmonic_score, 3)),
        "am_depth": float(np.round(am, 3)),
        "energy_db": float(np.round(energy_db, 2)),
        "suspicion_score": float(np.round(score, 3)),
    }
    return score, features

# ---------------- ATOMIC SAVE ----------------
def atomic_save_flac(path, samples, fs):
    tmp = path + ".tmp"
    sf.write(tmp, samples, fs, format="FLAC", subtype="PCM_24")
    os.replace(tmp, path)

def atomic_save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------------- MAIN LOOP ----------------
def run_monitor(
    device,
    fs=DEFAULT_RATE,
    window=DEFAULT_WINDOW,
    threshold=DEFAULT_THRESHOLD,
    save_clips=False,
    clips_dir="detections",
    min_consecutive=1,
):
    if save_clips:
        os.makedirs(clips_dir, exist_ok=True)

    sd.default.samplerate = fs
    sd.default.channels = 1
    sd.default.device = device

    log("INIT", f"drone_monitor start window={window}s fs={fs} threshold={threshold}")
    consecutive_hits = 0
    frame_len = int(round(window * fs))

    while True:
        try:
            t0 = time.time()
            rec = sd.rec(frame_len, samplerate=fs, channels=1, dtype="float32")
            sd.wait()
            samples = rec.flatten()

            score, feats = compute_suspicion(samples, fs)
            log("DSP",
                f"suspicion_score={feats['suspicion_score']} "
                f"centroid={feats['centroid_hz']}Hz am={feats['am_depth']} "
                f"harmonic={feats['harmonic_score']} energy_db={feats['energy_db']}")

            if score >= threshold:
                consecutive_hits += 1
                log("INFER", f"hit score={score:.3f} consecutive={consecutive_hits}")
            else:
                consecutive_hits = 0

            if consecutive_hits >= min_consecutive and score >= threshold:
                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                clip_name = f"drone_{timestamp}.flac"
                clip_path = os.path.join(clips_dir, clip_name)

                event = {
                    "timestamp": timestamp,
                    "score": float(np.round(score, 3)),
                    "features": feats,
                    "device": str(device),
                }

                if save_clips:
                    atomic_save_flac(clip_path, samples, fs)
                    atomic_save_json(clip_path.replace(".flac", ".json"), event)
                    log("WRITE", f"saved {clip_name}")

                log("EVENT", f"drone_like score={score:.3f}")
                consecutive_hits = 0

            # pacing
            elapsed = time.time() - t0
            to_sleep = max(0.0, window - elapsed)
            if to_sleep > 0:
                time.sleep(to_sleep)

        except KeyboardInterrupt:
            log("CTRL", "keyboard interrupt — exiting")
            break
        except Exception as e:
            log("ERR", f"runtime exception: {e}")
            time.sleep(1.0)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="drone_monitor MVP (acoustic-only)")
    p.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    p.add_argument("--rate", type=int, default=DEFAULT_RATE)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--save-clips", action="store_true")
    p.add_argument("--clips-dir", default="detections")
    p.add_argument("--min-consecutive", type=int, default=1)
    p.add_argument("--device", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = int(args.device) if (args.device and args.device.isdigit()) else find_input_device()
    run_monitor(
        device=device,
        fs=args.rate,
        window=args.window,
        threshold=args.threshold,
        save_clips=args.save_clips,
        clips_dir=args.clips_dir,
        min_consecutive=args.min_consecutive,
    )

if __name__ == "__main__":
    main()
