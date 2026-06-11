
import os

# 1. DOSYA İÇERİKLERİNİ TANIMLA

readme_content = """# Smart RGB Audio Analyzer 🎵💡

An advanced, AI-ready 3-zone audio spectrum analyzer that processes music files, extracts distinct frequency features, and maps them onto smooth, dynamic RGB color patterns. Designed to drive hardware-based smart lighting systems (like ESP32/WS2812B LED strips) with high-fidelity music synchronization.

Developed as a collaborative project between Computer Engineering and Electrical-Electronics Engineering.

---

## 🚀 Features

* **Full-Song Mel-Spectrogram Visualization:** Converts standard audio into human-ear-scaled visual heatmaps.
* **3-Zone Frequency Splitting:** Separates audio slices into independent channels:
  * **Left Zone (Bass & Drums):** Low-frequency range mapped to vibrant pinks/reds.
  * **Middle Zone (Guitars & Synths):** Mid-frequency range mapped to deep purples/blues.
  * **Right Zone (Vocals):** High-frequency range mapped to retro turquoise/greens.
* **Advanced DSP Effects:**
  * **Noise Gate:** Automatically blacks out individual light zones when instruments or vocals go silent.
  * **Dynamic Contrast:** Amplifies volume peaks and dims ambient noises for a dramatic pulsing effect.
  * **Exponential Moving Average (Smoothing):** Blends sequential colors seamlessly to eliminate harsh digital flashes and create organic fluid transitions.
* **Hardware-Ready JSON Output:** Exports time-synced RGB matrices to a lightweight JSON file.
* **Virtual LED Simulator:** Includes a built-in interactive simulation window to test and visualize the light patterns in real-time.

---

## 📂 Project Structure

```text
├── visualization.py      # Generates full-track Mel-Spectrogram analysis
├── audio_processor.py    # Sliding window feature extraction & 3-zone color mapping
├── simulator.py          # Real-time virtual 3-zone LED lamp preview
├── light_script.json     # Generated time-synced RGB dataset for hardware use
└── .gitignore            # Prevents local media (.mp3) from cluttering the repo
