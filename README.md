# Smart RGB Audio Analyzer 🎵💡

> **🇹🇷 Türkçe için → [README.tr.md](README.tr.md)**

An AI-driven music-to-light system. Instead of mapping instruments to fixed
colors, an LLM **listens to the emotional arc of a song** and decides what
colors a human would *feel* at each moment — then a real-time simulator paints
those emotions as a living field of light.

Built as a collaborative project between Computer Engineering and
Electrical–Electronics Engineering, with the long-term goal of driving physical
smart lighting (ESP32 / WS2812B LED strips).

---

## ✨ The Core Idea

Most audio visualizers are mechanical: bass → red, treble → blue. This project
takes a different path.

1. **librosa** extracts acoustic features from the track (energy, onset, spectral
   brightness, dominant note) one second at a time.
2. **Claude** ([Anthropic API](https://www.anthropic.com)) receives those features
   *with the song's emotional context* and returns, for every second:
   - a short **emotion label** (e.g. *"euphoric melancholy"*, *"jealous ache"*)
   - **three independent RGB colors** — one each for the rhythm, melody and vocal layers
3. Colors are **smoothed in HSL space** so transitions feel like a slow sunset,
   never a harsh flash.
4. A **real-time gradient simulator** renders the result as 12 glowing lights,
   each tied to a frequency band, pulsing in sync with the music.

---

## 🚀 Features

- **Emotion-first color generation** — the AI interprets feeling, not just frequency.
- **Three independent zones** — rhythm, melody and vocals each get their own color story.
- **Organic gradient field** — 12 frequency-band lights in a slowly rotating
  sunflower layout, blended in *screen* mode so colors stay vivid and distinct.
- **Frequency-reactive** — bass, mids and treble explode independently instead of
  one global brightness pulse.
- **Spectrum hue spread** — the emotional tone stays at the center while colors
  fan out across the full wheel for a richer look (`HUE_SPREAD`).
- **Silky transitions** — HSL Gaussian smoothing + `smootherstep` interpolation
  between keyframes at render time.
- **Tight audio sync** — audio is preloaded and playback starts back-to-back with
  the visual clock (no drift).
- **~50 FPS** — `blit` rendering caches the static background and redraws only what changes.
- **Resilient analysis** — small batches, JSON-salvage parsing, retries, and
  neighbor-fill for any missing frames.

---

## 📂 Project Structure

```text
emotion_color_analyzer.py   # ⭐ Main pipeline: features → Claude → emotional colors → JSON
simulator.py                # ⭐ Real-time gradient light-field visualizer (reads the JSON)
light_script.json           # Generated time-synced color timeline (the AI's output)
requirements.txt            # Python dependencies

stem_processor.py           # Experimental: Demucs stem separation (reference only)
calibrate.py                # Dev tool: measure feature ranges per stem
check_json.py               # Dev tool: inspect the generated JSON
```

> `audio_processor.py` and `visualization.py` belong to an earlier
> instrument→fixed-color approach and are kept for history only.

---

## 🔧 Requirements

- **Python 3.12** (3.14 is not yet compatible with the audio stack)
- An **Anthropic API key** → [console.anthropic.com](https://console.anthropic.com)
- **FFmpeg** (only if you need to convert an MP3 to WAV)

Python packages (see `requirements.txt`):
`anthropic`, `librosa`, `numpy`, `matplotlib`, `sounddevice`, `soundfile`

---

## 📦 Setup

```bash
# 1. Create and activate a Python 3.12 virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide your API key
#   Windows (PowerShell):
$env:ANTHROPIC_API_KEY = "sk-ant-..."
#   macOS/Linux:
#   export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## ▶️ Usage

```bash
# Step 1 — analyze the song (calls Claude, writes light_script.json)
python emotion_color_analyzer.py

# Step 2 — play the light show in sync with the music
python simulator.py
```

Want to analyze a **different song**? Drop your `.wav` in place and edit the song
block at the top of `emotion_color_analyzer.py`:

```python
SONG_TITLE  = "The Less I Know The Better"
SONG_ARTIST = "Tame Impala"
SONG_INFO   = "Bittersweet psychedelic pop. Emotional arc: longing → euphoria → nostalgia."
```

---

## ⚙️ Tuning

**`emotion_color_analyzer.py`**

| Constant | Default | What it does |
|----------|---------|--------------|
| `WINDOW_SEC` | `1.0` | Seconds per analysis window (one color keyframe) |
| `BATCH_SEC`  | `10.0` | Window seconds sent to Claude per request |
| `SIGMA`      | `3.0` | HSL smoothing width (higher = softer transitions) |
| `MODEL`      | `claude-haiku-4-5` | Claude model used for analysis |

**`simulator.py`**

| Constant | Default | What it does |
|----------|---------|--------------|
| `N_BANDS`      | `12` | Number of frequency-band lights |
| `HUE_SPREAD`   | `0.62` | Color variety across the spectrum (0 = emotion tone only) |
| `AUDIO_OFFSET` | `0.0` | Sync fine-tune (s). +ve if visuals lead, −ve if they lag |
| `TARGET_FPS`   | `50` | Frame-rate cap |

---

## 🗺️ Roadmap

- [x] AI emotional color analysis (Claude)
- [x] Independent rhythm / melody / vocal zones
- [x] Smooth HSL transitions
- [x] Real-time gradient simulator with frequency-band lights
- [ ] Real-time analysis (no pre-processing step)
- [ ] Physical hardware: ESP32 + WS2812B LED strips
- [ ] Microphone / live-input mode

---

## 🙏 Acknowledgements

- Emotional analysis powered by [Claude](https://www.anthropic.com) (Anthropic API)
- Audio features via [librosa](https://librosa.org)
- Reference track: *"The Less I Know The Better"* — Tame Impala
