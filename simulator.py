import json
import time
import threading
import colorsys
import numpy as np
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

AUDIO_PATH       = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.wav"
JSON_PATH        = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\light_script.json"
FEATURE_RES      = 0.1    # saniye — audio özellik çözünürlüğü
ZONE_NAMES       = ["BÜYÜK HALKA", "ORTA HALKA", "KÜÇÜK HALKA"]


# ── Ses çalma ────────────────────────────────────────────────────────────────

def play_audio():
    try:
        y, sr = librosa.load(AUDIO_PATH, sr=44100, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        sd.play(y.T, samplerate=sr)
    except Exception as e:
        print(f"Ses çalınamadı: {e}")


# ── Audio özellik pre-compute ─────────────────────────────────────────────────

def precompute_features(audio_path: str, resolution: float = 0.1) -> list[dict]:
    """
    Tüm şarkıyı 0.1s pencerelerle tara: RMS enerji + onset gücü.
    Onset'e vuruş sonrası yavaşça sönen decay uygular (beat pulse efekti).
    """
    print("Ses özellikleri hesaplanıyor (0.1s çözünürlük)…")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    ws    = int(resolution * sr)
    total = len(y) // ws

    rms_raw   = []
    onset_raw = []

    for i in range(total):
        seg   = y[i * ws : (i + 1) * ws]
        rms   = float(np.mean(librosa.feature.rms(y=seg)))
        onset = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))
        rms_raw.append(rms)
        onset_raw.append(onset)

    # Normalize
    rms_max   = max(rms_raw)   if rms_raw   else 1.0
    onset_max = max(onset_raw) if onset_raw else 1.0

    # Onset decay: vuruş anında sert spike, ardından 0.6^n ile söner
    DECAY = 0.6
    onset_decayed = []
    val = 0.0
    for o in onset_raw:
        val = max(o / onset_max, val * DECAY)
        onset_decayed.append(val)

    features = []
    for i in range(total):
        features.append({
            "t":      round(i * resolution, 3),
            "energy": rms_raw[i] / rms_max,
            "onset":  onset_decayed[i],
        })

    print(f"  {len(features)} özellik karesi hazır.")
    return features


# ── Renk yardımcıları ─────────────────────────────────────────────────────────

def smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp_hsl(c1, c2, t: float) -> list[int]:
    """HSL uzayında smootherstep ile geçiş."""
    t = smootherstep(t)
    r1, g1, b1 = (v / 255.0 for v in c1)
    r2, g2, b2 = (v / 255.0 for v in c2)
    h1, l1, s1 = colorsys.rgb_to_hls(r1, g1, b1)
    h2, l2, s2 = colorsys.rgb_to_hls(r2, g2, b2)
    dh = h2 - h1
    if dh >  0.5: dh -= 1.0
    if dh < -0.5: dh += 1.0
    h = (h1 + dh * t) % 1.0
    l = l1 + (l2 - l1) * t
    s = s1 + (s2 - s1) * t
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return [int(r * 255), int(g * 255), int(b * 255)]


def modulate(rgb: list[int], energy: float, onset: float,
             energy_weight: float = 1.0, onset_weight: float = 1.0) -> list[int]:
    """
    Duygusal baz rengi ses özelliklerine göre modüle et:
    - energy → parlaklık (L) ±%35
    - onset  → doygunluk (S) artışı, beat vuruşunda canlılanma
    """
    r, g, b = (max(0.02, v / 255.0) for v in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    l_mod = l * (0.68 + 0.50 * energy * energy_weight)
    s_mod = s * (0.80 + 0.45 * onset  * onset_weight)

    l_mod = max(0.03, min(0.95, l_mod))
    s_mod = max(0.00, min(1.00, s_mod))

    r2, g2, b2 = colorsys.hls_to_rgb(h, l_mod, s_mod)
    return [int(r2 * 255), int(g2 * 255), int(b2 * 255)]


def to01(rgb):
    return tuple(max(0.03, v / 255.0) for v in rgb)


def rgb_hex(rgb):
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


# ── Çizim ─────────────────────────────────────────────────────────────────────

def draw_rings(ax, z1, z2, z3):
    ax.clear()
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#000000")

    R1, R2, R3 = 1.15, 0.73, 0.38
    BORDER      = 0.028

    brightness = sum(z1) / 765.0
    for r, a in [(R1 + 0.18, 0.05 * brightness),
                 (R1 + 0.10, 0.10 * brightness),
                 (R1 + 0.04, 0.18 * brightness)]:
        ax.add_patch(plt.Circle((0, 0), r, color=to01(z1), alpha=a))

    ax.add_patch(plt.Circle((0, 0), R1,          color=to01(z1), zorder=1))
    ax.add_patch(plt.Circle((0, 0), R2 + BORDER, color="white",  zorder=2))
    ax.add_patch(plt.Circle((0, 0), R2,          color=to01(z2), zorder=3))
    ax.add_patch(plt.Circle((0, 0), R3 + BORDER, color="white",  zorder=4))
    ax.add_patch(plt.Circle((0, 0), R3,          color=to01(z3), zorder=5))


def make_info_panel(fig, info_x=0.67):
    fig.text(info_x, 0.90, "Tame Impala",               fontsize=13,
             color="#ffffff", fontfamily="monospace", fontweight="bold")
    fig.text(info_x, 0.84, "The Less I Know The Better", fontsize=9,
             color="#aaaaaa", fontfamily="monospace")

    labels, hexes = [], []
    for i, name in enumerate(ZONE_NAMES):
        y0 = 0.72 - i * 0.175
        fig.text(info_x, y0 + 0.04, name, fontsize=7, color="#555555",
                 fontfamily="monospace")
        lbl = fig.text(info_x, y0,        "─", fontsize=10,
                       color="#ffffff", fontfamily="monospace")
        hx  = fig.text(info_x, y0 - 0.04, "#??????", fontsize=9,
                       color="#888888", fontfamily="monospace")
        labels.append(lbl)
        hexes.append(hx)

    time_t    = fig.text(0.33, 0.03,  "", ha="center", fontsize=10,
                         color="#dddddd", fontfamily="monospace")
    emotion_t = fig.text(0.33, 0.005, "", ha="center", fontsize=9,
                         color="#9977bb", fontfamily="monospace", style="italic")
    return labels, hexes, time_t, emotion_t


def update_info(labels, hexes, zones):
    for i, rgb in enumerate(zones):
        labels[i].set_text(f"RGB  {tuple(rgb)}")
        h = rgb_hex(rgb)
        hexes[i].set_text(h)
        hexes[i].set_color(h)


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            light_data = json.load(f)
    except FileNotFoundError:
        print("light_script.json bulunamadı. Önce emotion_color_analyzer.py çalıştırın.")
        return

    # 0.1s audio özellikleri pre-compute
    if LIBROSA_AVAILABLE:
        audio_features = precompute_features(AUDIO_PATH, FEATURE_RES)
    else:
        audio_features = []
        print("librosa yok — audio modülasyonu devre dışı.")

    # Ses çalma
    if AUDIO_AVAILABLE and LIBROSA_AVAILABLE:
        threading.Thread(target=play_audio, daemon=True).start()
        time.sleep(0.3)

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#000000")
    fig.canvas.manager.set_window_title("Smart RGB Audio Analyzer — Simülatör")

    ax = fig.add_axes([0.01, 0.08, 0.62, 0.88])
    ax.set_facecolor("#000000")

    labels, hexes, time_t, emotion_t = make_info_panel(fig)

    plt.ion()
    plt.show()

    # Duygusal keyframe dt
    kf_dt = (light_data[1]["timestamp"] - light_data[0]["timestamp"]
             if len(light_data) >= 2 else 1.0)

    TARGET_FPS  = 30
    frame_sleep = 1.0 / TARGET_FPS
    start_time  = time.time()

    while plt.fignum_exists(fig.number):
        elapsed = time.time() - start_time

        if elapsed > light_data[-1]["timestamp"] + kf_dt:
            break

        # ── Duygusal baz rengi (Claude, 1s) ──────────────────────────────
        raw_idx = elapsed / kf_dt
        idx_lo  = max(0, min(int(raw_idx), len(light_data) - 2))
        idx_hi  = idx_lo + 1
        t_kf    = max(0.0, min(1.0, raw_idx - idx_lo))

        kf_lo = light_data[idx_lo]
        kf_hi = light_data[idx_hi]

        base1 = lerp_hsl(kf_lo["zone1"], kf_hi["zone1"], t_kf)
        base2 = lerp_hsl(kf_lo["zone2"], kf_hi["zone2"], t_kf)
        base3 = lerp_hsl(kf_lo["zone3"], kf_hi["zone3"], t_kf)

        # ── Audio modülasyonu (librosa, 0.1s) ────────────────────────────
        if audio_features:
            feat_idx = min(int(elapsed / FEATURE_RES), len(audio_features) - 1)
            feat     = audio_features[feat_idx]
            energy   = feat["energy"]
            onset    = feat["onset"]

            # Dış halka en reaktif, iç daire en sakin
            z1 = modulate(base1, energy, onset, energy_weight=1.0, onset_weight=1.0)
            z2 = modulate(base2, energy, onset, energy_weight=0.75, onset_weight=0.65)
            z3 = modulate(base3, energy, onset, energy_weight=0.50, onset_weight=0.35)
        else:
            z1, z2, z3 = base1, base2, base3

        draw_rings(ax, z1, z2, z3)
        update_info(labels, hexes, [z1, z2, z3])

        mins = int(elapsed // 60)
        secs = elapsed % 60
        time_t.set_text(
            f"♪  {mins}:{secs:05.2f}  |  Tame Impala — The Less I Know The Better")

        emotion = kf_lo.get("emotion", "")
        if emotion:
            emotion_t.set_text(f'"{emotion}"')

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        next_tick = start_time + (int(elapsed / frame_sleep) + 1) * frame_sleep
        wait = next_tick - time.time()
        plt.pause(max(0.001, wait))

    if AUDIO_AVAILABLE:
        sd.stop()
    plt.ioff()
    plt.close("all")
    print("Simülasyon tamamlandı.")


if __name__ == "__main__":
    main()
