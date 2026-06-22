"""
Emotion-based color analyzer — "The Less I Know The Better" by Tame Impala.
Improved: richer audio features, song structure context, per-zone colors,
30-second batches so Claude sees the full emotional arc.
"""

import os
import json
import colorsys
import math
import numpy as np
import librosa
import anthropic

PROJECT_DIR = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje"
AUDIO_PATH  = os.path.join(PROJECT_DIR, "theless.wav")
STEMS_DIR   = os.path.join(PROJECT_DIR, "separated", "htdemucs", "theless")
OUTPUT_JSON = os.path.join(PROJECT_DIR, "light_script.json")

WINDOW_SEC   = 1.0
BATCH_SEC    = 30.0     # 30 saniye per batch — Claude daha geniş bağlam görür
SMOOTH_SIGMA = 3.0
MODEL        = "claude-haiku-4-5"

# ── Şarkı yapısı ─────────────────────────────────────────────────────────────
# Claude bu bilgiyle hangi bölümde olduğunu bilerek renk seçer

SONG_STRUCTURE = [
    (  0,  20, "INTRO",        "drums only, building mystery and anticipation"),
    ( 20,  60, "VERSE 1",      "bass groove enters, jealous longing begins — 'she had a boyfriend'"),
    ( 60,  83, "PRE-CHORUS 1", "tension rising, longing intensifies, emotional climb"),
    ( 83, 115, "CHORUS 1",     "euphoric release — 'the less I know the better' — aching nostalgia"),
    (115, 155, "VERSE 2",      "same groove, emotion deepens, bittersweet acceptance grows"),
    (155, 178, "PRE-CHORUS 2", "bigger build, more urgent longing"),
    (178, 215, "CHORUS 2",     "peak euphoria, full arrangement, cathartic release"),
    (215, 250, "BRIDGE",       "psychedelic breakdown, floating, dreamy introspection"),
    (250, 300, "OUTRO",        "gradual fade, wistful acceptance, letting go"),
]

SONG_CONTEXT = """\
Song: "The Less I Know The Better" — Tame Impala (2015)
BPM: ~116 | Genre: psychedelic pop / neo-disco / dream pop
Overall emotional arc: jealous longing → euphoric release → dreamy nostalgia → wistful acceptance
Essence: wanting something you can't have, and finding unexpected beauty in that loss.

COLOR MOOD GUIDE (map emotions to these families):
  MYSTERY / ANTICIPATION  → deep electric blue      R:20-60   G:30-90   B:180-240
  JEALOUS LONGING         → dark crimson-indigo     R:90-150  G:10-45   B:130-190
  TENSION BUILDING        → magenta-crimson         R:160-210 G:20-65   B:90-160
  EUPHORIC RELEASE        → bright pink-magenta     R:215-255 G:55-120  B:160-225
  DREAMY FLOAT            → electric violet-blue    R:100-160 G:70-140  B:200-255
  WISTFUL ACCEPTANCE      → warm rose-coral         R:175-215 G:75-115  B:110-165

ZONE ROLES (each zone tells a different part of the same emotional story):
  zone1 (outer ring) = the RHYTHM / PHYSICAL feeling — most reactive to beat energy
  zone2 (middle)     = the MELODIC / HARMONIC feeling — the song's "color"
  zone3 (inner disc) = the VOCAL / INTIMATE feeling — what Kevin is saying inside"""


def get_section(t: float) -> tuple[str, str]:
    for start, end, name, desc in SONG_STRUCTURE:
        if start <= t < end:
            return name, desc
    return "OUTRO", "fading away"


# ── Audio feature extraction ──────────────────────────────────────────────────

def load_stem(name: str, sr: int) -> np.ndarray | None:
    path = os.path.join(STEMS_DIR, f"{name}.wav")
    if not os.path.exists(path):
        return None
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_features(audio_path: str) -> list[dict]:
    print("Ses yükleniyor…")
    y_mix, sr = librosa.load(audio_path, sr=None, mono=True)
    ws         = int(WINDOW_SEC * sr)
    total      = len(y_mix) // ws

    # Stem'leri yükle (varsa)
    stems = {}
    for name in ["drums", "bass", "vocals", "other"]:
        s = load_stem(name, sr)
        if s is not None:
            stems[name] = s
    has_stems = len(stems) > 0
    if has_stems:
        print(f"  Stem'ler bulundu: {list(stems.keys())}")

    print(f"Özellik çıkarılıyor — {total} kare ({total * WINDOW_SEC:.0f}s)…")
    frames = []
    prev_rms = 0.0

    for i in range(total):
        seg = y_mix[i * ws: (i + 1) * ws]

        rms      = float(np.mean(librosa.feature.rms(y=seg)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        onset    = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))
        zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=seg)))

        # Harmonik / perküsif ayrımı
        y_harm, y_perc = librosa.effects.hpss(seg)
        harm_energy = float(np.mean(np.abs(y_harm)))
        perc_energy = float(np.mean(np.abs(y_perc)))
        total_e     = harm_energy + perc_energy + 1e-9
        harm_ratio  = round(harm_energy / total_e, 3)   # 1=tamamen melodik, 0=tamamen ritim

        # Enerji trendi
        delta = rms - prev_rms
        trend = "rising" if delta > 0.0008 else "falling" if delta < -0.0008 else "steady"
        prev_rms = rms

        # Dominant nota
        chroma   = librosa.feature.chroma_stft(y=seg, sr=sr)
        note_idx = int(np.argmax(np.mean(chroma, axis=1)))
        note     = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][note_idx]

        frame = {
            "timestamp":    round(i * WINDOW_SEC, 2),
            "energy":       round(rms, 5),
            "brightness":   round(centroid / (sr / 2), 4),
            "onset":        round(onset, 3),
            "roughness":    round(zcr, 5),
            "harmonic":     harm_ratio,
            "trend":        trend,
            "note":         note,
        }

        # Stem enerjileri
        if has_stems:
            for sname, sy in stems.items():
                if i * ws + ws <= len(sy):
                    sseg = sy[i * ws: (i + 1) * ws]
                else:
                    sseg = sy[-ws:] if len(sy) >= ws else sy
                frame[sname] = round(float(np.mean(librosa.feature.rms(y=sseg))), 5)

        frames.append(frame)

    return frames


# ── Claude API ────────────────────────────────────────────────────────────────

def build_prompt(batch: list[dict]) -> str:
    t_start = batch[0]["timestamp"]
    t_end   = batch[-1]["timestamp"]
    sec_name, sec_desc = get_section(t_start)

    # Özellik satırları
    has_stems = "drums" in batch[0]
    lines = []
    for f in batch:
        line = (f"t={f['timestamp']:6.1f}s  "
                f"energy={f['energy']:.4f}  "
                f"bright={f['brightness']:.3f}  "
                f"onset={f['onset']:.2f}  "
                f"harmonic={f['harmonic']:.2f}  "
                f"trend={f['trend']:7s}  "
                f"note={f['note']:3s}")
        if has_stems:
            line += (f"  drums={f.get('drums',0):.4f}"
                     f"  bass={f.get('bass',0):.4f}"
                     f"  vocals={f.get('vocals',0):.4f}")
        lines.append(line)

    return f"""{SONG_CONTEXT}

═══════════════════════════════════════════════
CURRENT BATCH: t={t_start:.0f}s – {t_end:.0f}s
SONG SECTION:  {sec_name} — {sec_desc}
═══════════════════════════════════════════════

You are designing an AMBIENT LIGHT SHOW for this song. Your goal is to create
a COHERENT COLOR NARRATIVE across all {len(batch)} windows in this batch.
Think like a lighting director: set the mood for the whole section, then let
individual moments breathe within that mood.

Rules:
1. Colors should FLOW — consecutive windows shift by ≤40 per channel max.
2. Match the section mood above — don't contradict it.
3. Each zone tells its own part of the story (outer=body/rhythm, middle=melody, inner=soul/voice).
4. Higher onset + lower harmonic → more saturated, energetic outer ring.
5. Higher harmonic + vocals → warmer, more intimate inner disc.

Audio features (1-second windows):
{chr(10).join(lines)}

Return ONLY a valid JSON array, one object per window, no extra text:
[
  {{
    "t": 0.0,
    "emotion": "one short phrase",
    "zone1": [r, g, b],
    "zone2": [r, g, b],
    "zone3": [r, g, b]
  }},
  ...
]"""


def query_claude(client: anthropic.Anthropic, batch: list[dict]) -> list[dict]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": build_prompt(batch)}],
    )
    text  = response.content[0].text.strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"JSON array bulunamadı:\n{text[:400]}")
    return json.loads(text[start:end])


# ── HSL smoothing ─────────────────────────────────────────────────────────────

def _to_hls(rgb):
    r, g, b = (v / 255.0 for v in rgb)
    return colorsys.rgb_to_hls(r, g, b)


def _from_hls(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return [int(r * 255), int(g * 255), int(b * 255)]


def gaussian_smooth(colors: list[list[int]], sigma: float) -> list[list[int]]:
    n      = len(colors)
    radius = max(1, int(math.ceil(3 * sigma)))
    kernel = [math.exp(-0.5 * ((k - radius) / sigma) ** 2)
              for k in range(2 * radius + 1)]
    k_sum  = sum(kernel)
    kernel = [w / k_sum for w in kernel]

    hls_all = [_to_hls(c) for c in colors]
    result  = []

    for i in range(n):
        ref_h  = hls_all[i][0]
        wh = wl = ws_acc = tw = 0.0
        for ki, kw in enumerate(kernel):
            j = i - radius + ki
            if j < 0 or j >= n:
                continue
            h, l, s = hls_all[j]
            dh = h - ref_h
            if dh >  0.5: dh -= 1.0
            if dh < -0.5: dh += 1.0
            wh     += (ref_h + dh) * kw
            wl     += l * kw
            ws_acc += s * kw
            tw     += kw
        if tw > 0:
            result.append(_from_hls(wh / tw, wl / tw, ws_acc / tw))
        else:
            result.append(colors[i])
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY ayarlanmamış.")
        return

    client   = anthropic.Anthropic(api_key=api_key)
    features = extract_features(AUDIO_PATH)
    total    = len(features)
    print(f"Toplam: {total} kare ({total * WINDOW_SEC:.0f}s)")

    batch_size = int(BATCH_SEC / WINDOW_SEC)  # 30 kare per batch
    n_batches  = math.ceil(total / batch_size)
    print(f"Claude'a {n_batches} batch gönderiliyor ({MODEL}, 30s/batch)…\n")

    raw = {"zone1": [], "zone2": [], "zone3": []}
    emotions: list[str] = []

    for b_idx in range(n_batches):
        lo    = b_idx * batch_size
        hi    = min(lo + batch_size, total)
        batch = features[lo:hi]
        sec_name, _ = get_section(batch[0]["timestamp"])

        print(f"  [{b_idx+1:2d}/{n_batches}]  "
              f"t={batch[0]['timestamp']:5.0f}s–{batch[-1]['timestamp']:5.0f}s  "
              f"({sec_name})  ", end="", flush=True)

        try:
            results = query_claude(client, batch)
        except Exception as exc:
            print(f"HATA — {exc}")
            fallback = [120, 60, 180]
            for _ in batch:
                raw["zone1"].append(fallback)
                raw["zone2"].append(fallback)
                raw["zone3"].append(fallback)
                emotions.append("unknown")
            continue

        result_map = {r["t"]: r for r in results if isinstance(r, dict)}
        batch_emotions = []
        for f in batch:
            res = result_map.get(f["timestamp"])
            if res:
                def clamp(v): return max(0, min(255, int(v)))
                raw["zone1"].append([clamp(res["zone1"][i]) for i in range(3)])
                raw["zone2"].append([clamp(res["zone2"][i]) for i in range(3)])
                raw["zone3"].append([clamp(res["zone3"][i]) for i in range(3)])
                emotions.append(str(res.get("emotion", "")))
                batch_emotions.append(str(res.get("emotion", "")))
            else:
                raw["zone1"].append([120, 60, 180])
                raw["zone2"].append([120, 60, 180])
                raw["zone3"].append([120, 60, 180])
                emotions.append("unknown")
                batch_emotions.append("unknown")

        sample = " / ".join(batch_emotions[:3])
        print(f"→ {sample}")

    # Smooth her zone ayrı ayrı
    print(f"\nHSL smoothing (σ={SMOOTH_SIGMA})…")
    smooth1 = gaussian_smooth(raw["zone1"], SMOOTH_SIGMA)
    smooth2 = gaussian_smooth(raw["zone2"], SMOOTH_SIGMA)
    smooth3 = gaussian_smooth(raw["zone3"], SMOOTH_SIGMA)

    # JSON çıktısı
    light_data = []
    for i, frame in enumerate(features):
        light_data.append({
            "timestamp": frame["timestamp"],
            "emotion":   emotions[i],
            "zone1":     smooth1[i],
            "zone2":     smooth2[i],
            "zone3":     smooth3[i],
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(light_data, f, indent=2, ensure_ascii=False)

    print(f"\nTamamlandı! {len(light_data)} kare → {OUTPUT_JSON}")
    print("Başlatmak için:  python simulator.py\n")
    print("── İlk 10 kare ──────────────────────────────────────────")
    for fr in light_data[:10]:
        print(f"  t={fr['timestamp']:5.1f}s  {fr['emotion'][:38]:<38}"
              f"  z1={tuple(fr['zone1'])}  z3={tuple(fr['zone3'])}")


if __name__ == "__main__":
    main()
