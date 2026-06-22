import subprocess
import sys
import os
import json
import numpy as np
import librosa
import soundfile as sf

AUDIO_PATH = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.mp3"
PROJECT_DIR = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje"
STEMS_DIR = os.path.join(PROJECT_DIR, "separated")
OUTPUT_JSON = os.path.join(PROJECT_DIR, "light_script.json")

WINDOW_DURATION = 0.5
SMOOTHING = 0.3


# ─── Duygusal renk paleti (The Less I Know The Better için özelleştirildi) ────

def drums_to_color(onset_strength, rms):
    """Davul vuruşları: güçlü vuruşlarda sıcak turuncu-kırmızı flaş, arada karanlık."""
    # p95 onset = 1.658 → /1.5 ile 0-1 arası normalize
    intensity = np.clip(onset_strength / 1.5, 0, 1)
    if intensity > 0.6:
        # Güçlü davul vuruşu — ateşli turuncu
        r = int(255 * intensity)
        g = int(90 * intensity)
        b = int(20 * intensity)
    elif intensity > 0.25:
        # Orta şiddet — amber
        r = int(200 * intensity)
        g = int(60 * intensity)
        b = int(40 * intensity)
    else:
        # Sessizlik — koyu, neredeyse siyah
        r, g, b = int(15 * rms * 10), 0, int(20 * rms * 10)
    return [r, g, b]


def bass_to_color(bass_energy, rms):
    """Bass: derin mor ve indigo. Vuruşlarda parlayan koyu kırmızı-mor."""
    # p95 bass = 119.50
    intensity = np.clip(bass_energy / 120.0, 0, 1)
    if intensity > 0.6:
        r = int(140 * intensity)
        g = 0
        b = int(220 * intensity)
    elif intensity > 0.3:
        r = int(80 * intensity)
        g = 0
        b = int(160 * intensity)
    else:
        r = int(30 * intensity)
        g = 0
        b = int(60 * intensity)
    return [r, g, b]


def guitar_to_color(mid_energy, spectral_centroid, rms):
    """
    Gitar / synth: o ikonik tatlı-hüzünlü his.
    Tiz notalar → pembe-lavanta. Ağır riff → sıcak violet.
    """
    # p95 mid = 6.56 → /6.0 ile 0-1 arası normalize
    intensity = np.clip(mid_energy / 6.0, 0, 1)
    # Spectral centroid ne kadar yüksekse o kadar "tiz" → pembeye kayar
    tiz_oran = np.clip(spectral_centroid / 4000.0, 0, 1)

    r = int((160 + 80 * tiz_oran) * intensity)
    g = int(30 * intensity)
    b = int((200 + 55 * (1 - tiz_oran)) * intensity)
    return [r, g, b]


def vocals_to_color(vocal_energy, pitch_confidence, rms):
    """
    Kevin'in vokali: umutsuz ama heyecanlı o his.
    Yüksek nota → beyaza yakın lavanta. Sessiz → soluk pembe-mor.
    """
    # p95 vocal mid = 7.79 → /8.0 ile 0-1 arası normalize
    intensity = np.clip(vocal_energy / 8.0, 0, 1)
    confidence = np.clip(pitch_confidence, 0, 1)

    if intensity > 0.5 and confidence > 0.4:
        # Vokal güçlü ve net → parlak lavanta-beyaz
        r = int(220 * intensity)
        g = int(150 * intensity * confidence)
        b = int(255 * intensity)
    elif intensity > 0.2:
        # Orta güç → pembe-mor
        r = int(180 * intensity)
        g = int(60 * intensity)
        b = int(200 * intensity)
    else:
        # Fısıltı ya da sessizlik → çok soluk
        r = int(40 * intensity)
        g = 0
        b = int(60 * intensity)
    return [r, g, b]


def smooth(current, previous, factor=SMOOTHING):
    return [int(factor * c + (1 - factor) * p) for c, p in zip(current, previous)]


# ─── Stem analizi ─────────────────────────────────────────────────────────────

def analyze_stem(stem_path, stem_type):
    """Bir stem wav dosyasını yükle ve her pencere için renk üret."""
    y, sr = librosa.load(stem_path, sr=None, mono=True)
    window_size = int(WINDOW_DURATION * sr)
    total_windows = len(y) // window_size

    colors = []
    prev = [0, 0, 0]

    for i in range(total_windows):
        segment = y[i * window_size: (i + 1) * window_size]

        rms = float(np.mean(librosa.feature.rms(y=segment)))
        spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)

        if stem_type == "drums":
            onset = librosa.onset.onset_strength(y=segment, sr=sr)
            onset_val = float(np.mean(onset))
            color = drums_to_color(onset_val, rms)

        elif stem_type == "bass":
            bass_energy = float(np.mean(spec[:15, :]))
            color = bass_to_color(bass_energy, rms)

        elif stem_type == "other":
            mid_energy = float(np.mean(spec[20:80, :]))
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            centroid_val = float(np.mean(centroid))
            color = guitar_to_color(mid_energy, centroid_val, rms)

        elif stem_type == "vocals":
            vocal_energy = float(np.mean(spec[30:100, :]))
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment, fmin=80, fmax=1100,
                sr=sr, fill_na=0.0
            )
            pitch_conf = float(np.mean(voiced_probs)) if voiced_probs is not None else 0.0
            color = vocals_to_color(vocal_energy, pitch_conf, rms)

        else:
            color = [0, 0, 0]

        color = smooth(color, prev)
        colors.append(color)
        prev = color

    return colors


# ─── Ana akış ─────────────────────────────────────────────────────────────────

def run_demucs():
    """
    Demucs ile şarkıyı 4 sese ayır.
    torchaudio/torchcodec'i atlatmak için librosa ile yükleyip
    doğrudan demucs modeline tensor veriyoruz.
    """
    import torch
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    os.makedirs(STEMS_DIR, exist_ok=True)
    wav_path = AUDIO_PATH.replace(".mp3", ".wav")

    # WAV yoksa mevcut MP3'ten oluştur (ffmpeg subprocess ile)
    if not os.path.exists(wav_path):
        ffmpeg_candidates = [
            r"C:\Users\Mehmet Sadık Gürler\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
            "ffmpeg",
        ]
        for ffmpeg in ffmpeg_candidates:
            r = subprocess.run(
                [ffmpeg, "-i", AUDIO_PATH, "-ar", "44100", wav_path, "-y"],
                capture_output=True
            )
            if r.returncode == 0:
                break
        else:
            print("FFmpeg bulunamadı, MP3'ü WAV'a çevirme atlandı — WAV dosyası aranıyor.")

    src_path = wav_path if os.path.exists(wav_path) else AUDIO_PATH
    print(f"Ses yükleniyor: {os.path.basename(src_path)}")
    y, sr = librosa.load(src_path, sr=44100, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono → stereo

    # (1, 2, T) batch tensor
    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)

    print("Demucs modeli yükleniyor (htdemucs)...")
    model = get_model("htdemucs")
    model.eval()

    print("Ses ayrıştırılıyor (bu 2-5 dakika sürebilir)...")
    with torch.no_grad():
        sources = apply_model(
            model, wav_tensor,
            shifts=1, split=True, overlap=0.25, progress=True
        )

    sources = sources[0]  # batch boyutunu kaldır → (4, 2, T)
    out_dir = os.path.join(STEMS_DIR, "htdemucs", "theless")
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(model.sources):
        stem_np = sources[i].numpy()  # (2, T)
        out_path = os.path.join(out_dir, f"{name}.wav")
        sf.write(out_path, stem_np.T, samplerate=44100)
        print(f"  Kaydedildi: {name}.wav")

    print("Ayrıştırma tamamlandı.")


def find_stems():
    """Ayrıştırılmış stem dosyalarını bul."""
    for root, dirs, files in os.walk(STEMS_DIR):
        wav_files = [f for f in files if f.endswith(".wav")]
        if len(wav_files) >= 2:
            stem_map = {}
            for f in wav_files:
                name = os.path.splitext(f)[0].lower()
                stem_map[name] = os.path.join(root, f)
            return stem_map
    return {}


def main():
    # 1. Demucs çalıştır (zaten varsa atla)
    stem_map = find_stems()
    if len(stem_map) < 2:
        run_demucs()
        stem_map = find_stems()

    print("Bulunan stemler:", list(stem_map.keys()))

    # htdemucs 4 stem çıkarır: drums, bass, other, vocals
    # two-stems sadece vocals + no_vocals çıkarır (fallback)
    stem_types = {
        "drums": "drums",
        "bass": "bass",
        "other": "other",
        "guitar": "other",
        "vocals": "vocals",
        "no_vocals": "other",
    }

    # 2. Her stem için renk hesapla
    all_colors = {}
    for stem_name, stem_path in stem_map.items():
        key = stem_name.replace("theless", "").strip("_- ") if "theless" in stem_name else stem_name
        stype = stem_types.get(key, "other")
        print(f"Analiz ediliyor: {key} ({stype})...")
        all_colors[key] = analyze_stem(stem_path, stype)

    # 3. JSON çıktısı üret
    # Zone1 = Bass + Drums karışımı, Zone2 = Guitar/Other, Zone3 = Vocals
    drums_colors = all_colors.get("drums", all_colors.get(list(all_colors.keys())[0]))
    bass_colors  = all_colors.get("bass",  drums_colors)
    other_colors = all_colors.get("other", all_colors.get("guitar", all_colors.get("no_vocals", drums_colors)))
    vocal_colors = all_colors.get("vocals", all_colors.get(list(all_colors.keys())[-1]))

    min_len = min(len(drums_colors), len(bass_colors), len(other_colors), len(vocal_colors))

    light_data = []
    for i in range(min_len):
        # Zone1: Davul + Bass ortalaması
        z1 = [
            (drums_colors[i][c] + bass_colors[i][c]) // 2
            for c in range(3)
        ]
        z2 = other_colors[i]
        z3 = vocal_colors[i]

        light_data.append({
            "timestamp": round(i * WINDOW_DURATION, 2),
            "zone1": z1,   # Ritim bölgesi (davul + bass)
            "zone2": z2,   # Melodi bölgesi (gitar)
            "zone3": z3,   # Vokal bölgesi
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(light_data, f, indent=2)

    print(f"\nTamamlandi! {len(light_data)} kare --> {OUTPUT_JSON}")
    print("Simülatörü başlatmak için: python simulator.py")


if __name__ == "__main__":
    main()
