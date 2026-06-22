import json
import time
import math
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

AUDIO_PATH  = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.wav"
JSON_PATH   = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\light_script.json"
FEATURE_RES = 0.1
ZONE_NAMES  = ["RİTİM", "MELODİ", "VOKAL"]

# ── Gradient render çözünürlüğü (bilinear ile yumuşatılır) ───────────────────
H, W = 160, 240
_yy, _xx = np.mgrid[0:H, 0:W].astype(float)

# Frekans spektrumu kaç ışığa bölünsün (düşük→yüksek frekans)
N_BANDS = 12

# Spektrum boyunca renk çeşitliliği — 0=sadece duygu tonu, yüksek=gökkuşağı
HUE_SPREAD = 0.62

# Ayçiçeği (golden-angle) deseni: ışıkları ekrana dengeli dağıt
GOLDEN = 2.399963   # altın açı (radyan)
_blob_layout = []
for j in range(N_BANDS):
    frac = j / max(1, N_BANDS - 1)
    ang  = j * GOLDEN
    rad  = 0.42 * math.sqrt((j + 0.5) / N_BANDS)   # merkezden dışa spiral
    bx   = 0.5 + rad * math.cos(ang)
    by   = 0.5 + rad * math.sin(ang)
    _blob_layout.append((bx, by, ang, frac))


# ── Ses ──────────────────────────────────────────────────────────────────────

# Ses çıkış gecikmesi telafisi (saniye). Görsel sesin önündeyse artır,
# gerisindeyse azalt (negatif olabilir).
AUDIO_OFFSET = 0.0


def load_playback(path):
    """Çalma için sesi önceden belleğe yükle — senkron için kritik."""
    y, sr = librosa.load(path, sr=44100, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.T, sr     # (örnek, 2)


def precompute_features(audio_path, resolution=0.1):
    """Frekans spektrumunu N_BANDS banda böl, her bant için 0.1s seviye sinyali."""
    print(f"Ses özellikleri hesaplanıyor ({N_BANDS} frekans bandı)…")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    ws    = int(resolution * sr)

    # Tüm şarkı için tek mel spektrogram (hızlı, vektörize)
    n_mels = N_BANDS * 4
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=ws)

    def process(raw):
        mx = float(np.max(raw)) or 1.0
        out, val = [], 0.0
        for r in raw:
            n = float(r) / mx
            val = max(n, val * 0.80)      # beat'te hızlı yüksel, yumuşak söner
            out.append(val)
        return out

    # Her bandı 4 mel'lik gruptan oluştur, ayrı ayrı normalize+decay
    band_levels = []
    for j in range(N_BANDS):
        raw = S[j*4:(j+1)*4].mean(axis=0)
        band_levels.append(process(raw))

    total = len(band_levels[0])
    feats = [{"t": round(i*resolution, 3),
              "bands": [band_levels[j][i] for j in range(N_BANDS)]}
             for i in range(total)]
    print(f"  {len(feats)} kare hazır.")
    return feats


# ── Renk yardımcıları ─────────────────────────────────────────────────────────

def smootherstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp_hsl(c1, c2, t):
    t = smootherstep(t)
    h1, l1, s1 = colorsys.rgb_to_hls(*[v/255 for v in c1])
    h2, l2, s2 = colorsys.rgb_to_hls(*[v/255 for v in c2])
    dh = h2 - h1
    if dh >  0.5: dh -= 1
    if dh < -0.5: dh += 1
    h = (h1 + dh*t) % 1.0
    l = l1 + (l2-l1)*t
    s = s1 + (s2-s1)*t
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return [r*255, g*255, b*255]


def vivid(rgb):
    """Rengi canlandır — doygunluğu yükselt, parlaklığı vivid aralığa çek."""
    h, l, s = colorsys.rgb_to_hls(*[v/255 for v in rgb])
    s = min(1.0, s * 1.35 + 0.15)
    l = min(0.62, max(0.42, l))      # mat/koyu renkleri parlat
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([r, g, b]) * 255.0


# ── Gradient alanı render ─────────────────────────────────────────────────────

def spectrum_color(frac, z1, z2, z3):
    """
    Duygusal ton (z1→z2→z3) merkez alınır, ama spektrum boyunca hue geniş bir
    yelpazeye yayılır → çok daha fazla renk çeşitliliği (kırmızı/turuncu/yeşil/cyan).
    """
    if frac < 0.5:
        base = lerp_hsl(z1, z2, frac * 2)
    else:
        base = lerp_hsl(z2, z3, (frac - 0.5) * 2)

    h, l, s = colorsys.rgb_to_hls(*[v/255 for v in base])
    h = (h + (frac - 0.5) * HUE_SPREAD) % 1.0      # spektrum boyunca hue kaydır
    s = min(1.0, s * 1.05 + 0.05)                  # biraz daha doygun
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return [r*255, g*255, b*255]


def render_field(zone_colors, t, feat):
    """
    N_BANDS ışık — her biri bir frekans bandı. Düşük→yüksek frekans boyunca
    renk z1→z2→z3 geçer. Işıklar ayçiçeği deseninde yavaşça döner, her biri
    kendi bandının enerjisiyle bağımsız patlar. Screen blending → canlı.
    """
    z1, z2, z3 = zone_colors
    bands = feat["bands"]

    spin = t * 0.06   # tüm desen yavaşça döner
    inv  = np.ones((H, W, 3))

    for j, (bx, by, ang, frac) in enumerate(_blob_layout):
        lv  = bands[j]
        col = vivid(spectrum_color(frac, z1, z2, z3)) / 255.0

        # Desen merkez etrafında döner + her ışık hafifçe nefes alır
        rad = math.hypot(bx - 0.5, by - 0.5)
        a   = ang + spin
        cx  = (0.5 + rad * math.cos(a)) * W
        cy  = (0.5 + rad * math.sin(a)) * H

        sigma  = (0.11 + 0.09 * lv) * W
        bright = 0.18 + 1.25 * lv

        d2 = (_xx - cx)**2 + (_yy - cy)**2
        w  = np.exp(-d2 / (2*sigma*sigma))

        layer = np.clip((w * bright)[..., None] * col, 0, 1)
        inv  *= (1.0 - layer)            # screen blend

    img = 1.0 - inv
    img += 0.03                          # hafif ambient taban
    return np.clip(img, 0, 1)


# ── Bilgi paneli ──────────────────────────────────────────────────────────────

def rgb_hex(rgb):
    r, g, b = (max(0, min(255, int(v))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def make_info_panel(fig, light_data, info_x=0.67):
    artist = light_data[0].get("artist", "") if light_data else ""
    title  = light_data[0].get("title",  "") if light_data else ""

    fig.text(info_x, 0.90, artist or "Smart RGB",      fontsize=13,
             color="#ffffff", fontfamily="monospace", fontweight="bold")
    fig.text(info_x, 0.84, title  or "Audio Analyzer", fontsize=9,
             color="#aaaaaa", fontfamily="monospace")

    labels, hexes = [], []
    for i, name in enumerate(ZONE_NAMES):
        y0 = 0.70 - i*0.16
        fig.text(info_x, y0+0.04, name, fontsize=8, color="#666666",
                 fontfamily="monospace")
        lbl = fig.text(info_x, y0,      "─",       fontsize=10,
                       color="#ffffff", fontfamily="monospace")
        hx  = fig.text(info_x, y0-0.04, "#??????", fontsize=9,
                       color="#888888", fontfamily="monospace")
        labels.append(lbl); hexes.append(hx)

    time_t    = fig.text(0.33, 0.03,  "", ha="center", fontsize=10,
                         color="#dddddd", fontfamily="monospace")
    emotion_t = fig.text(0.33, 0.005, "", ha="center", fontsize=9,
                         color="#bb99dd", fontfamily="monospace", style="italic")
    return labels, hexes, time_t, emotion_t


def update_info(labels, hexes, zones):
    for i, rgb in enumerate(zones):
        labels[i].set_text(f"RGB  ({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})")
        h = rgb_hex(rgb)
        hexes[i].set_text(h)
        hexes[i].set_color(h)


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            light_data = json.load(f)
    except FileNotFoundError:
        print("light_script.json yok. Önce emotion_color_analyzer.py çalıştırın.")
        return

    audio_features = precompute_features(AUDIO_PATH, FEATURE_RES) if LIBROSA_AVAILABLE else []

    # Çalma sesini ÖNCEDEN yükle (senkron için kritik — loop başlarken hazır olmalı)
    playback = None
    if AUDIO_AVAILABLE and LIBROSA_AVAILABLE:
        print("Ses çalmaya hazırlanıyor…")
        try:
            playback = load_playback(AUDIO_PATH)
        except Exception as e:
            print(f"Ses yüklenemedi: {e}")

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#000000")
    fig.canvas.manager.set_window_title("Smart RGB Audio Analyzer — Simülatör")

    ax = fig.add_axes([0.01, 0.06, 0.62, 0.90])
    ax.axis("off")
    im = ax.imshow(np.zeros((H, W, 3)), aspect="auto",
                   origin="lower", interpolation="bilinear")

    labels, hexes, time_t, emotion_t = make_info_panel(fig, light_data)

    plt.ion()
    plt.show()

    kf_dt = (light_data[1]["timestamp"] - light_data[0]["timestamp"]
             if len(light_data) >= 2 else 1.0)

    TARGET_FPS  = 50
    frame_dt    = 1.0 / TARGET_FPS

    # Sabit şarkı başlığı (her frame yeniden oluşturmaya gerek yok)
    a  = light_data[0].get("artist", ""); ti = light_data[0].get("title", "")
    song = f"{a} — {ti}" if a and ti else "Smart RGB Audio Analyzer"

    # İlk frame'i ses başlamadan çiz (ağır ilk render'ı warm-up et)
    feat0 = audio_features[0] if audio_features else {"bands": [0.3]*N_BANDS}
    img0  = render_field([light_data[0]["zone1"], light_data[0]["zone2"],
                          light_data[0]["zone3"]], 0.0, feat0)
    im.set_data(img0)

    # ── Blit kurulumu: dinamik öğeleri işaretle, statik arka planı cache'le ──
    dyn_texts = [time_t, emotion_t] + labels + hexes
    for art in [im] + dyn_texts:
        art.set_animated(True)
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    # Ses ve saat tam ardışık başlar → görsel ile müzik aynı t=0'dan
    if playback is not None:
        sd.play(playback[0], samplerate=playback[1])
    start_time = time.time() + AUDIO_OFFSET

    while plt.fignum_exists(fig.number):
        elapsed = time.time() - start_time
        if elapsed > light_data[-1]["timestamp"] + kf_dt:
            break

        # Duygusal baz renkler (Claude, keyframe interpolation)
        raw_idx = elapsed / kf_dt
        idx_lo  = max(0, min(int(raw_idx), len(light_data)-2))
        idx_hi  = idx_lo + 1
        t_kf    = max(0.0, min(1.0, raw_idx - idx_lo))
        lo, hi  = light_data[idx_lo], light_data[idx_hi]

        z1 = lerp_hsl(lo["zone1"], hi["zone1"], t_kf)
        z2 = lerp_hsl(lo["zone2"], hi["zone2"], t_kf)
        z3 = lerp_hsl(lo["zone3"], hi["zone3"], t_kf)

        # Audio özellikleri — 0.1s pencere merkezine hizala (round)
        if audio_features:
            fi   = min(max(0, round(elapsed / FEATURE_RES)), len(audio_features)-1)
            feat = audio_features[fi]
        else:
            feat = {"bands": [0.3]*N_BANDS}

        img = render_field([z1, z2, z3], elapsed, feat)
        im.set_data(img)
        update_info(labels, hexes, [z1, z2, z3])

        mins, secs = int(elapsed // 60), elapsed % 60
        time_t.set_text(f"♪  {mins}:{secs:05.2f}  |  {song}")
        emotion = lo.get("emotion", "")
        if emotion:
            emotion_t.set_text(f'"{emotion}"')

        # ── Blit: arka planı geri yükle, sadece değişenleri çiz ──
        try:
            fig.canvas.restore_region(bg)
            ax.draw_artist(im)
            for art in dyn_texts:
                fig.draw_artist(art)
            fig.canvas.blit(fig.bbox)
        except Exception:
            # Pencere yeniden boyutlandı → tam çiz, arka planı yenile
            fig.canvas.draw()
            bg = fig.canvas.copy_from_bbox(fig.bbox)
        fig.canvas.flush_events()

        # FPS sınırla (render hızlıysa bekle, yavaşsa frame atla — senkron korunur)
        nxt = start_time + (int(elapsed/frame_dt)+1) * frame_dt
        slp = nxt - time.time()
        if slp > 0:
            time.sleep(slp)

    if AUDIO_AVAILABLE:
        sd.stop()
    plt.ioff()
    plt.close("all")
    print("Simülasyon tamamlandı.")


if __name__ == "__main__":
    main()
