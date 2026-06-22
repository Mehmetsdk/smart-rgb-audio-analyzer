# Smart RGB Audio Analyzer 🎵💡

> **🇬🇧 For English → [README.md](README.md)**

Yapay zeka destekli bir müzik-ışık sistemi. Enstrümanları sabit renklere
eşlemek yerine, bir yapay zeka **şarkının duygusal akışını dinler** ve bir
insanın o anda *ne hissedeceğine* göre renkler üretir — gerçek zamanlı bir
simülatör de bu duyguları canlı bir ışık alanı olarak ekrana yansıtır.

Bilgisayar Mühendisliği ve Elektrik-Elektronik Mühendisliği ortak projesi olarak
geliştirildi. Uzun vadeli hedef: fiziksel akıllı aydınlatma (ESP32 / WS2812B LED
şeritleri) sürmek.

---

## ✨ Temel Fikir

Çoğu ses görselleştiricisi mekaniktir: bas → kırmızı, tiz → mavi. Bu proje farklı
bir yol izler.

1. **librosa** parçadan akustik özellikler çıkarır (enerji, vuruş/onset, spektral
   parlaklık, baskın nota) — saniyede bir.
2. **Claude** ([Anthropic API](https://www.anthropic.com)) bu özellikleri
   *şarkının duygusal bağlamıyla birlikte* alır ve her saniye için şunları döndürür:
   - kısa bir **duygu etiketi** (örn. *"euphoric melancholy"*, *"jealous ache"*)
   - **üç bağımsız RGB renk** — ritim, melodi ve vokal katmanları için ayrı ayrı
3. Renkler **HSL uzayında yumuşatılır**; geçişler yavaş bir gün batımı gibi olur,
   asla sert bir flaş gibi değil.
4. **Gerçek zamanlı gradient simülatör** sonucu, her biri bir frekans bandına bağlı,
   müzikle senkron nabız atan 12 parlayan ışık olarak gösterir.

---

## 🚀 Özellikler

- **Önce duygu, sonra renk** — yapay zeka frekansı değil, hissi yorumlar.
- **Üç bağımsız bölge** — ritim, melodi ve vokal her biri kendi renk hikâyesini anlatır.
- **Organik gradient alanı** — yavaşça dönen ayçiçeği deseninde 12 frekans bantlı ışık,
  *screen* modunda harmanlanır; renkler canlı ve ayrı kalır.
- **Frekansa tepkili** — bas, orta ve tiz tek bir global parlaklık yerine bağımsız patlar.
- **Spektrum hue yayılımı** — duygusal ton merkezde kalır, renkler tüm tekere yayılarak
  daha zengin bir görünüm verir (`HUE_SPREAD`).
- **İpeksi geçişler** — render anında HSL Gaussian yumuşatma + `smootherstep` interpolasyon.
- **Sıkı ses senkronu** — ses önceden yüklenir, çalma görsel saatiyle ardışık başlar (kayma yok).
- **~50 FPS** — `blit` ile statik arka plan önbelleğe alınır, yalnızca değişen çizilir.
- **Dayanıklı analiz** — küçük batch'ler, JSON kurtarma ayrıştırıcı, retry ve eksik
  kareler için komşudan doldurma.

---

## 📂 Proje Yapısı

```text
emotion_color_analyzer.py   # ⭐ Ana hat: özellik → Claude → duygusal renk → JSON
simulator.py                # ⭐ Gerçek zamanlı gradient ışık alanı (JSON'u okur)
light_script.json           # Üretilen zaman senkronlu renk zaman çizelgesi (AI çıktısı)
requirements.txt            # Python bağımlılıkları

stem_processor.py           # Deneysel: Demucs ile stem ayırma (yalnızca referans)
calibrate.py                # Geliştirici aracı: stem başına özellik aralıklarını ölçer
check_json.py               # Geliştirici aracı: üretilen JSON'u inceler
```

> `audio_processor.py` ve `visualization.py` daha eski bir enstrüman→sabit-renk
> yaklaşımına aittir, yalnızca geçmiş için tutulmaktadır.

---

## 🔧 Gereksinimler

- **Python 3.12** (3.14 henüz ses kütüphaneleriyle uyumlu değil)
- Bir **Anthropic API anahtarı** → [console.anthropic.com](https://console.anthropic.com)
- **FFmpeg** (yalnızca MP3'ü WAV'a çevirmen gerekirse)

Python paketleri (`requirements.txt`):
`anthropic`, `librosa`, `numpy`, `matplotlib`, `sounddevice`, `soundfile`

---

## 📦 Kurulum

```bash
# 1. Python 3.12 sanal ortamı oluştur ve etkinleştir
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. API anahtarını ver
#   Windows (PowerShell):
$env:ANTHROPIC_API_KEY = "sk-ant-..."
#   macOS/Linux:
#   export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## ▶️ Kullanım

```bash
# Adım 1 — şarkıyı analiz et (Claude'u çağırır, light_script.json yazar)
python emotion_color_analyzer.py

# Adım 2 — ışık gösterisini müzikle senkron oynat
python simulator.py
```

**Farklı bir şarkı** analiz etmek mi istiyorsun? `.wav` dosyanı koy ve
`emotion_color_analyzer.py` başındaki şarkı bloğunu düzenle:

```python
SONG_TITLE  = "The Less I Know The Better"
SONG_ARTIST = "Tame Impala"
SONG_INFO   = "Bittersweet psychedelic pop. Emotional arc: longing → euphoria → nostalgia."
```

---

## ⚙️ İnce Ayar

**`emotion_color_analyzer.py`**

| Sabit | Varsayılan | Görevi |
|-------|-----------|--------|
| `WINDOW_SEC` | `1.0` | Analiz penceresi başına saniye (bir renk keyframe'i) |
| `BATCH_SEC`  | `10.0` | İstek başına Claude'a gönderilen pencere saniyesi |
| `SIGMA`      | `3.0` | HSL yumuşatma genişliği (yüksek = daha yumuşak geçiş) |
| `MODEL`      | `claude-haiku-4-5` | Analiz için kullanılan Claude modeli |

**`simulator.py`**

| Sabit | Varsayılan | Görevi |
|-------|-----------|--------|
| `N_BANDS`      | `12` | Frekans bantlı ışık sayısı |
| `HUE_SPREAD`   | `0.62` | Spektrum boyunca renk çeşitliliği (0 = sadece duygu tonu) |
| `AUDIO_OFFSET` | `0.0` | Senkron ince ayarı (sn). Görsel öndeyse +, gerideyse − |
| `TARGET_FPS`   | `50` | Kare hızı sınırı |

---

## 🗺️ Yol Haritası

- [x] AI duygusal renk analizi (Claude)
- [x] Bağımsız ritim / melodi / vokal bölgeleri
- [x] Yumuşak HSL geçişleri
- [x] Frekans bantlı ışıklarla gerçek zamanlı gradient simülatör
- [ ] Gerçek zamanlı analiz (ön işleme adımı olmadan)
- [ ] Fiziksel donanım: ESP32 + WS2812B LED şeritleri
- [ ] Mikrofon / canlı giriş modu

---

## 🙏 Teşekkürler

- Duygusal analiz [Claude](https://www.anthropic.com) (Anthropic API) ile
- Ses özellikleri [librosa](https://librosa.org) ile
- Referans parça: *"The Less I Know The Better"* — Tame Impala
