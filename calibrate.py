import librosa
import numpy as np
import os

STEMS_DIR = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\separated\htdemucs\theless"
WINDOW = 0.5

for stem in ["drums", "bass", "other", "vocals"]:
    path = os.path.join(STEMS_DIR, f"{stem}.wav")
    y, sr = librosa.load(path, sr=None, mono=True)
    ws = int(WINDOW * sr)

    rms_vals, mid_vals, bass_vals, onset_vals = [], [], [], []
    for i in range(min(300, len(y)//ws)):
        seg = y[i*ws:(i+1)*ws]
        rms = float(np.mean(librosa.feature.rms(y=seg)))
        spec = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
        rms_vals.append(rms)
        bass_vals.append(float(np.mean(spec[:15, :])))
        mid_vals.append(float(np.mean(spec[20:80, :])))
        onset = librosa.onset.onset_strength(y=seg, sr=sr)
        onset_vals.append(float(np.mean(onset)))

    print(f"\n{stem.upper()}:")
    print(f"  RMS    — min:{min(rms_vals):.4f}  mean:{np.mean(rms_vals):.4f}  max:{max(rms_vals):.4f}  p95:{np.percentile(rms_vals,95):.4f}")
    print(f"  bass   — min:{min(bass_vals):.2f}  mean:{np.mean(bass_vals):.2f}  max:{max(bass_vals):.2f}  p95:{np.percentile(bass_vals,95):.2f}")
    print(f"  mid    — min:{min(mid_vals):.2f}  mean:{np.mean(mid_vals):.2f}  max:{max(mid_vals):.2f}  p95:{np.percentile(mid_vals,95):.2f}")
    print(f"  onset  — min:{min(onset_vals):.3f}  mean:{np.mean(onset_vals):.3f}  max:{max(onset_vals):.3f}  p95:{np.percentile(onset_vals,95):.3f}")
