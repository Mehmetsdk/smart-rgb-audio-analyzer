import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the entire audio file
audio_path = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.mp3"
y, sr = librosa.load(audio_path)

# 2. Compute the Mel-Spectrogram
# n_mels=128 divides the frequency spectrum into 128 vertical bins
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# 3. Convert power spectrogram to decibel (dB) units (logarithmic scale)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 4. Plot the full song analysis
plt.figure(figsize=(15, 5))
librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='coolwarm')

plt.colorbar(format='%+2.0f dB')
plt.title('Tame Impala - The Less I Know The Better (Full Audio Analysis)')
plt.tight_layout()
plt.show()