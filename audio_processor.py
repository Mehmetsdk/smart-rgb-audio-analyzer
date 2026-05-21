import librosa
import numpy as np
import json

# 1. Load the audio file
audio_path = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.mp3"
y, sr = librosa.load(audio_path)

# Define 0.5-second windows for real-time simulation
window_duration = 0.5  # seconds
window_size = int(window_duration * sr)
total_windows = len(y) // window_size

light_data = []

print("Analyzing audio and generating color scripts...")

# 2. Process the audio window by window
for i in range(total_windows):
    start = i * window_size
    end = start + window_size
    audio_slice = y[start:end]
    
    # Track the current timestamp in the song
    current_time = i * window_duration
    
    # a) Calculate Root-Mean-Square (RMS) Energy for overall loudness
    rms = librosa.feature.rms(y=audio_slice)
    average_energy = np.mean(rms) if rms.size > 0 else 0
    
    # b) Filter Bass Frequencies (First 10 bins of the Mel-Spectrogram)
    spec = librosa.feature.melspectrogram(y=audio_slice, sr=sr, n_mels=128)
    bass_energy = np.mean(spec[:10, :]) if spec.size > 0 else 0

    # 3. Rule-Based Color Mapping (Tame Impala Palette)
    if average_energy > 0.08:  
        # High Energy Peak (Chorus): VIBRANT PURPLE
        rgb = [148, 0, 211]
        status = "Chorus (Purple)"
    elif bass_energy > 0.15:  
        # Strong Bass Groove / Kick: NEON PINK
        rgb = [255, 20, 147]
        status = "Bass Hit (Pink)"
    else:  
        # Melodic / Dreamy sections (Verse): RETRO TURQUOISE
        rgb = [64, 224, 208]
        status = "Chill Flow (Turquoise)"
        
    # Append the processed slice data to the list
    light_data.append({
        "timestamp": current_time,
        "rgb": rgb,
        "status": status
    })

# 4. Save the final dataset as a JSON file for the hardware controller
with open("light_script.json", "w") as f:
    json.dump(light_data, f, indent=4)

print("Analysis completed successfully! 'light_script.json' has been created.")