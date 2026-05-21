import librosa
import numpy as np
import json

# 1. Load the audio file
audio_path = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.mp3"
y, sr = librosa.load(audio_path)

# Define 0.5-second windows
window_duration = 0.5  
window_size = int(window_duration * sr)
total_windows = len(y) // window_size

# --- DESIGN PARAMETERS ---
# Lower values (e.g., 0.15) = Smoother, slower transitions (Fade effect)
# Higher values (e.g., 0.60) = Faster, punchier transitions
SMOOTHING_FACTOR = 0.25  

light_data = []

# Keep track of the color from the previous frame to calculate the blend
previous_rgb = [0, 0, 0]

print("Analyzing audio and generating smooth color scripts...")

# 2. Process the audio window by window
for i in range(total_windows):
    start = i * window_size
    end = start + window_size
    audio_slice = y[start:end]
    
    current_time = i * window_duration
    
    # Calculate Energy & Bass
    rms = librosa.feature.rms(y=audio_slice)
    average_energy = np.mean(rms) if rms.size > 0 else 0
    
    spec = librosa.feature.melspectrogram(y=audio_slice, sr=sr, n_mels=128)
    bass_energy = np.mean(spec[:10, :]) if spec.size > 0 else 0

    # 3. Step A: Define the "Target" Base Color (Same rules as before)
    if average_energy > 0.08:  
        target_rgb = [148, 0, 211]    # Chorus Purple
        status = "Chorus (Purple)"
    elif bass_energy > 0.15:  
        target_rgb = [255, 20, 147]    # Bass Pink
        status = "Bass Hit (Pink)"
    else:  
        target_rgb = [64, 224, 208]    # Chill Turquoise
        status = "Chill Flow (Turquoise)"
        
    # 4. Step B: Dynamic Brightness Modulation (Pulsing / Yanıp Sönme Effect)
    # We map the energy to a scale factor so the light dims/brightens with the beat
    # Using np.clip to make sure it doesn't exceed bounds
    brightness_multiplier = np.clip(average_energy * 10, 0.2, 1.0) 
    target_rgb = [int(color * brightness_multiplier) for color in target_rgb]

    # 5. Step C: Mathematical Blending (Fade / Exponential Smoothing)
    # New Color = (Alpha * Target Color) + ((1 - Alpha) * Previous Color)
    if i == 0:
        smoothed_rgb = target_rgb
    else:
        smoothed_rgb = [
            int(SMOOTHING_FACTOR * target_rgb[j] + (1 - SMOOTHING_FACTOR) * previous_rgb[j])
            for j in range(3)
        ]
        
    # Update previous color for the next iteration
    previous_rgb = smoothed_rgb

    # Append the beautifully smoothed data
    light_data.append({
        "timestamp": current_time,
        "rgb": smoothed_rgb,
        "status": status
    })

# 4. Save to JSON
with open("light_script.json", "w") as f:
    json.dump(light_data, f, indent=4)

print("Analysis completed successfully! Check your new 'light_script.json' for smooth values.")