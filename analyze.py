import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the recorded audio
y, sr = librosa.load('Noise1.wav', sr=16000)

# Create subplot figure
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title('Noise one')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude')

# Plot spectrogram
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title('Spectrogram: "Noise one"')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()