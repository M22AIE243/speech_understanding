import numpy as np
import librosa
import matplotlib.pyplot as plt
import os


#  Cepstrum
def cepstrum(signal):
    spectrum = np.fft.fft(signal)
    log_spec = np.log(np.abs(spectrum) + 1e-10)
    return np.fft.ifft(log_spec).real


#  Voiced / Unvoiced Detection
def detect_voiced(frame, threshold=0.1):
    cep = cepstrum(frame)

    high_q = cep[20:100]

    if np.max(high_q) > threshold:
        return 1
    else:
        return 0


#  Framing (reuse logic)
def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
    frame_len = int(frame_size * sr)
    frame_step = int(frame_stride * sr)

    frames = []
    for i in range(0, len(signal) - frame_len, frame_step):
        frames.append(signal[i:i + frame_len])

    return np.array(frames)


#  Main Execution
if __name__ == "__main__":

    folder = "content/drive/MyDrive/audio/"
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    for file in files:
        path = os.path.join(folder, file)

        signal, sr = librosa.load(path, sr=16000)

        # Create frames
        frames = framing(signal, sr)

        # Detect voiced/unvoiced
        voiced_flags = [detect_voiced(f) for f in frames]

        # Time axis
        time_signal = np.linspace(0, len(signal) / sr, len(signal))
        frame_time = np.linspace(0, len(signal) / sr, len(voiced_flags))

        # Plot
        plt.figure(figsize=(12, 4))

        plt.plot(time_signal, signal, alpha=0.6, label="Signal")
        plt.step(frame_time, voiced_flags, where='post', color='red', label="Voiced")

        plt.title(f"Voiced/Unvoiced Detection - {file}")
        plt.xlabel("Time (seconds)")
        plt.legend()
        plt.show()