import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import os


#  SNR function
def compute_snr(signal, noisy_signal):
    noise = noisy_signal - signal
    return 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-10))


#  Window function
def apply_window(signal, type="hamming"):
    if type == "hamming":
        return signal * np.hamming(len(signal))
    elif type == "hanning":
        return signal * np.hanning(len(signal))
    else:
        return signal  # rectangular


if __name__ == "__main__":

    folder = "content/drive/MyDrive/audio"
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    snr_results = []

    for file in files:
        path = os.path.join(folder, file)
        signal, _ = librosa.load(path, sr=16000)

        clean = signal[:1024]

        # Add noise
        noise = np.random.normal(0, 0.01, len(clean))
        noisy = clean + noise

        plt.figure()

        for w in ["rectangular", "hamming", "hanning"]:
            # Apply window
            clean_win = apply_window(clean, w)
            noisy_win = apply_window(noisy, w)

            # SNR
            snr = compute_snr(clean_win, noisy_win)

            snr_results.append({
                "file": file,
                "window": w,
                "snr": snr
            })

            # Spectral plot
            spectrum = np.abs(np.fft.fft(noisy_win))
            plt.plot(spectrum, label=w)

        plt.title(f"Spectral Leakage - {file}")
        plt.legend()
        plt.show()

    # Create table
    df = pd.DataFrame(snr_results)
    pivot_table = df.pivot(index="file", columns="window", values="snr")

    print("\nSNR Comparison Table:\n")
    print(pivot_table)