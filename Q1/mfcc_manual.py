#pip install numpy librosa scipy
import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.fftpack import dct
import os


#  1. Pre-emphasis
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


#  2. Framing
def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
    frame_len = int(frame_size * sr)  # samples per frame
    frame_step = int(frame_stride * sr)  # step size

    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_len) / frame_step)) + 1

    pad_len = num_frames * frame_step + frame_len
    pad_signal = np.append(signal, np.zeros(pad_len - signal_length))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T

    frames = pad_signal[indices.astype(np.int32)]
    return frames


#  3. Windowing (Hamming applied later directly)


#  4. FFT + Power Spectrum
def power_spectrum(frames, NFFT=512):
    fft = np.fft.rfft(frames, NFFT)
    power = (1.0 / NFFT) * (np.abs(fft) ** 2)
    return power


#  5. Mel Filterbank
def mel_filterbank(pow_frames, sr, nfilt=26, NFFT=512):
    # Convert Hz → Mel
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    # Convert Mel → Hz
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))

    for m in range(1, nfilt + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)

        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)

    filter_banks = np.dot(pow_frames, fbank.T)

    # Avoid log(0)
    filter_banks = np.where(filter_banks == 0, 1e-10, filter_banks)

    return filter_banks


#  6. Full Pipeline Function
def process_audio(file_path):
    signal, sr = librosa.load(file_path, sr=16000)

    # Step 1: Pre-emphasis
    emphasized = pre_emphasis(signal)

    # Step 2: Framing
    frames = framing(emphasized, sr)

    # Step 3: Windowing
    windowed = frames * np.hamming(frames.shape[1])

    # Step 4: FFT + Power Spectrum
    power_spec = power_spectrum(windowed)

    # Step 5: Mel Filterbank
    mel_energy = mel_filterbank(power_spec, sr)

    # Step 6: Log + DCT → MFCC
    log_energy = np.log(mel_energy)
    mfccs = dct(log_energy, type=2, axis=1, norm='ortho')[:, :13]

    return signal, mfccs, frames


if __name__ == "__main__":


    folder = "content/drive/MyDrive/audio"
    #Created this path for local system also

    results = {}

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            signal, mfccs, frames = process_audio(path)

            results[file] = {
                "mfcc": mfccs,
                "frames": frames
            }

            print(f"Processed: {file}")
            print(f"{file} → MFCC shape: {mfccs.shape}")


            plt.imshow(mfccs.T, aspect='auto', origin='lower')
            plt.title(f"MFCC - {file}")
            plt.colorbar()
            plt.show()