import numpy as np
import librosa
import torch
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

#  Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


#  Cepstrum
def cepstrum(signal):
    spectrum = np.fft.fft(signal)
    log_spec = np.log(np.abs(spectrum) + 1e-10)
    return np.fft.ifft(log_spec).real


#  Voiced Detection
def detect_voiced(frame, threshold=0.1):
    cep = cepstrum(frame)
    return 1 if np.max(cep[20:100]) > threshold else 0


#  Framing
def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
    frame_len = int(frame_size * sr)
    step = int(frame_stride * sr)

    frames = []
    for i in range(0, len(signal) - frame_len, step):
        frames.append(signal[i:i + frame_len])

    return np.array(frames)


#  Energy-based labels (reference)
def energy_based_labels(frames, threshold_ratio=0.5):
    energies = np.sum(frames ** 2, axis=1)
    threshold = threshold_ratio * np.max(energies)
    return (energies > threshold).astype(int)


#  RMSE
def compute_rmse(pred, true):
    n = min(len(pred), len(true))
    return np.sqrt(np.mean((pred[:n] - true[:n]) ** 2))


#  Main
if __name__ == "__main__":

    folder = "content/drive/MyDrive/audio/"
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    for file in files:
        path = os.path.join(folder, file)

        # Load audio
        signal, sr = librosa.load(path, sr=16000)

        #  Wav2Vec2 transcription
        input_values = processor(signal, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.decode(pred_ids[0])

        print(f"\n{file} → {text}")

        #  Your segmentation
        frames = framing(signal, sr)
        pred = np.array([detect_voiced(f) for f in frames])

        #  Reference segmentation
        true = energy_based_labels(frames)

        # RMSE
        rmse_val = compute_rmse(pred, true)

        print(f"{file} → RMSE: {rmse_val:.4f}")
