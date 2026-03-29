import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
from sklearn.metrics import roc_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load("baseline_model.pth", map_location=DEVICE)
num_speakers = checkpoint["num_speakers"]

print(f"Loaded model with {num_speakers} speakers")

# =========================
# MODEL
# =========================
class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def extract_embedding(self, x):
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x

model = BaselineModel(num_speakers).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

# =========================
# FEATURE EXTRACTOR
# =========================
mfcc = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40
)

def extract_features(audio):
    waveform = torch.tensor(audio).float().unsqueeze(0)
    features = mfcc(waveform)
    return features.mean(dim=2).squeeze()

# =========================
# LOAD TRAIN DATA AGAIN (CONSISTENT)
# =========================
from datasets import load_dataset

dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)

data = []
speaker_map = {}
speaker_count = 0

for sample in dataset:
    spk = sample["speaker_id"]

    if spk not in speaker_map:
        if speaker_count >= num_speakers:
            continue
        speaker_map[spk] = speaker_count
        speaker_count += 1

    data.append({
        "audio": sample["audio"]["array"],
        "label": speaker_map[spk]
    })

    if len(data) >= 2000:
        break

# =========================
# SPLIT INTO TRAIN / TEST
# =========================
random.shuffle(data)
split = int(0.8 * len(data))

train_data = data[:split]
test_data = data[split:]

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# =========================
# CLASSIFICATION ACCURACY
# =========================
correct = 0
total = 0

for item in test_data:
    x = extract_features(item["audio"]).to(DEVICE)
    y = item["label"]

    with torch.no_grad():
        out = model(x.unsqueeze(0))
        pred = out.argmax(dim=1).item()

    if pred == y:
        correct += 1
    total += 1

accuracy = correct / total
print(f"\nClassification Accuracy: {accuracy:.4f}")

# =========================
# SPEAKER VERIFICATION
# =========================
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

scores = []
labels = []

for _ in range(300):
    a, b = random.sample(test_data, 2)

    x1 = extract_features(a["audio"]).to(DEVICE)
    x2 = extract_features(b["audio"]).to(DEVICE)

    with torch.no_grad():
        emb1 = model.extract_embedding(x1.unsqueeze(0))
        emb2 = model.extract_embedding(x2.unsqueeze(0))

    score = cosine_similarity(emb1, emb2)
    same = 1 if a["label"] == b["label"] else 0

    scores.append(score)
    labels.append(same)

# =========================
# EER
# =========================
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr

eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

print(f"EER: {eer:.4f}")