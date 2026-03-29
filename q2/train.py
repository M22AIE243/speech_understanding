import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from datasets import load_dataset
import random

# =========================
# CONFIG
# =========================
NUM_SPEAKERS = 50   # Keep small for Colab
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD SMALL DATASET (NO DISK EXPLOSION)
# =========================
print("Loading dataset (streaming mode)...")

dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)

# Take limited samples (VERY IMPORTANT)
data = []
speaker_map = {}
speaker_count = 0

for sample in dataset:
    spk = sample["speaker_id"]

    if spk not in speaker_map:
        if speaker_count >= NUM_SPEAKERS:
            continue
        speaker_map[spk] = speaker_count
        speaker_count += 1

    data.append({
        "audio": sample["audio"]["array"],
        "label": speaker_map[spk]
    })

    if len(data) >= 2000:   # LIMIT SIZE (prevents disk issue)
        break

print(f"Loaded {len(data)} samples with {speaker_count} speakers")

# =========================
# FEATURE EXTRACTOR (MFCC)
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
# DATASET CLASS
# =========================
class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = extract_features(item["audio"])
        y = item["label"]
        return x, y

dataset = SpeakerDataset(data)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

model = BaselineModel(num_classes=speaker_count).to(DEVICE)

# =========================
# TRAINING
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting training...")

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")

# =========================
# SAVE MODEL + METADATA
# =========================
torch.save({
    "model": model.state_dict(),
    "num_speakers": speaker_count
}, "baseline_model.pth")

print("Model saved successfully!")