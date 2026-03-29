from datasets import load_dataset
import random
from collections import Counter

print("Loading dataset (fast subset)...")

dataset = load_dataset(
    "librispeech_asr",
    "clean",
    split="train.100",
    streaming=True
)

samples = []
speakers = set()

MAX_SPEAKERS = 18
MAX_SAMPLES = 2000

for sample in dataset:
    samples.append(sample)
    speakers.add(sample["speaker_id"])

    if len(speakers) >= MAX_SPEAKERS or len(samples) >= MAX_SAMPLES:
        break

print(f"Loaded {len(samples)} samples")
print(f"Unique speakers: {len(speakers)}")

#  Simulated metadata
metadata = {}
for spk in speakers:
    metadata[spk] = {
        "gender": random.choice(["male", "female"]),
        "age": random.choice(["young", "old"])
    }

#  Bias distribution
gender_dist = Counter([metadata[s]["gender"] for s in speakers])
age_dist = Counter([metadata[s]["age"] for s in speakers])

print("Gender Distribution:", gender_dist)
print("Age Distribution:", age_dist)
