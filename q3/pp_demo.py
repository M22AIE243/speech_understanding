import torch
import torchaudio
from privacymodule import PrivacyObfuscator

def extract_mfcc(audio, sr=16000):
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=40
    )(audio)
    return mfcc

# Dummy audio (you can replace with real sample)
audio = torch.randn(1, 16000)

mfcc = extract_mfcc(audio)

model = PrivacyObfuscator()
mfcc_private = model(mfcc)

print("Original MFCC:", mfcc.shape)
print("Obfuscated MFCC:", mfcc_private.shape)