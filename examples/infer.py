"""Minimal tensor API example; supply your own compatible checkpoint."""

import argparse

import torch
from trifusion_l4 import configure_runtime, load_student, InferenceEngine

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
args = parser.parse_args()
configure_runtime()
model = load_student(args.weights)
engine = InferenceEngine(model, graphs=True, max_graphs=1)
# Replace this synthetic signal with your own PCM16-equivalent 16-kHz audio.
audio = torch.zeros(1, 1, 4 * 16000, device="cuda", dtype=torch.float32)
lengths = torch.tensor([audio.shape[-1]], device="cuda", dtype=torch.int64)
channels = torch.tensor([1], device="cuda", dtype=torch.int64)
probabilities = engine(audio, lengths, channels).float().sigmoid()
print(probabilities.cpu().tolist())
engine.clear()
