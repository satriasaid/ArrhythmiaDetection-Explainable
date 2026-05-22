#!/usr/bin/env python3
"""Run a trained ECG classifier on one sample.

Install:
    pip install torch numpy pillow scikit-learn

Usage:
    # Signal models (xresnet1d101, hubert_ecg) — input: .npy of shape (12, 1000), 100Hz, mV
    python predict.py xresnet1d101_traced.pt sample.npy
    python predict.py hubert_ecg_traced.pt   sample.npy

    # Image model (biomedclip) — input: PNG of rendered 12-lead ECG
    python predict.py biomedclip_traced.pt sample.png
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
META = json.loads((HERE / 'classes.json').read_text())
CLASSES = META['classes']

def load_signal_tensor(npy_path):
    with open(HERE / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    sig = np.load(npy_path).astype(np.float32)   # (12, 1000)
    assert sig.shape == (12, 1000), f'expected (12,1000), got {sig.shape}'
    scaled = scaler.transform(sig.T).T.astype(np.float32)  # scaler expects (T, 12)
    return torch.from_numpy(scaled).unsqueeze(0)  # (1, 12, 1000)

def load_image_tensor(png_path):
    pp = META['biomedclip_preprocess']
    size = pp['resize']
    mean = torch.tensor(pp['mean']).view(3, 1, 1)
    std  = torch.tensor(pp['std']).view(3, 1, 1)
    img = Image.open(png_path).convert('RGB').resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return ((arr - mean) / std).unsqueeze(0)

def main(model_path, sample_path):
    model = torch.jit.load(model_path, map_location='cpu').eval()
    sample_path = Path(sample_path)
    if sample_path.suffix == '.npy':
        x = load_signal_tensor(sample_path)
    elif sample_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        x = load_image_tensor(sample_path)
    else:
        sys.exit(f'unknown sample type: {sample_path.suffix}')
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = int(probs.argmax())
    print(json.dumps({
        'prediction': CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'probabilities': {c: float(probs[i]) for i, c in enumerate(CLASSES)},
    }, indent=2))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
