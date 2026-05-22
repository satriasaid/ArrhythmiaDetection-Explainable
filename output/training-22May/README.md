# ECG 7-Class Classifier — Bundle

Three trained models for 7-class ECG diagnosis.

## Classes
`NORM, AFIB, AFLT, 1dAVb, RBBB, LBBB, OTHERS`

## Install
```
pip install torch numpy pillow scikit-learn
```

## Files
- `xresnet1d101_traced.pt` — 1D conv on raw signal. Input: (12, 1000) .npy
- `hubert_ecg_traced.pt`   — Transformer on lead II. Input: (12, 1000) .npy
- `biomedclip_traced.pt`   — ViT on rendered 12-lead PNG. Input: PNG
- `scaler.pkl`             — per-lead z-score scaler (for signal models)
- `classes.json`           — class order + preprocessing config
- `predict.py`             — runs any model on one sample

## Run
```bash
python predict.py xresnet1d101_traced.pt sample.npy
python predict.py hubert_ecg_traced.pt   sample.npy
python predict.py biomedclip_traced.pt   sample.png
```

Output: JSON with predicted class, confidence, and full prob distribution.

## Signal format
- 12 leads, 100 Hz, 10 seconds = (12, 1000) float32 in mV
- Order: I, II, III, aVR, aVL, aVF, V1-V6
- Save as `np.save('sample.npy', signal_12x1000)`

## If a `_traced.pt` failed export
A `_state_dict.pt` + `_class.py` pair is present instead. Load via:
```python
from xxx_class import TheClass
m = TheClass(); m.load_state_dict(torch.load('xxx_state_dict.pt'))
```
