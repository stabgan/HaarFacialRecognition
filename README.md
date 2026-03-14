# Haar Cascade Facial Recognition

Real-time face and eye detection from a webcam feed using OpenCV's Haar feature-based cascade classifiers.

## What It Does

Captures live video from your webcam and draws bounding boxes around detected **faces** (blue) and **eyes** (green) using the Viola-Jones object detection framework. The algorithm applies a cascade of increasingly complex classifiers to quickly discard non-face regions and focus computation where it matters.

## How It Works

1. **Haar Features** — Rectangular filters (similar to convolution kernels) compute intensity differences across image regions.
2. **AdaBoost** — Selects the most discriminative features from 160 000+ candidates down to ~6 000.
3. **Cascade of Classifiers** — Features are grouped into stages; a region that fails any stage is immediately rejected, making detection fast enough for real-time use.

> Based on: Viola & Jones, *"Rapid Object Detection using a Boosted Cascade of Simple Features"*, 2001.

## 🛠 Tech Stack

| Component | Tool |
|-----------|------|
| 🐍 Language | Python 3.8+ |
| 👁 Computer Vision | OpenCV (`opencv-python`) |
| 🎯 Detection | Haar Cascade Classifiers |

## Getting Started

### Install dependencies

```bash
pip install opencv-python
```

### Run

```bash
cd "Facial Recognition"
python face_recognition_commented.py
```

Press **q** to quit the video window.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Webcam device index |
| `--face-cascade` | `haarcascade_frontalface_default.xml` | Path to face cascade XML |
| `--eye-cascade` | `haarcascade_eye.xml` | Path to eye cascade XML |

## Project Structure

```
Facial Recognition/
├── face_recognition_commented.py        # Main detection script
├── haarcascade_frontalface_default.xml   # Pre-trained face cascade
└── haarcascade_eye.xml                   # Pre-trained eye cascade
```

## ⚠️ Known Issues

- Haar cascades can produce false positives in complex lighting or with tilted faces. For production use, consider DNN-based detectors (`cv2.dnn`) or MediaPipe.
- Requires a working webcam; headless environments will fail to open the video capture.

## License

MIT
