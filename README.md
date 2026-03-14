# 🎯 Haar Cascade Facial Recognition

Real-time **face and eye detection** using OpenCV's Haar feature-based cascade classifiers. Based on the Viola-Jones object detection framework, this project captures webcam video and draws bounding boxes around detected faces (blue) and eyes (green).

---

## 📖 Description

This project implements facial recognition using **Haar cascades** — a machine-learning approach where a cascade function is trained on positive (face) and negative (non-face) images. The trained classifiers (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`) are applied to each video frame to detect faces and eyes in real time.

Key concepts behind the algorithm:
- **Haar features** act as convolutional kernels to extract facial structure patterns
- **AdaBoost** selects the most discriminative features from 160,000+ candidates
- **Cascade of classifiers** quickly rejects non-face regions for efficient processing

> Originally proposed by Paul Viola & Michael Jones in *"Rapid Object Detection using a Boosted Cascade of Simple Features"* (2001).

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| 🐍 Language | Python 3.8+ |
| 👁️ Computer Vision | OpenCV (`cv2`) |
| 🧠 Detection Model | Haar Cascade Classifiers (XML) |
| 📷 Input | Webcam (device 0) |

---

## 📦 Dependencies

- **Python** ≥ 3.8
- **opencv-python** ≥ 4.5

Install with pip:

```bash
pip install opencv-python
```

---

## 🚀 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/stabgan/HaarFacialRecognition.git
   cd HaarFacialRecognition
   ```

2. **Install dependencies**

   ```bash
   pip install opencv-python
   ```

3. **Run the detector**

   ```bash
   cd "Facial Recognition"
   python face_recognition_commented.py
   ```

4. **Quit** — press `q` in the video window to stop.

---

## 📁 Project Structure

```
HaarFacialRecognition/
├── Facial Recognition/
│   ├── face_recognition_commented.py   # Main detection script
│   ├── haarcascade_frontalface_default.xml  # Pre-trained face model
│   └── haarcascade_eye.xml             # Pre-trained eye model
├── LICENSE
└── README.md
```

---

## ⚠️ Known Issues

- **Webcam required** — the script expects a camera at device index `0`. If you have multiple cameras, edit the `cv2.VideoCapture(0)` argument.
- **Lighting sensitivity** — Haar cascades perform best under even, front-facing lighting. Poor lighting or extreme angles may reduce accuracy.
- **False positives on eyes** — the eye detector can occasionally trigger on eyebrows, nostrils, or other high-contrast regions.
- **No GPU acceleration** — detection runs on CPU only; frame rate may drop on older hardware.
- **Folder name with spaces** — the `Facial Recognition` directory contains a space, which may require quoting on some shells.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
