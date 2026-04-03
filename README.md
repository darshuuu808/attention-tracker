# Real-Time Attention Estimation System

![Python](https://img.shields.io/badge/Python-3.x-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-purple) ![CPU Only](https://img.shields.io/badge/Inference-CPU%20Only-gray)

Lightweight attention monitoring using geometric multi-cue fusion — no ML training, no GPU required. Runs at 30FPS on any laptop.

---

## How it works

Three independent cues fused into a continuous attention score (0–100%):

| Cue | Method |
|-----|--------|
| Eye closure | Eye Aspect Ratio (EAR) from 6 landmark points per eye |
| Fatigue onset | Blink frequency over a 30-second sliding window |
| Distraction | Head yaw/pitch estimation via facial geometry |

---

## Attention Classification

| Score | Status |
|-------|--------|
| 70–100% | Alert |
| 40–69% | Drowsy |
| 0–39% | Distracted |

---

## Key Features

- Personalized calibration — 10-second baseline EAR measurement per user
- Multi-cue fusion — combines eye, blink, and head signals into one score
- Session logging — saves attention data to `session_log.csv` every second
- Auto report — generates attention graph on session end
- Audio alerts — beeps when attention drops critically
- Zero ML training — pure geometric computation, runs at 30FPS on any laptop

---

## Project Structure
```
attention-tracker/
├── main.py          # Entry point, webcam loop, UI overlay
├── ear.py           # Eye Aspect Ratio computation
├── head_pose.py     # Yaw/pitch estimation from landmarks
├── scorer.py        # Multi-cue attention score fusion
├── logger.py        # CSV logging + matplotlib report
└── face_landmarker.task  # MediaPipe face model
```

---

## Setup & Run

**1. Install dependencies**
```bash
pip install opencv-python mediapipe numpy matplotlib
```

**2. Clone the repo**
```bash
git clone https://github.com/darshuuu808/attention-tracker.git
cd attention-tracker
```

**3. Download model**
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task'); print('Done')"
```

**4. Run**
```bash
python main.py
```

> Calibrates for 10 seconds — look at the camera naturally. Press `q` to end session and generate report.

---

## Session Report

After each session, an attention graph is auto-generated and saved as `session_report.png`:

- Green dots = Alert
- Orange dots = Drowsy
- Red dots = Distracted

---

## Tech Stack

- Python 3.x
- OpenCV — webcam feed, frame processing, UI overlay
- MediaPipe — 468-point facial landmark detection
- NumPy — EAR geometric computation
- Matplotlib — session analytics visualization

---

## Use Cases

- Driver drowsiness detection
- Online exam attention monitoring
- Workplace focus tracking
- Patient alertness monitoring

---

## Author

**Darshan** — Rajalakshmi Engineering College
[GitHub](https://github.com/darshuuu808) · [LinkedIn](https://linkedin.com/in/darshannns)
