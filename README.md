# FaceMatch Recognition

FaceMatch Recognition is a high-performance, lightweight facial recognition system built with Python, OpenCV, InsightFace (SCRFD + MobileFaceNet/ArcFace), ONNX Runtime, and atomic local JSON storage.

Optimized for multi-core CPU architectures with low latency, high throughput, and zero external database dependencies.

---

## Features

1. **Register Face (Live Webcam):** Interactive webcam capture with real-time face alignment guides and strict quality verification (resolution, detector score, and Laplacian blur checks).
2. **Register Face from Image:** Supports JPG, JPEG, PNG, and WEBP formats with Unicode Windows file path resilience and quality validation.
3. **Recognize Face (Live Webcam):** Real-time multi-face identification with tolerant recognition settings and live processing FPS monitoring.
4. **Recognize Face from Video File:** High-throughput recorded video recognition (35+ FPS) with accurate frame pacing, source vs. processing FPS reporting, and comprehensive summary statistics.
5. **List Registered People:** View all enrolled identities and vector embedding dimensions stored in `faces.json`.
6. **Delete Registered Person:** Interactive identity removal with immediate atomic storage persistence and vectorized cache synchronization.
7. **Vectorized Cosine Similarity:** Fast NumPy matrix batch dot product for matching against enrolled identities.

---

## Performance Highlights

* **Thread-Safe Initialization:** Pinned ONNX Runtime session thread pools avoiding OpenMP contention.
* **Separated Detector Sizes:** 640×640 detection for high-detail registration; 320×320 for real-time 30+ FPS recognition.
* **High-Precision Scaled Pipeline:** Downscaled detection for fast bounding-box discovery coupled with full-resolution ArcFace feature extraction.
* **Atomic Storage:** Atomic rename with file system flushing prevents corruption during sudden process terminations.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/antrikshjainsquadkin/facematch-recognition.git
cd facematch-recognition
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

Run the main application:

```bash
python main.py
```

Console menu:
```text
==========================================
         FaceMatch Recognition           
==========================================
1. Register Face (Live Webcam)
2. Register Face from Image
3. Recognize Face (Live Webcam)
4. Recognize Face from Video File
5. List Registered People
6. Delete Registered Person
7. Exit
==========================================
```

---

## Configuration

All runtime and threshold settings are centralized in `config.py`:
* `SIMILARITY_THRESHOLD`: Cosine similarity cutoff (default: `0.50`).
* `ONNX_NUM_THREADS`: CPU worker threads for ONNX Runtime (default: `2`).
* `PROCESS_EVERY_N_FRAMES`: Deep learning inference stride for video processing (default: `3`).
* `REG_DETECTION_SIZE` & `REC_DETECTION_SIZE`: Detector resolutions for registration vs. recognition.

---

## License

MIT License.
