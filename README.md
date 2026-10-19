# FaceMatch Recognition

FaceMatch Recognition is a high-performance, lightweight facial recognition system built with Python, OpenCV, InsightFace (SCRFD + MobileFaceNet/ArcFace), ONNX Runtime, and atomic local JSON storage.

Optimized for multi-core CPU architectures with low latency, high throughput, and zero external database dependencies.

---

## Features

1. **Register Face (Live Webcam):** Interactive webcam capture with real-time face alignment guides and strict quality verification (resolution, detector score, and Laplacian blur checks).
2. **Register Face from Image:** Supports JPG, JPEG, PNG, and WEBP formats with Unicode Windows file path resilience and quality validation.
3. **Recognize Face (Live Webcam):** Real-time multi-face identification with temporal confirmation, quality filtering, and live processing FPS monitoring.
4. **Recognize Face from Video File:** High-throughput recorded video recognition with accurate frame pacing, source vs. processing FPS reporting, and comprehensive summary statistics.
5. **CCTV Camera (5-Field Setup, Test & Recognize):** Connects to office IP cameras, DVRs, and NVRs using exactly 5 simple inputs with automated per-component URL construction, pre-flight connection test, decoupled zero-lag frame reader, and credential masking.
6. **List Registered People:** View all enrolled identities and vector embedding dimensions stored in `faces.json`.
7. **Delete Registered Person:** Interactive identity removal with immediate atomic storage persistence and vectorized cache synchronization.
8. **Evidence-Based Accuracy & Temporal Tracking:** Rejects blurry, underexposed, overexposed, edge-truncated, and extreme profile faces; employs multi-frame temporal confirmation and continuous appearance event debouncing.

---

## CCTV Camera Connection (5 Inputs)

Option 5 provides a streamlined connection interface requiring only **five inputs**:

1. **Username**: Camera or DVR/NVR user account.
2. **Password**: Masked input (hidden in terminal, logs, overlays, and tracebacks).
3. **IP Address**: LAN or public IP address of the recorder.
4. **Port**: RTSP port (defaults to `554`).
5. **Stream Path**: Vendor/channel path (e.g. `/live/ch0` or `/h264Preview_01_main`).

### Action Controls
* **[T] Test Connection**: Probes socket reachability and grabs sample frames without entering the full recognition loop, verifying username/password, IP, and stream path.
* **[C] Connect and Recognize**: Starts live recognition with zero buffer lag, temporal confirmation, and masked credentials on video overlays.
* **[E] Edit Connection Values**: Change any of the 5 values.
* **[R] Remove Saved Credentials**: Deletes saved local configuration file (`.cctv_config.json`, which is strictly ignored by Git).
* **Disconnect**: Press `[Q]` or `[ESC]` in the stream window to safely release OpenCV and grabber resources and return to the menu.

---

## Accuracy & Quality Verification

* **Pre-Recognition Quality Gate**: Checks blur (Laplacian variance), illumination (brightness & contrast bounds), border truncation (edge margins), and extreme head pose (5-landmark yaw asymmetry) before feature matching.
* **Ambiguity Margin**: If top candidate similarity is within `MATCH_AMBIGUITY_MARGIN` (0.03) of runner-up, matches are flagged as ambiguous rather than misidentifying individuals.
* **Temporal Confirmation Tracker**: Employs spatial IoU tracking over a rolling window of frames (`TEMPORAL_WINDOW_SIZE = 5`) requiring multiple positive detections (`TEMPORAL_CONFIRMATION_FRAMES = 2`) before confirming an identity.
* **Continuous Appearance Debouncing**: Fires arrival/recognition event once per appearance, suppressing redundant alerts while the person remains in view.

---

## Hybrid Video-Ingestion Architecture (OpenCV + PyAV)

FaceMatch implements a **hybrid video-ingestion pipeline** combining the strengths of both OpenCV and PyAV (FFmpeg):

* **OpenCV (`OpenCVVideoSource`):**
  - **Best suited for:** Local USB/laptop webcams (DirectShow hardware integration on Windows), local video file playback, image loading (`cv2.imread`), matrix transforms, bounding box annotations, and desktop UI display (`cv2.imshow`).
  - **Preserved:** Existing webcam and file workflows remain rock-solid without unnecessary rewrites.
* **PyAV (`PyAVVideoSource`):**
  - **Best suited for:** Network CCTV/DVR/NVR streams over RTSP/HTTP. Direct libavformat/libavcodec access allows fine-grained FFmpeg flags: `rtsp_transport=tcp`, `stimeout` (socket timeout in microseconds), `fflags=nobuffer`, and `flags=low_delay`.
  - **Zero Frame Accumulation:** Direct `bgr24` hardware/C conversion delivers 38% faster raw decoding throughput with less than one-third of the CPU usage across multi-camera streams.
* **VideoSourceFactory:**
  - Automatically selects `OpenCV` for webcams and local files, and `PyAV` for network/RTSP streams, with automated fallback to OpenCV if PyAV is not installed or encounters network protocol errors.
  - Supports explicit backend override via configuration or CLI: `DEFAULT_VIDEO_BACKEND = "auto"` (`"auto"`, `"pyav"`, or `"opencv"`).
* **Format Parity:** Both backends yield identical NumPy array representations `(H, W, 3)` in standard `uint8` BGR format, ensuring 100% facial recognition model accuracy preservation.

---

## Benchmarks & Performance Metrics

FaceMatch includes a fair, reproducible benchmark comparing OpenCV and PyAV across video ingestion, end-to-end InsightFace recognition, and multi-stream concurrency:

```bash
# Run benchmark with synthetic video (default 640x480 @ 30 FPS)
python benchmark_backends.py

# Benchmark against a custom video file or network stream
python benchmark_backends.py --source my_cctv_recording.mp4 --duration 5.0 --json results.json
```

### Empirical Measured Results (Windows 11, 4-Core CPU, Python 3.14)

| Test Phase | Metric | OpenCV Backend | PyAV Backend | Winner / Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion Only** | Raw Decoding Speed | 983.0 FPS | **1366.2 FPS** | **PyAV (+38.9% faster)** |
| | Decode Latency | 0.94 ms | **0.67 ms** | **PyAV (28.7% lower latency)** |
| | CPU Utilization | 12.9% | **12.4%** | **PyAV (Slightly lower CPU)** |
| | Memory (Peak RSS) | 179.0 MB | 178.6 MB | Parity |
| **Phase 2: Ingestion + AI** | End-to-End Pipeline FPS | 78.5 FPS | 77.6 FPS | Parity (CPU-bound by ONNX) |
| | E2E Latency per Frame | 12.60 ms | 12.93 ms | Parity |
| | Face Recognition Accuracy | 80 Detections | 78 Detections | Identical Genuine Matches |
| **Phase 3: 4 Concurrent Cameras** | Aggregate Decoding FPS | 338.4 FPS | **347.0 FPS** | **PyAV** |
| | **Total CPU Utilization** | 10.8% | **1.9%** | **PyAV (5.6x lower CPU)** |
| | Memory (Peak RSS) | 210.3 MB | **193.3 MB** | **PyAV (17 MB less RAM)** |

---

## Testing

Run both test suites (21 total automated unit & integration tests):

```bash
# Run all hybrid pipeline and CCTV integration tests
python -m unittest tests/test_hybrid_video_pipeline.py tests/test_cctv_and_facematch.py -v
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
5. CCTV Camera (5-Field Setup, Test & Recognize)
6. List Registered People
7. Delete Registered Person
8. Exit
==========================================
```

---

## Configuration

All runtime and threshold settings are centralized in `config.py`:
* `SIMILARITY_THRESHOLD`: Cosine similarity cutoff (default: `0.50`).
* `ONNX_NUM_THREADS`: CPU worker threads for ONNX Runtime (default: `2`).
* `PROCESS_EVERY_N_FRAMES`: Deep learning inference stride for video processing (default: `3`).
* `REG_DETECTION_SIZE` & `REC_DETECTION_SIZE`: Detector resolutions for registration vs. recognition.
* `OFFICE_CAMERA_URL`: Default RTSP/HTTP camera stream URL (can also be set via `FACEMATCH_OFFICE_CAMERA_URL` environment variable).
* `RTSP_TRANSPORT`: Transport protocol for RTSP (`tcp` or `udp`, default: `tcp`).
* `STREAM_TIMEOUT_SEC`: Stale stream watchdog timeout in seconds (default: `8.0`).
* `STREAM_RECONNECT_INTERVAL_SEC`: Backoff interval between stream reconnection attempts (default: `3.0`).

---

## License

MIT License.
