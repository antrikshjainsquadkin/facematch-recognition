"""Configuration module for FaceMatch.

Holds model, camera, recognition threshold, storage, and performance settings.
Separates strict registration criteria from fast, tolerant real-time recognition.
"""

import os
from pathlib import Path

# ==========================================
# Project Base Directory & JSON Storage
# ==========================================
BASE_DIR: Path = Path(__file__).resolve().parent
FACES_JSON_PATH: Path = BASE_DIR / "faces.json"
EMBEDDING_DIM: int = 512

# Optional project-local models directory (if bundled for offline sandboxes)
LOCAL_MODELS_DIR: Path = BASE_DIR / "models"

# ==========================================
# InsightFace Model Configuration
# ==========================================
# buffalo_s is ultra-fast, lightweight (MobileFaceNet + 500M SCRFD detector)
MODEL_PACKAGE_NAME: str = "buffalo_s"
ALLOWED_MODULES: list[str] = ["detection", "recognition"]
CTX_ID: int = -1

# Separate detector resolutions for registration vs real-time recognition:
# Registration: 640x640 provides higher precision for face landmarks and alignment
REG_DETECTION_SIZE: tuple[int, int] = (640, 640)

# Recognition: 320x320 provides ~2x faster detection throughput (25+ FPS) on CPU
REC_DETECTION_SIZE: tuple[int, int] = (320, 320)

# Default detection size for backwards compatibility
DETECTION_SIZE: tuple[int, int] = REC_DETECTION_SIZE

# ==========================================
# CPU & ONNX Runtime Performance Optimization
# ==========================================
# Benchmarked optimal thread count: 2 threads avoids OpenMP thrashing/overhead
# on multi-core CPU architectures (2 threads is ~3x faster than 4 threads for SCRFD/ArcFace).
_TOTAL_CORES = os.cpu_count() or 2
ONNX_NUM_THREADS: int = 2 if _TOTAL_CORES >= 2 else 1

# Live webcam preview downscaling factor (0.5 for fast preview, 1.0 for original resolution)
INFERENCE_SCALE: float = 0.5

# Run deep learning inference once every N frames to keep CPU usage low
PROCESS_EVERY_N_FRAMES: int = 3

# ==========================================
# Registration Quality Thresholds (STRICT)
# Enforces high-quality, clear, sharp faces only
# ==========================================
REG_MIN_FACE_WIDTH: int = 80        # Minimum face box width in pixels
REG_MIN_FACE_HEIGHT: int = 80       # Minimum face box height in pixels
REG_MIN_DET_SCORE: float = 0.65     # Strict detector confidence (0.0 to 1.0)
REG_MIN_SHARPNESS: float = 40.0     # Minimum Laplacian variance (blur rejection)

# ==========================================
# Recognition Quality Thresholds (Evidence-Based Accuracy)
# Enforces reliable detection, rejecting blur, poor lighting, and edge crops
# ==========================================
REC_MIN_FACE_WIDTH: int = 36        # Minimum face box width in pixels (reject tiny artifacts)
REC_MIN_FACE_HEIGHT: int = 36       # Minimum face box height in pixels
REC_MIN_DET_SCORE: float = 0.40     # Detector confidence threshold
REC_INFERENCE_SCALE: float = 1.0    # Full resolution inference for recorded video analysis
REC_MIN_SHARPNESS: float = 14.0     # Minimum Laplacian variance for recognition (blur rejection)
REC_MIN_BRIGHTNESS: float = 32.0    # Minimum average pixel luminance (darkness rejection)
REC_MAX_BRIGHTNESS: float = 228.0   # Maximum average pixel luminance (overexposure rejection)
REC_MIN_CONTRAST: float = 14.0      # Minimum grayscale std-dev (flat region rejection)
REC_EDGE_MARGIN_PX: int = 4         # Border margin in pixels; faces touching edge are rejected
REC_MAX_YAW_RATIO: float = 0.75     # Extreme side-profile pose rejection from 5 facial landmarks

# ==========================================
# Face Identity Matching & Ambiguity Thresholds
# ==========================================
# Cosine similarity threshold for MobileFaceNet/ArcFace embeddings.
# Identities are recognized only when similarity score >= SIMILARITY_THRESHOLD;
# otherwise classified as 'Unknown Person'.
SIMILARITY_THRESHOLD: float = 0.50
MATCH_AMBIGUITY_MARGIN: float = 0.03  # Top candidate must beat runner-up by this margin

# ==========================================
# Masked-Face Recognition & Detection Thresholds
# ==========================================
# Enable or disable automated face mask detection
MASK_DETECTION_ENABLED: bool = True

# Probability threshold (0.0 - 1.0) for classifying a face as wearing a mask
MASK_SCORE_THRESHOLD: float = 0.45

# Cosine similarity threshold for faces wearing masks.
# Occlusion of lower facial features (mouth/nose/chin) reduces full-face score,
# while periocular and upper facial features reliably produce ~0.60-0.72 similarity
# for genuine identities and <= 0.15 for imposters. A calibrated threshold of 0.38
# allows robust matching under varied illumination and angles while preventing false acceptances.
MASKED_SIMILARITY_THRESHOLD: float = 0.38

# Ambiguity margin for masked faces: top candidate must beat runner-up by this margin
MASKED_MATCH_AMBIGUITY_MARGIN: float = 0.03

# ==========================================
# Temporal Confirmation & Event Tracking
# ==========================================
# Require multiple confirmed frames within a rolling window before confirming identity
TEMPORAL_WINDOW_SIZE: int = 5
TEMPORAL_CONFIRMATION_FRAMES: int = 2
EVENT_DEBOUNCE_COOLDOWN_SEC: float = 5.0  # Cooldown between event logs for same person

# ==========================================
# Camera & Image Input Settings
# ==========================================
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
WINDOW_TITLE: str = "FaceMatch"

# Supported image formats for image-based face registration
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

# ==========================================
# Hybrid Video Pipeline & Ingestion Backends
# ==========================================
# Video backend selection: "auto", "pyav", or "opencv"
# - "auto": Uses OpenCV for webcams & local files; uses PyAV for RTSP network streams.
# - "pyav": Forces PyAV (with graceful fallback to OpenCV on failure).
# - "opencv": Forces OpenCV (cv2.VideoCapture) everywhere.
DEFAULT_VIDEO_BACKEND: str = os.getenv("FACEMATCH_VIDEO_BACKEND", "auto").lower().strip()

# PyAV FFmpeg-level Demuxer Options
PYAV_RTSP_TRANSPORT: str = "tcp"  # TCP transport avoids UDP packet drops
PYAV_TIMEOUT_SEC: float = 5.0     # Network connection & read timeout in seconds
PYAV_LOW_DELAY: bool = True       # Force low_delay flags in FFmpeg
PYAV_NOBUFFER: bool = True        # Discard internal buffer queues for 0 ms live lag
PYAV_BUFFER_SIZE: int = 1         # Bounded frame buffer depth

# ==========================================
# Office Camera / RTSP & IP Stream Settings
# ==========================================
# Saved CCTV configuration file (gitignored to protect credentials)
CCTV_CONFIG_PATH: Path = BASE_DIR / ".cctv_config.json"
DEFAULT_RTSP_PORT: int = 554
RTSP_CONNECT_TIMEOUT_SEC: float = 5.0

# Can be set via environment variable FACEMATCH_OFFICE_CAMERA_URL or entered interactively.
OFFICE_CAMERA_URL: str = os.getenv("FACEMATCH_OFFICE_CAMERA_URL", "").strip()
RTSP_TRANSPORT: str = "tcp"  # TCP transport avoids UDP packet loss and corrupted frames
STREAM_TIMEOUT_SEC: float = 8.0  # Seconds without new frame before triggering reconnection
STREAM_RECONNECT_INTERVAL_SEC: float = 3.0  # Backoff delay between reconnection attempts
STREAM_MAX_RECONNECT_ATTEMPTS: int = 5

