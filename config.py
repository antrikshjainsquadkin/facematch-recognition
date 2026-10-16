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
# Recognition Quality Thresholds (TOLERANT)
# Enables detection of smaller, distant, or low-res faces
# ==========================================
REC_MIN_FACE_WIDTH: int = 24        # Tolerates smaller/distant faces in low-res videos
REC_MIN_FACE_HEIGHT: int = 24       # Tolerates smaller/distant faces in low-res videos
REC_MIN_DET_SCORE: float = 0.35     # Tolerant detector confidence for low light / compression
REC_INFERENCE_SCALE: float = 1.0    # Full resolution inference for recorded video analysis

# ==========================================
# Face Identity Matching Threshold
# ==========================================
# Cosine similarity threshold for MobileFaceNet/ArcFace embeddings.
# Identities are recognized only when similarity score >= SIMILARITY_THRESHOLD;
# otherwise classified as 'Unknown Person'.
SIMILARITY_THRESHOLD: float = 0.50

# ==========================================
# Camera & Image Input Settings
# ==========================================
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
WINDOW_TITLE: str = "FaceMatch"

# Supported image formats for image-based face registration
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
