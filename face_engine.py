"""Face Recognition Engine using InsightFace and ONNX Runtime.

Handles model initialization, face detection, embedding extraction,
quality verification, identity matching, and JSON storage serialization.
Optimized for high CPU throughput, vectorized matching, thread safety, and zero redundant passes.
"""

import os
import json
import time
import threading
import warnings
from pathlib import Path

# Suppress external library deprecation warnings from insightface / scikit-image
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import config

# Restrict lower-level threading to avoid thrashing on multi-core CPU architectures
if config.ONNX_NUM_THREADS > 0:
    os.environ["OMP_NUM_THREADS"] = str(config.ONNX_NUM_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(config.ONNX_NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(config.ONNX_NUM_THREADS)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(config.ONNX_NUM_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(config.ONNX_NUM_THREADS)

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    ort = None

try:
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face
except ImportError as exc:
    raise ImportError(
        "InsightFace is not installed. Please install dependencies using: "
        "pip install opencv-python onnxruntime insightface"
    ) from exc


# ============================================================================
# Quality Verification Helpers
# ============================================================================

def enhance_low_light_frame(
    frame: np.ndarray,
    brightness_threshold: float = config.LOW_LIGHT_BRIGHTNESS_THRESHOLD,
) -> Tuple[np.ndarray, bool]:
    """Dynamically enhance underexposed RGB or monochrome IR frames using adaptive Gamma and CLAHE.

    Bypasses enhancement for well-lit frames (mean brightness >= threshold) with zero latency overhead.
    For underexposed frames, lifts deep shadows and improves local facial contrast without blowing highlights
    or amplifying high-frequency noise.

    Args:
        frame: Input frame (BGR or single-channel grayscale).
        brightness_threshold: Luminance below which enhancement is activated.

    Returns:
        Tuple of (processed_frame: np.ndarray, was_enhanced: bool).
    """
    if frame is None or frame.size == 0 or not config.ENHANCE_LOW_LIGHT_ENABLED:
        return frame, False

    # Ensure 3 channels for InsightFace SCRFD and ArcFace
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Fast luminance check on downscaled grayscale
    h, w = frame.shape[:2]
    small_gray = cv2.resize(frame, (min(w, 160), min(h, 120)), interpolation=cv2.INTER_NEAREST)
    if len(small_gray.shape) == 3:
        small_gray = cv2.cvtColor(small_gray, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(small_gray))

    if mean_val >= brightness_threshold:
        return frame, False

    # Check if frame is monochrome / IR
    b, g, r = cv2.split(frame)
    diff_bg = cv2.absdiff(b, g)
    diff_gr = cv2.absdiff(g, r)
    is_monochrome = float(np.mean(diff_bg)) < 4.0 and float(np.mean(diff_gr)) < 4.0

    gamma = float(np.clip(np.log(45.0 / max(mean_val, 1.0)) / np.log(2.0), 1.2, 2.8))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)

    if is_monochrome:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_gamma = cv2.LUT(gray, table)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enh = clahe.apply(gray_gamma)
        enhanced = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)
        return enhanced, True
    else:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        l_gamma = cv2.LUT(l, table)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enh = clahe.apply(l_gamma)
        enhanced_lab = cv2.merge([l_enh, a, b_ch])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced, True


def compute_sharpness(crop: np.ndarray) -> float:
    """Compute image sharpness metric using Laplacian variance.

    Higher values indicate sharp, in-focus images.
    Lower values indicate blurriness, motion blur, or camera out-of-focus.

    Args:
        crop: BGR image crop of the face.

    Returns:
        Laplacian variance score as a float.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_mask_probability(
    frame: np.ndarray,
    face: Any,
    score_threshold: float = config.MASK_SCORE_THRESHOLD,
) -> Tuple[bool, float]:
    """Detect whether a face is wearing a face mask (surgical, N95, cloth, or dark mask).

    Supports both full-color RGB frames and monochrome/near-infrared (IR) nighttime frames.
    Analyzes facial geometry from 5-point facial landmarks:
    - Upper periocular region (forehead, eyebrows, eyes)
    - Lower facial region (nose bridge down to chin)

    Compares color tone, saturation, edge density, and lip presence across the two regions.
    In monochrome/IR frames, evaluates smooth lower-face fabric texture and upper-to-lower
    luminance contrast across the nose bridge rim.

    Args:
        frame: BGR image frame.
        face: Detected InsightFace Face instance.
        score_threshold: Probability threshold above which face is classified as masked.

    Returns:
        Tuple of (is_masked: bool, mask_score: float).
    """
    if frame is None or frame.size == 0 or face is None:
        return False, 0.0

    bx1, by1, bx2, by2 = [int(v) for v in face.bbox]
    kps = getattr(face, "kps", None)
    h_img, w_img = frame.shape[:2]
    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(w_img, bx2)
    by2 = min(h_img, by2)
    w = bx2 - bx1
    h = by2 - by1
    if w < 10 or h < 10:
        return False, 0.0

    # Upper region (forehead / periocular) and lower region (nose bridge down to chin)
    if kps is not None and len(kps) >= 5:
        nose_y = int(kps[2][1])
    else:
        nose_y = int(by1 + 0.55 * h)

    # Ensure valid vertical bounds
    nose_y = max(by1 + int(0.30 * h), min(by2 - int(0.20 * h), nose_y))

    upper_crop = frame[by1:nose_y, bx1:bx2]
    lower_crop = frame[nose_y:by2, bx1:bx2]
    if upper_crop.size == 0 or lower_crop.size == 0:
        return False, 0.0

    upper_hsv = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2HSV)
    lower_hsv = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2HSV)

    u_h, u_s, u_v = [float(np.median(upper_hsv[:, :, i])) for i in range(3)]
    l_h, l_s, l_v = [float(np.median(lower_hsv[:, :, i])) for i in range(3)]

    # Lower face edge density
    lower_gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(lower_gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(edges.size)

    # Check if frame is monochrome / night-vision IR
    is_monochrome = (u_s < 12.0 and l_s < 12.0)
    score = 0.0

    if is_monochrome:
        u_gray = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2GRAY)
        l_std = float(np.std(lower_gray))
        lum_diff = abs(u_v - l_v)

        # In IR/monochrome, smooth fabric (low edge density) and texture drop are essential
        if edge_density < 0.040:
            if l_std < 18.0 or lum_diff > 40.0:
                score += 0.55
                if edge_density < 0.020:
                    score += 0.20 * (1.0 - (edge_density / 0.020))
                if l_std < 12.0:
                    score += 0.15 * (1.0 - (l_std / 12.0))
        elif edge_density > 0.06:
            # Clear lip, nostril, and mouth contours confirm unmasked face in IR
            score = 0.0
    else:
        # Normalized color distance in HSV space
        dh = min(abs(u_h - l_h), 180.0 - abs(u_h - l_h)) / 90.0
        ds = abs(u_s - l_s) / 255.0
        dv = abs(u_v - l_v) / 255.0
        color_dist = float(np.sqrt(dh * dh + ds * ds + dv * dv))

        # Lip presence (skin/lip redness)
        b, g, r = cv2.split(lower_crop.astype(np.float32))
        lip_mask = (r > (g + 15)) & (r > (b + 15)) & (r > 60)
        lip_ratio = float(np.sum(lip_mask)) / float(lower_crop.shape[0] * lower_crop.shape[1])

        # Distinctive mask color signatures:
        # 1. Surgical blue/cyan (75 <= H <= 130) or surgical green (35 <= H <= 75 with saturation)
        is_surgical_blue = (75 <= l_h <= 130 and l_s > 30 and (l_h - u_h > 35 or u_h < 30))
        is_surgical_green = (35 <= l_h <= 75 and l_s > 35 and (l_h - u_h > 20 or u_h < 25))
        is_surgical_tint = is_surgical_blue or is_surgical_green

        # 2. Neutral mask (white N95, gray, or black fabric)
        is_white_or_gray_mask = (l_s < 28 and (u_s - l_s) > 20 and l_v > 90)
        is_black_mask = (l_v < 60 and (u_v - l_v) > 40)
        is_neutral_mask = is_white_or_gray_mask or is_black_mask

        # Color discrepancy is the primary indicator of a physical face mask in RGB
        has_color_clue = is_surgical_tint or is_neutral_mask or (color_dist > 0.42)

        if is_surgical_tint:
            score += 0.55
        elif is_neutral_mask:
            score += 0.50
        elif color_dist > 0.42:
            score += min(0.40, (color_dist - 0.42) * 2.0)

        # Edge density and lip absence corroborate a mask only when a color discrepancy is present
        if has_color_clue:
            if edge_density < 0.040:
                score += 0.20 * (1.0 - (edge_density / 0.040))
            if lip_ratio < 0.05:
                score += 0.15 * (1.0 - (lip_ratio / 0.05))
        elif lip_ratio > 0.02:
            # Visible lips with matching skin tones confirm an unmasked face
            score = max(0.0, score - 0.20)

    mask_score = min(1.0, max(0.0, score))
    is_masked = mask_score >= score_threshold
    return is_masked, mask_score


def compute_eye_anchored_transform(landmarks: np.ndarray, dst_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute analytical 2D similarity transform mapping 2 eye landmarks to canonical ArcFace template coordinates.

    Uses closed-form complex division for maximum speed without external solvers or deprecation warnings.

    Args:
        landmarks: Array of shape (>=2, 2) containing left eye and right eye coordinates.
        dst_landmarks: Canonical destination coordinates (defaults to ArcFace 112x112 eye landmarks).

    Returns:
        2x3 affine transformation matrix M.
    """
    if dst_landmarks is None:
        # Canonical ArcFace 112x112 left and right eye coordinates
        dst_landmarks = np.array([[38.2946, 51.6963], [73.5318, 51.5014]], dtype=np.float32)

    p1, p2 = landmarks[0], landmarks[1]
    d1, d2 = dst_landmarks[0], dst_landmarks[1]

    dp = p2 - p1
    dd = d2 - d1

    denom = float(dp[0] * dp[0] + dp[1] * dp[1])
    if denom <= 1e-7:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # Complex division (dd[0] + i*dd[1]) / (dp[0] + i*dp[1])
    a = float((dd[0] * dp[0] + dd[1] * dp[1]) / denom)
    b = float((dd[1] * dp[0] - dd[0] * dp[1]) / denom)
    c = float(d1[0] - (a * p1[0] - b * p1[1]))
    d = float(d1[1] - (b * p1[0] + a * p1[1]))

    return np.array([[a, -b, c], [b, a, d]], dtype=np.float32)


def check_face_quality(
    frame: np.ndarray,
    face: Any,
    min_size: Tuple[int, int] = (config.REC_MIN_FACE_WIDTH, config.REC_MIN_FACE_HEIGHT),
    min_sharpness: float = config.REC_MIN_SHARPNESS,
    edge_margin: int = config.REC_EDGE_MARGIN_PX,
    is_low_light: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Verify facial crop quality, rejecting blur, poor lighting, extreme angles, and edge clipping.

    Handles both unmasked faces and faces wearing surgical/cloth masks safely across normal,
    dark, and night-vision (IR) lighting conditions.

    Args:
        frame: Original full-resolution image/frame.
        face: Detected InsightFace Face instance.
        min_size: (min_width, min_height) required.
        min_sharpness: Minimum Laplacian variance for blur rejection.
        edge_margin: Minimum margin from frame boundaries.
        is_low_light: Optional flag indicating dark or night-vision IR frame (auto-detected if None).

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    if frame is None or frame.size == 0 or face is None:
        return False, "Invalid image data"

    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    w = x2 - x1
    h = y2 - y1

    # 1. Edge boundary check (reject face clipped by frame borders)
    if x1 <= edge_margin or y1 <= edge_margin or x2 >= (w_img - edge_margin) or y2 >= (h_img - edge_margin):
        return False, "Face truncated at frame edge"

    # 2. Minimum dimension check
    min_w, min_h = min_size
    if w < min_w or h < min_h:
        return False, f"Face too small ({w}x{h} < {min_w}x{min_h})"

    # 3. Safe face crop
    x1_cl = max(0, x1)
    y1_cl = max(0, y1)
    x2_cl = min(w_img, x2)
    y2_cl = min(h_img, y2)
    crop = frame[y1_cl:y2_cl, x1_cl:x2_cl]
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return False, "Invalid face crop"

    # 4. Lighting: Brightness & Contrast (evaluated first as low brightness degrades Laplacian response)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Auto-detect low-light condition if not explicitly supplied
    if is_low_light is None:
        is_low_light = bool(getattr(face, "is_low_light", False))
        if not is_low_light:
            sample = frame[::8, ::8]
            sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if len(sample.shape) == 3 else sample
            is_low_light = float(np.mean(sample_gray)) < config.LOW_LIGHT_BRIGHTNESS_THRESHOLD

    # In low-light conditions, evaluate enhanced crop to verify quality after adaptive shadow lifting
    eval_crop = crop
    if is_low_light and mean_val < config.LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        crop_enh, _ = enhance_low_light_frame(crop)
        gray_enh = cv2.cvtColor(crop_enh, cv2.COLOR_BGR2GRAY) if len(crop_enh.shape) == 3 else crop_enh
        mean_val = float(np.mean(gray_enh))
        std_val = float(np.std(gray_enh))
        eval_crop = crop_enh

    min_brightness = config.REC_MIN_BRIGHTNESS_LOW_LIGHT if is_low_light else config.REC_MIN_BRIGHTNESS
    min_contrast = config.REC_MIN_CONTRAST_LOW_LIGHT if is_low_light else config.REC_MIN_CONTRAST

    if mean_val < min_brightness:
        return False, f"Face underexposed ({mean_val:.1f} < {min_brightness:.1f})"
    if mean_val > config.REC_MAX_BRIGHTNESS:
        return False, f"Face overexposed ({mean_val:.1f} > {config.REC_MAX_BRIGHTNESS:.1f})"
    if std_val < min_contrast:
        return False, f"Low contrast ({std_val:.1f} < {min_contrast:.1f})"

    # 5. Extreme pose check (yaw asymmetry from 5 facial landmarks evaluated before blur)
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist > 5.0:
            d_left = float(np.linalg.norm(nose - left_eye))
            d_right = float(np.linalg.norm(nose - right_eye))
            yaw_ratio = abs(d_left - d_right) / eye_dist
            if yaw_ratio > config.REC_MAX_YAW_RATIO:
                return False, f"Extreme head pose (yaw {yaw_ratio:.2f} > {config.REC_MAX_YAW_RATIO:.2f})"

    # 6. Blur / Sharpness check
    sharpness = compute_sharpness(eval_crop)
    if sharpness < min_sharpness:
        # If lower face is occluded by smooth mask cloth, evaluate sharpness on upper periocular region
        is_masked = getattr(face, "is_masked", False)
        if not is_masked:
            is_masked, _ = compute_mask_probability(frame, face)
        upper_sharpness = 0.0
        if is_masked and eval_crop.shape[0] >= 20:
            upper_crop = eval_crop[: int(eval_crop.shape[0] * 0.55), :]
            upper_sharpness = compute_sharpness(upper_crop)
        if upper_sharpness >= min_sharpness:
            sharpness = upper_sharpness
        else:
            return False, f"Face blurry (sharpness {sharpness:.1f} < {min_sharpness:.1f})"

    return True, "Quality OK"


def load_image_safely(file_path: Path) -> Optional[np.ndarray]:
    """Load an image from disk safely supporting Unicode / non-ASCII Windows paths.

    Args:
        file_path: Path to the target image file.

    Returns:
        Decoded BGR numpy array if successful, or None if unreadable/corrupted.
    """
    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        return None

    # Try standard cv2.imread first
    image = cv2.imread(str(path_obj))
    if image is not None and image.size > 0:
        return image

    # Fallback for Windows Unicode file paths
    try:
        with open(path_obj, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is not None and image.size > 0:
                return image
    except Exception:
        pass

    return None


# ============================================================================
# JSON Storage Operations (Local faces.json persistence)
# ============================================================================

def load_faces_from_json(file_path: Path = config.FACES_JSON_PATH) -> Dict[str, np.ndarray]:
    """Load existing registered identities and embeddings from a local JSON file.

    Performs comprehensive schema, dimension, and numerical validity checks.

    Args:
        file_path: Path to the local JSON file.

    Returns:
        Dictionary mapping person name -> 512-D normalized NumPy float32 array.
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()
            if not raw_content:
                return {}
            data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse '{file_path}': Invalid JSON syntax on line {exc.lineno}. "
            "Please check the file content before starting FaceMatch."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read '{file_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Malformed data in '{file_path}': Root element must be a JSON object/dictionary."
        )

    registry: Dict[str, np.ndarray] = {}

    for name, record in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid identity key in '{file_path}': Name must be a non-empty string.")

        clean_name = name.strip()

        if isinstance(record, dict) and "embedding" in record:
            raw_emb = record["embedding"]
        elif isinstance(record, list):
            raw_emb = record
        else:
            raise ValueError(
                f"Invalid record structure for '{clean_name}' in '{file_path}'. "
                "Expected a list of numbers or an object containing an 'embedding' list."
            )

        if not isinstance(raw_emb, list):
            raise ValueError(f"Embedding for '{clean_name}' in '{file_path}' must be a JSON list.")

        if len(raw_emb) != config.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch for '{clean_name}' in '{file_path}': "
                f"Got {len(raw_emb)} dimensions, expected {config.EMBEDDING_DIM}."
            )

        try:
            arr = np.array(raw_emb, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Non-numeric values found in embedding vector for '{clean_name}' in '{file_path}'."
            ) from exc

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Embedding for '{clean_name}' contains non-finite values (NaN/Inf).")

        norm = float(np.linalg.norm(arr))
        if norm <= 1e-7:
            raise ValueError(f"Embedding for '{clean_name}' has zero length (zero norm).")

        registry[clean_name] = (arr / norm).astype(np.float32)

    return registry


def save_faces_to_json(
    registry: Dict[str, np.ndarray],
    file_path: Path = config.FACES_JSON_PATH,
) -> None:
    """Save enrolled identities and embeddings to a local JSON file atomically.

    Uses temporary file with explicit sync and atomic rename to protect against
    file corruption during sudden power loss or process interruptions.

    Args:
        registry: Dictionary mapping names to embedding NumPy vectors.
        file_path: Destination JSON file path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data: Dict[str, Any] = {}
    for name, emb in registry.items():
        if not isinstance(emb, np.ndarray) or emb.shape[0] != config.EMBEDDING_DIM:
            raise ValueError(f"Invalid embedding array for identity '{name}'.")
        if not np.all(np.isfinite(emb)):
            raise ValueError(f"Embedding for '{name}' contains non-finite values (NaN/Inf).")

        serializable_data[name] = {
            "embedding": [round(float(v), 6) for v in emb.tolist()],
            "dim": int(emb.shape[0]),
        }

    temp_path = file_path.with_name(f"{file_path.name}.tmp_{os.getpid()}_{int(time.time()*1000)}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replacement with retry for transient Windows file-lock scenarios
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(temp_path, file_path)
                break
            except OSError as exc:
                if attempt == max_retries - 1:
                    raise exc
                time.sleep(0.05)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise IOError(f"Failed to safely write registered faces to '{file_path}': {exc}") from exc


# ============================================================================
# Core Face Recognition Engine (High Performance, Vectorized Matching)
# ============================================================================

class FaceEngine:
    """Manages face detection and feature extraction with high performance, thread safety, and zero redundant passes."""

    def __init__(self) -> None:
        """Initialize the FaceEngine instance (model loaded on first use)."""
        self._app: Optional[FaceAnalysis] = None
        self._is_ready: bool = False
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

        # Vectorized matching cache to avoid repeated matrix reconstructions
        self._matrix_cache: Optional[np.ndarray] = None
        self._matrix_names: List[str] = []
        self._registry_signature: Optional[Tuple[Any, ...]] = None

    def initialize(self) -> None:
        """Load InsightFace models once and prepare optimized inference session with ONNX Runtime."""
        with self._init_lock:
            if self._is_ready and self._app is not None:
                return

            if self._init_error is not None:
                raise RuntimeError(self._init_error)

            print("\n[FaceEngine] Initializing face recognition models...")
            print(f"[FaceEngine] Package: {config.MODEL_PACKAGE_NAME} (Fast & Low-Load)")
            print(f"[FaceEngine] Det Size: {config.REC_DETECTION_SIZE} (Rec) / {config.REG_DETECTION_SIZE} (Reg)")
            print(f"[FaceEngine] Threads: {config.ONNX_NUM_THREADS} (Benchmarked Optimal)")

            kwargs: Dict[str, Any] = {}
            if config.LOCAL_MODELS_DIR.exists():
                kwargs["root"] = str(config.BASE_DIR)

            # Configure ONNX Runtime CPU session options directly for graph optimization and thread pinning
            if ort is not None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = config.ONNX_NUM_THREADS
                sess_options.inter_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                kwargs["sess_options"] = sess_options

            try:
                self._app = FaceAnalysis(
                    name=config.MODEL_PACKAGE_NAME,
                    allowed_modules=config.ALLOWED_MODULES,
                    **kwargs,
                )
                # Pass det_thresh into prepare() so SCRFD detects down to tolerant recognition threshold
                self._app.prepare(
                    ctx_id=config.CTX_ID,
                    det_size=config.REC_DETECTION_SIZE,
                    det_thresh=min(config.REC_MIN_DET_SCORE, 0.25),
                )
                self._is_ready = True
                print("[FaceEngine] Models ready for real-time inference.\n")
            except Exception as exc:
                self._is_ready = False
                self._app = None
                self._init_error = (
                    f"Failed to initialize InsightFace models: {exc}\n"
                    "Ensure internet access for initial download or check model cache."
                )
                raise RuntimeError(self._init_error) from exc

    @property
    def is_ready(self) -> bool:
        """Return True if the model has been initialized and is ready."""
        return self._is_ready

    def ensure_ready(self) -> None:
        """Ensure model is loaded on-demand before processing frames."""
        if not self._is_ready or self._app is None:
            self.initialize()

    def _set_detector_input_size(self, target_size: Tuple[int, int]) -> None:
        """Dynamically configure SCRFD detector input resolution without reloading models."""
        if self._app is not None and "detection" in self._app.models:
            det_model = self._app.models["detection"]
            if getattr(det_model, "input_size", None) != target_size:
                det_model.input_size = target_size
                det_model.input_sizes = [target_size]

    def detect_faces(
        self,
        frame: np.ndarray,
        scale: float = 1.0,
        min_score: float = config.REC_MIN_DET_SCORE,
        min_size: Tuple[int, int] = (config.REC_MIN_FACE_WIDTH, config.REC_MIN_FACE_HEIGHT),
        det_size: Optional[Tuple[int, int]] = None,
    ) -> List[Any]:
        """Detect faces in a BGR frame with configurable quality thresholds.

        Handles normal daytime RGB, underexposed dark RGB, and monochrome/IR nighttime frames.
        When scale < 1.0, detection runs on the downscaled frame for maximum speed,
        while facial landmark coordinates are scaled back so ArcFace recognition
        extracts the feature embedding directly from the original full-resolution frame.

        Args:
            frame: BGR or grayscale image (from webcam, recorded video, or RTSP stream).
            scale: Downscaling factor for faster inference (e.g. 0.5).
            min_score: Minimum detection confidence threshold to accept.
            min_size: Minimum (width, height) bounding box size to accept.
            det_size: Optional detector input resolution (defaults to config.REC_DETECTION_SIZE).

        Returns:
            List of detected Face objects meeting the specified score and size.
        """
        self.ensure_ready()
        assert self._app is not None

        if frame is None or frame.size == 0:
            return []

        # Ensure 3-channel BGR for InsightFace SCRFD and ArcFace (e.g. grayscale IR camera frames)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Detect and enhance underexposed low-light/night-vision frames
        proc_frame, was_enhanced = enhance_low_light_frame(frame)

        # Configure detector resolution (defaulting to REC_DETECTION_SIZE for real-time throughput)
        target_size = det_size if det_size is not None else config.REC_DETECTION_SIZE
        self._set_detector_input_size(target_size)

        min_w, min_h = min_size
        rec_model = self._app.models.get("recognition")

        if scale < 1.0:
            small_frame = cv2.resize(proc_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            bboxes, kpss = self._app.det_model.detect(small_frame)
            if bboxes.shape[0] == 0:
                return []

            inv_scale = 1.0 / scale
            scaled_bboxes = bboxes.copy()
            scaled_bboxes[:, 0:4] *= inv_scale
            scaled_kpss = kpss.copy() * inv_scale if kpss is not None else None

            filtered_faces: List[Face] = []
            for i in range(scaled_bboxes.shape[0]):
                det_score = float(scaled_bboxes[i, 4])
                if det_score < min_score:
                    continue

                bbox = scaled_bboxes[i, 0:4]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < min_w or h < min_h:
                    continue

                kps = scaled_kpss[i] if scaled_kpss is not None else None
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                face.is_low_light = was_enhanced

                # Extract recognition embedding from processed frame using mapped landmarks
                if rec_model is not None:
                    rec_model.get(proc_frame, face)

                filtered_faces.append(face)

            if config.MASK_DETECTION_ENABLED:
                for f in filtered_faces:
                    is_m, m_sc = compute_mask_probability(proc_frame, f)
                    f.is_masked = is_m
                    f.mask_score = m_sc
                    if is_m:
                        self._refine_masked_face_embedding(proc_frame, f)
            else:
                for f in filtered_faces:
                    f.is_masked = False
                    f.mask_score = 0.0

            return filtered_faces

        # Full resolution inference
        raw_faces = self._app.get(proc_frame)
        filtered_faces = []
        for face in raw_faces:
            det_score = float(face.det_score)
            if det_score < min_score:
                continue

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            width = x2 - x1
            height = y2 - y1

            if width >= min_w and height >= min_h:
                face.is_low_light = was_enhanced
                filtered_faces.append(face)

        if config.MASK_DETECTION_ENABLED:
            for f in filtered_faces:
                is_m, m_sc = compute_mask_probability(proc_frame, f)
                f.is_masked = is_m
                f.mask_score = m_sc
                if is_m:
                    self._refine_masked_face_embedding(proc_frame, f)
        else:
            for f in filtered_faces:
                f.is_masked = False
                f.mask_score = 0.0

        return filtered_faces

    def _refine_masked_face_embedding(self, frame: np.ndarray, face: Any) -> None:
        """Refine feature embedding for masked faces using eye-anchored canonical alignment.

        When the lower face (nose/mouth/chin) is covered by a mask, standard 5-point alignment
        is distorted by occluded mouth landmark guesses. This extracts the canonical 112x112
        crop aligned strictly on the visible left and right eye centers, then fuses this
        periocular feature representation with the base embedding.
        """
        rec_model = self._app.models.get("recognition") if self._app is not None else None
        if rec_model is None or getattr(face, "kps", None) is None or len(face.kps) < 2:
            return

        M_eye = compute_eye_anchored_transform(face.kps)
        eye_crop = cv2.warpAffine(frame, M_eye, (112, 112), borderValue=0.0)
        eye_feat = rec_model.get_feat(eye_crop).flatten()
        eye_norm = float(np.linalg.norm(eye_feat))
        if eye_norm <= 1e-7:
            return

        eye_emb = (eye_feat / eye_norm).astype(np.float32)
        curr_emb = getattr(face, "normed_embedding", None)
        if curr_emb is None and getattr(face, "embedding", None) is not None:
            c_norm = float(np.linalg.norm(face.embedding))
            curr_emb = (face.embedding / c_norm).astype(np.float32) if c_norm > 1e-7 else None

        if curr_emb is not None:
            fused = curr_emb + eye_emb
            f_norm = float(np.linalg.norm(fused))
            if f_norm > 1e-7:
                face.embedding = (fused / f_norm).astype(np.float32)
        else:
            face.embedding = eye_emb

    def detect_face_mask(self, frame: np.ndarray, face: Any) -> Tuple[bool, float]:
        """Detect whether a face in the frame is wearing a face mask (surgical, N95, cloth, etc.).

        Args:
            frame: Full BGR image frame.
            face: Detected Face instance.

        Returns:
            Tuple of (is_masked: bool, mask_score: float).
        """
        if hasattr(face, "is_masked") and hasattr(face, "mask_score"):
            return face.is_masked, face.mask_score
        return compute_mask_probability(frame, face)

    def extract_single_face_for_registration(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Tuple[int, int, int, int]]]:
        """Validate and extract the face embedding strictly for enrollment in a single efficient pass.

        Enforces:
        - Exactly one face in the frame.
        - Face detection confidence >= REG_MIN_DET_SCORE (0.65).
        - Face size >= REG_MIN_FACE_WIDTH x REG_MIN_FACE_HEIGHT (80x80 px).
        - Face sharpness >= REG_MIN_SHARPNESS (Laplacian blur rejection).

        Args:
            frame: Full-resolution BGR frame.

        Returns:
            Tuple of (embedding, error_message, bbox).
        """
        self.ensure_ready()
        assert self._app is not None

        if frame is None or frame.size == 0:
            return None, "Invalid image data provided.", None

        # Run detection ONCE at base threshold with REG_DETECTION_SIZE (640x640) for precise alignment
        try:
            raw_faces = self.detect_faces(
                frame,
                scale=1.0,
                min_score=0.25,
                min_size=(20, 20),
                det_size=config.REG_DETECTION_SIZE,
            )

            if len(raw_faces) == 0:
                return None, "No face detected. Please face the camera directly in good lighting.", None

            # Check for face mask first on detected face candidate(s), reusing cached mask detection if present
            if config.MASK_DETECTION_ENABLED:
                for f in raw_faces:
                    is_m = getattr(f, "is_masked", None)
                    if is_m is None:
                        is_m, _ = compute_mask_probability(frame, f)
                    if is_m:
                        bx = [int(v) for v in f.bbox]
                        return (
                            None,
                            "Face mask detected. Face registration requires an unmasked face to ensure reliable biometric identification.",
                            (bx[0], bx[1], bx[2], bx[3]),
                        )

            # Check how many qualify under strict registration criteria
            valid_candidates = []
            for f in raw_faces:
                w = int(f.bbox[2] - f.bbox[0])
                h = int(f.bbox[3] - f.bbox[1])
                score = float(f.det_score)
                if w >= config.REG_MIN_FACE_WIDTH and h >= config.REG_MIN_FACE_HEIGHT and score >= config.REG_MIN_DET_SCORE:
                    valid_candidates.append(f)

            if len(valid_candidates) == 0:
                # Diagnose reason on first candidate without re-running detection
                lead = raw_faces[0]
                lw = int(lead.bbox[2] - lead.bbox[0])
                lh = int(lead.bbox[3] - lead.bbox[1])
                score = float(lead.det_score)
                if lw < config.REG_MIN_FACE_WIDTH or lh < config.REG_MIN_FACE_HEIGHT:
                    return None, f"Face is too small ({lw}x{lh}px). Registration requires >= {config.REG_MIN_FACE_WIDTH}x{config.REG_MIN_FACE_HEIGHT}px.", None
                if score < config.REG_MIN_DET_SCORE:
                    return None, f"Face confidence too low ({score:.2f} < {config.REG_MIN_DET_SCORE:.2f}). Improve lighting.", None
                return None, "Face quality insufficient for registration. Ensure clear front-facing lighting.", None

            if len(valid_candidates) > 1:
                return (
                    None,
                    f"Multiple faces detected ({len(valid_candidates)}). Ensure only one person is in the frame.",
                    None,
                )

            face = valid_candidates[0]
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h_img, w_img = frame.shape[:2]

            # 1. Edge boundary check (reject face clipped by frame borders)
            edge_margin = config.REG_EDGE_MARGIN_PX
            if x1 <= edge_margin or y1 <= edge_margin or x2 >= (w_img - edge_margin) or y2 >= (h_img - edge_margin):
                return (
                    None,
                    "Face is partially outside the frame. Please center your face in the camera.",
                    (x1, y1, x2, y2),
                )

            x1_cl = max(0, x1)
            y1_cl = max(0, y1)
            x2_cl = min(w_img, x2)
            y2_cl = min(h_img, y2)

            face_crop = frame[y1_cl:y2_cl, x1_cl:x2_cl]
            if face_crop is None or face_crop.size == 0 or face_crop.shape[0] < 5 or face_crop.shape[1] < 5:
                return (
                    None,
                    "Invalid face crop. Ensure your face is clearly visible in the camera.",
                    (x1, y1, x2, y2),
                )

            # 2. Lighting: Brightness & Contrast (evaluated first as low brightness degrades Laplacian response)
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            mean_val = float(np.mean(gray))
            std_val = float(np.std(gray))

            if mean_val < config.REG_MIN_BRIGHTNESS:
                return (
                    None,
                    f"Face image is too dark (brightness: {mean_val:.1f} < {config.REG_MIN_BRIGHTNESS:.1f}). Please improve lighting.",
                    (x1, y1, x2, y2),
                )
            if mean_val > config.REG_MAX_BRIGHTNESS:
                return (
                    None,
                    f"Face image is overexposed (brightness: {mean_val:.1f} > {config.REG_MAX_BRIGHTNESS:.1f}). Reduce glare or harsh backlighting.",
                    (x1, y1, x2, y2),
                )
            if std_val < config.REG_MIN_CONTRAST:
                return (
                    None,
                    f"Face image has low contrast (contrast: {std_val:.1f} < {config.REG_MIN_CONTRAST:.1f}). Ensure clear and balanced lighting.",
                    (x1, y1, x2, y2),
                )

            # 3. Head pose & angle check (evaluated before blur so turned heads are not falsely reported as blurry)
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 5:
                left_eye = kps[0]
                right_eye = kps[1]
                nose = kps[2]
                eye_dist = float(np.linalg.norm(right_eye - left_eye))
                if eye_dist > 5.0:
                    d_left = float(np.linalg.norm(nose - left_eye))
                    d_right = float(np.linalg.norm(nose - right_eye))
                    yaw_ratio = abs(d_left - d_right) / eye_dist
                    if yaw_ratio > config.REG_MAX_YAW_RATIO:
                        return (
                            None,
                            f"Head is turned at an angle (yaw: {yaw_ratio:.2f} > {config.REG_MAX_YAW_RATIO:.2f}). Please face the camera directly.",
                            (x1, y1, x2, y2),
                        )
                    # Vertical head tilt / roll check
                    dy = abs(float(right_eye[1] - left_eye[1]))
                    tilt_ratio = dy / eye_dist
                    if tilt_ratio > config.REG_MAX_ROLL_TILT:
                        return (
                            None,
                            f"Head is tilted sideways. Please keep your head upright and level.",
                            (x1, y1, x2, y2),
                        )

            # 4. Blur / sharpness check: assess actual image detail on face crop and facial feature region
            crop_sharpness = compute_sharpness(face_crop)
            feat_sharpness = 0.0
            if kps is not None and len(kps) >= 5:
                kx1 = max(0, int(np.min(kps[:, 0])) - 8)
                ky1 = max(0, int(np.min(kps[:, 1])) - 8)
                kx2 = min(w_img, int(np.max(kps[:, 0])) + 8)
                ky2 = min(h_img, int(np.max(kps[:, 1])) + 8)
                if kx2 > kx1 + 10 and ky2 > ky1 + 10:
                    feat_crop = frame[ky1:ky2, kx1:kx2]
                    feat_sharpness = compute_sharpness(feat_crop)

            effective_sharpness = max(crop_sharpness, feat_sharpness)
            if effective_sharpness < config.REG_MIN_SHARPNESS:
                return (
                    None,
                    f"Face image is too blurry (sharpness: {effective_sharpness:.1f} < {config.REG_MIN_SHARPNESS:.1f}). Ensure a steady camera and good lighting.",
                    (x1, y1, x2, y2),
                )

            # 5. Enforce that the registered subject is not wearing a mask
            if config.MASK_DETECTION_ENABLED:
                is_masked = getattr(face, "is_masked", None)
                if is_masked is None:
                    is_masked, _ = compute_mask_probability(frame, face)
                if is_masked:
                    return (
                        None,
                        "Face mask detected. Face registration requires an unmasked face to ensure reliable biometric identification.",
                        (x1, y1, x2, y2),
                    )

            # Extract normalized embedding
            embedding = face.normed_embedding
            if embedding is None:
                raw_emb = face.embedding
                norm = float(np.linalg.norm(raw_emb))
                if norm <= 1e-7:
                    return None, "Failed to compute valid face embedding (zero norm).", (x1, y1, x2, y2)
                embedding = raw_emb / norm

            return embedding.astype(np.float32), None, (x1, y1, x2, y2)
        finally:
            # Restore real-time detector size for low-load recognition
            self._set_detector_input_size(config.REC_DETECTION_SIZE)

    def set_registry(self, registry: Dict[str, np.ndarray]) -> None:
        """Explicitly update and sync the in-memory vectorized matching matrix cache."""
        self._update_matrix_cache(registry, force=True)

    def invalidate_cache(self) -> None:
        """Invalidate the internal vectorized matrix cache."""
        self._matrix_cache = None
        self._matrix_names = []
        self._registry_signature = None

    def _update_matrix_cache(self, registry: Dict[str, np.ndarray], force: bool = False) -> None:
        """Build or refresh vectorized 2D matrix cache for high-speed batch matching."""
        # Signature tracks keys and array object references to detect in-place updates/overwrites
        current_sig = tuple(sorted((k, id(v)) for k, v in registry.items()))
        if force or self._matrix_cache is None or self._registry_signature != current_sig:
            self._matrix_names = list(registry.keys())
            if self._matrix_names:
                # Shape (N, 512)
                self._matrix_cache = np.stack([registry[name] for name in self._matrix_names]).astype(np.float32)
            else:
                self._matrix_cache = None
            self._registry_signature = current_sig

    def match_face(
        self,
        embedding: np.ndarray,
        registry: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
        is_masked: bool = False,
    ) -> Tuple[str, float]:
        """Compare a face embedding against all enrolled identities using vectorized BLAS dot product.

        Supports both unmasked faces (SIMILARITY_THRESHOLD = 0.50) and masked faces
        (MASKED_SIMILARITY_THRESHOLD = 0.38) with configurable thresholds.

        Achieves 60x+ faster comparison than Python loops.
        Never alters registry or writes to storage. Returns the recognized identity
        only when cosine similarity meets the threshold; otherwise 'Unknown Person'.
        """
        if threshold is None:
            threshold = config.MASKED_SIMILARITY_THRESHOLD if is_masked else config.SIMILARITY_THRESHOLD

        ident, score, _ = self.match_face_detailed(
            embedding=embedding,
            registry=registry,
            threshold=threshold,
            min_margin=config.MASKED_MATCH_AMBIGUITY_MARGIN if is_masked else config.MATCH_AMBIGUITY_MARGIN,
            is_masked=is_masked,
        )
        return ident, score

    def match_face_detailed(
        self,
        embedding: np.ndarray,
        registry: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
        is_masked: bool = False,
    ) -> Tuple[str, float, bool]:
        """Detailed vectorized matching with ambiguity rejection and masked-face support.

        Args:
            embedding: Query face embedding vector.
            registry: Registered identities dictionary.
            threshold: Cosine similarity threshold (defaults to MASKED_SIMILARITY_THRESHOLD if is_masked else SIMILARITY_THRESHOLD).
            min_margin: Minimum margin between top-1 and top-2 candidate to prevent identity ambiguity.
            is_masked: True if the face has been detected as wearing a face mask.

        Returns:
            Tuple of (identity_name, similarity_score, is_ambiguous).
        """
        if threshold is None:
            threshold = config.MASKED_SIMILARITY_THRESHOLD if is_masked else config.SIMILARITY_THRESHOLD
        if min_margin is None:
            min_margin = config.MASKED_MATCH_AMBIGUITY_MARGIN if is_masked else config.MATCH_AMBIGUITY_MARGIN

        unknown_label = "Unknown Person"

        if not registry:
            return unknown_label, 0.0, False

        self._update_matrix_cache(registry)
        if self._matrix_cache is None or len(self._matrix_names) == 0:
            return unknown_label, 0.0, False

        # Ensure query embedding is unit-normalized for true cosine similarity
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-7:
            return unknown_label, 0.0, False
        normed_emb = (embedding / norm).astype(np.float32) if abs(norm - 1.0) > 1e-4 else embedding.astype(np.float32)

        # Vectorized dot product against all N enrolled unit vectors in one fast call
        scores = np.dot(self._matrix_cache, normed_emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return unknown_label, max(0.0, best_score), False

        # Check for ambiguity if multiple identities are enrolled
        if len(self._matrix_names) > 1:
            if len(self._matrix_names) == 2:
                top1_score = best_score
                other_idx = 1 if best_idx == 0 else 0
                top2_score = float(scores[other_idx])
            else:
                top2_idx = np.argpartition(scores, -2)[-2:]
                top2_idx = top2_idx[np.argsort(scores[top2_idx])[::-1]]
                top1_score = float(scores[top2_idx[0]])
                top2_score = float(scores[top2_idx[1]])

            if top2_score >= (threshold - 0.05) and (top1_score - top2_score) < min_margin:
                # Top match cannot be decisively separated from the second match
                return unknown_label, top1_score, True

        return self._matrix_names[best_idx], best_score, False

    def match_faces_batch(
        self,
        embeddings: List[np.ndarray],
        registry: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
        is_masked_list: Optional[List[bool]] = None,
    ) -> List[Tuple[str, float, bool]]:
        """Match multiple detected face embeddings in a single optimized BLAS matrix multiplication.

        Args:
            embeddings: List of 512-D face embedding vectors.
            registry: Registered identities dictionary.
            threshold: Base cosine similarity threshold (or None to use per-face masked/unmasked config).
            min_margin: Margin required between top-1 and top-2 candidate.
            is_masked_list: Optional list indicating mask status for each embedding.

        Returns:
            List of (identity_name, similarity_score, is_ambiguous) for each embedding.
        """
        if not embeddings or not registry:
            return [("Unknown Person", 0.0, False) for _ in embeddings]

        self._update_matrix_cache(registry)
        if self._matrix_cache is None or len(self._matrix_names) == 0:
            return [("Unknown Person", 0.0, False) for _ in embeddings]

        # Stack and normalize query embeddings: shape (M, 512)
        query_matrix = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        norms[norms <= 1e-7] = 1.0
        query_matrix /= norms

        # Single BLAS matrix product: (M, 512) x (512, N) -> (M, N)
        scores_matrix = np.dot(query_matrix, self._matrix_cache.T)

        results = []
        for i in range(len(embeddings)):
            row_scores = scores_matrix[i]
            best_idx = int(np.argmax(row_scores))
            best_score = float(row_scores[best_idx])

            is_masked = is_masked_list[i] if (is_masked_list and i < len(is_masked_list)) else False
            eff_thresh = threshold if threshold is not None else (
                config.MASKED_SIMILARITY_THRESHOLD if is_masked else config.SIMILARITY_THRESHOLD
            )
            eff_margin = min_margin if min_margin is not None else (
                config.MASKED_MATCH_AMBIGUITY_MARGIN if is_masked else config.MATCH_AMBIGUITY_MARGIN
            )
            unknown_label = "Unknown Person"

            if best_score < eff_thresh:
                results.append((unknown_label, max(0.0, best_score), False))
                continue

            is_ambiguous = False
            if len(self._matrix_names) > 1:
                if len(self._matrix_names) == 2:
                    top1_score = best_score
                    other_idx = 1 if best_idx == 0 else 0
                    top2_score = float(row_scores[other_idx])
                else:
                    top2_idx = np.argpartition(row_scores, -2)[-2:]
                    top2_idx = top2_idx[np.argsort(row_scores[top2_idx])[::-1]]
                    top1_score = float(row_scores[top2_idx[0]])
                    top2_score = float(row_scores[top2_idx[1]])

                if top2_score >= (eff_thresh - 0.05) and (top1_score - top2_score) < eff_margin:
                    is_ambiguous = True

            if is_ambiguous:
                results.append((unknown_label, best_score, True))
            else:
                results.append((self._matrix_names[best_idx], best_score, False))

        return results
