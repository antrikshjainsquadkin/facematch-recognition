"""FaceMatch: Fast, Low-Load Face Recognition with Local JSON Storage.

Supports:
1. Register Face (Live Webcam with strict quality checks).
2. Register Face from Image (Upload image with strict quality checks).
3. Recognize Face (Live Webcam with tolerant recognition for low-light/small faces).
4. Recognize Face from Video File (Processes recorded/low-quality video files).
5. List Registered People (from faces.json).
6. Delete Registered Person (from faces.json).
7. Exit.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import config
from face_engine import FaceEngine, load_faces_from_json, save_faces_to_json, load_image_safely


def open_camera(camera_index: int = config.CAMERA_INDEX) -> Optional[cv2.VideoCapture]:
    """Open the webcam with resilient backend detection for both native and sandboxed OS.

    Tries DirectShow first on Windows, then gracefully falls back to default
    if running in a virtual sandbox or container.

    Args:
        camera_index: System camera index.

    Returns:
        cv2.VideoCapture object if successfully opened, otherwise None.
    """
    cap = None

    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    except Exception:
        cap = None

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[Error] Could not open webcam at index {camera_index}.")
            print("[Error] Please check camera permissions or USB connections.")
            return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def register_face(engine: FaceEngine, registry: Dict[str, np.ndarray]) -> None:
    """Capture a live face from the webcam, enforce strict quality, and persist to faces.json.

    Args:
        engine: FaceEngine instance.
        registry: In-memory dictionary mapping names to embedding vectors.
    """
    print("\n--- Register a New Face ---")
    raw_name = input("Enter person's name or ID: ").strip()

    if not raw_name:
        print("[Error] Name cannot be empty. Registration aborted.")
        return

    if raw_name in registry:
        print(f"[Warning] '{raw_name}' is already registered in '{config.FACES_JSON_PATH.name}'.")
        choice = input("Do you want to overwrite this registration? (y/n): ").strip().lower()
        if choice not in ("y", "yes"):
            print("Registration cancelled.")
            return

    engine.ensure_ready()

    cap = open_camera()
    if cap is None:
        return

    print("\n[Camera] Webcam started.")
    print("Instructions:")
    print(" - Look straight into the camera in good lighting.")
    print(" - Press [SPACE] to capture and register your face.")
    print(" - Press [Q] to cancel registration.\n")

    status_message = "Position your face and press [SPACE] to capture"
    status_color = (0, 255, 255)  # Yellow
    frame_count = 0
    cached_face_boxes: List[Tuple[int, int, int, int]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Error] Failed to read frame from webcam.")
                break

            display_frame = cv2.flip(frame, 1)
            h, w = display_frame.shape[:2]
            frame_count += 1

            # Preview face alignment guide (uses tolerant detection to help user align)
            if frame_count % config.PROCESS_EVERY_N_FRAMES == 0 or not cached_face_boxes:
                detected = engine.detect_faces(
                    display_frame,
                    scale=config.INFERENCE_SCALE,
                    min_score=config.REC_MIN_DET_SCORE,
                    min_size=(config.REC_MIN_FACE_WIDTH, config.REC_MIN_FACE_HEIGHT),
                )
                cached_face_boxes = [[int(v) for v in f.bbox] for f in detected]

            for x1, y1, x2, y2 in cached_face_boxes:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # Draw top status banner
            cv2.rectangle(display_frame, (0, 0), (w, 50), (30, 30, 30), -1)
            cv2.putText(
                display_frame,
                f"Registering: {raw_name}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                status_message,
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
                cv2.LINE_AA,
            )

            # Draw bottom guide
            cv2.rectangle(display_frame, (0, h - 30), (w, h), (30, 30, 30), -1)
            cv2.putText(
                display_frame,
                "[SPACE] Capture | [Q] Cancel",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(config.WINDOW_TITLE, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("Registration cancelled by user.")
                break

            if key == 32:  # SPACE to capture
                # Extract face with strict quality checks (clear, large, sharp, single face)
                emb, err, bbox = engine.extract_single_face_for_registration(display_frame)

                if err is not None:
                    status_message = f"Failed: {err}"
                    status_color = (0, 0, 255)  # Red
                    print(f"[Registration Rejected] {err}")
                    continue

                if emb is not None:
                    registry[raw_name] = emb
                    try:
                        save_faces_to_json(registry, config.FACES_JSON_PATH)
                        engine.set_registry(registry)
                        print(f"\n[Success] '{raw_name}' registered and saved to '{config.FACES_JSON_PATH.name}'!")
                    except Exception as exc:
                        print(f"\n[Error] Failed to persist data: {exc}")
                        registry.pop(raw_name, None)
                        break

                    if bbox:
                        bx1, by1, bx2, by2 = bbox
                        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                    cv2.putText(
                        display_frame,
                        f"REGISTERED: {raw_name}",
                        (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(config.WINDOW_TITLE, display_frame)
                    cv2.waitKey(1000)
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def pick_image_file() -> Optional[str]:
    """Open a GUI file picker to select an image file.

    Falls back to terminal input if GUI file dialog is unavailable or fails.

    Returns:
        Selected file path string, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Face Image to Register",
            filetypes=[
                ("Supported Images (*.jpg, *.jpeg, *.png, *.webp)", "*.jpg;*.jpeg;*.png;*.webp"),
                ("JPEG files (*.jpg, *.jpeg)", "*.jpg;*.jpeg"),
                ("PNG files (*.png)", "*.png"),
                ("WEBP files (*.webp)", "*.webp"),
                ("All files (*.*)", "*.*"),
            ],
        )
        root.destroy()
        if file_path and file_path.strip():
            return file_path.strip()
        return None
    except Exception as exc:
        print(f"[Notice] GUI file picker could not open: {exc}")
        raw = input("Enter path to image file (or 'c' to cancel): ").strip().strip('"').strip("'")
        if raw.lower() in ("c", "cancel", ""):
            return None
        return raw


def register_face_from_image(engine: FaceEngine, registry: Dict[str, np.ndarray]) -> None:
    """Register a new face by selecting an image file from the computer.

    Applies strict quality criteria:
    - Supported formats: JPG, JPEG, PNG, WEBP.
    - Exactly one face required.
    - Confidence score >= REG_MIN_DET_SCORE (0.65).
    - Face dimensions >= REG_MIN_FACE_WIDTH x REG_MIN_FACE_HEIGHT (80x80px).
    - Image sharpness >= REG_MIN_SHARPNESS (40.0) to reject blurry images.
    - Valid non-empty name with duplicate handling.
    - Saves embedding to faces.json upon success.

    Args:
        engine: FaceEngine instance.
        registry: In-memory dictionary mapping names to embedding vectors.
    """
    print("\n==========================================")
    print("         FaceMatch Recognition           ")
    print("==========================================")
    print("Select an image to register...")

    file_path = pick_image_file()
    if not file_path:
        print("Registration cancelled: No image was selected.")
        return

    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        print(f"[Error] File not found: '{file_path}'")
        return

    # Check image format
    ext = path_obj.suffix.lower()
    if ext not in config.SUPPORTED_IMAGE_EXTENSIONS:
        print(
            f"[Error] Unsupported image format '{ext}'. "
            f"Supported formats: {', '.join(config.SUPPORTED_IMAGE_EXTENSIONS)}"
        )
        return

    # Load image safely supporting Unicode / non-ASCII Windows paths
    image = load_image_safely(path_obj)
    if image is None or image.size == 0:
        print(f"[Error] Could not read image file '{path_obj.name}'. The file may be corrupted, empty, or unreadable.")
        return

    print(f"\nImage selected: {path_obj.name}")

    engine.ensure_ready()

    # Apply strict registration checks (single face, confidence >= 0.65, size >= 80x80, sharpness >= 40.0)
    emb, err, bbox = engine.extract_single_face_for_registration(image)

    if err is not None:
        print(f"\n[Registration Rejected] {err}")
        return

    if emb is None:
        print("\n[Registration Rejected] Failed to extract a valid face embedding.")
        return

    print("Face detected successfully.")
    print("Face quality validation passed.")

    # Prompt user for name after face passes validation
    while True:
        raw_name = input("\nEnter person's name: ").strip()
        if not raw_name:
            print("[Error] Name cannot be empty or whitespace only. Please enter a valid name.")
            continue
        break

    # Existing duplicate name handling
    if raw_name in registry:
        print(f"[Warning] '{raw_name}' is already registered in '{config.FACES_JSON_PATH.name}'.")
        choice = input("Do you want to overwrite this registration? (y/n): ").strip().lower()
        if choice not in ("y", "yes"):
            print("Registration cancelled.")
            return

    # Save embedding using safe storage logic
    registry[raw_name] = emb
    try:
        save_faces_to_json(registry, config.FACES_JSON_PATH)
        engine.set_registry(registry)
        print("\nRegistration successful!")
        print(f"Person: {raw_name}")
        print(f"Face embedding saved to {config.FACES_JSON_PATH.name}.\n")
    except Exception as exc:
        print(f"\n[Error] Failed to persist data to {config.FACES_JSON_PATH.name}: {exc}")
        registry.pop(raw_name, None)
        return

    # Visual confirmation preview
    if bbox:
        try:
            display_img = image.copy()
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(display_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

            label_text = f"REGISTERED: {raw_name}"
            cv2.putText(
                display_img,
                label_text,
                (max(10, bx1), max(30, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Resize window preview if original image is extremely large
            dh, dw = display_img.shape[:2]
            max_dim = 900
            if max(dh, dw) > max_dim:
                scale_fac = max_dim / float(max(dh, dw))
                display_img = cv2.resize(display_img, (int(dw * scale_fac), int(dh * scale_fac)))

            cv2.imshow(f"FaceMatch - {raw_name}", display_img)
            cv2.waitKey(1200)
            cv2.destroyAllWindows()
        except Exception:
            pass


def draw_face_annotation(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    box_color: Tuple[int, int, int],
    label_text: str,
) -> None:
    """Draw a face bounding box and label header on the image frame."""
    x1, y1, x2, y2 = bbox
    w = frame.shape[1]
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tag_y1 = max(0, y1 - label_size[1] - 8)
    tag_y2 = y1
    tag_x2 = min(w, x1 + label_size[0] + 6)
    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)

    cv2.putText(
        frame,
        label_text,
        (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def identify_face(
    engine: FaceEngine,
    face: Any,
    registry: Dict[str, np.ndarray],
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int], str, str]:
    """Compare a detected face against registered identities and format display properties.

    Returns:
        Tuple of ((x1, y1, x2, y2), box_color, label_text, identity_name).
    """
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    embedding = face.normed_embedding
    if embedding is None:
        raw_emb = face.embedding
        norm = float(np.linalg.norm(raw_emb))
        if norm > 0:
            embedding = (raw_emb / norm).astype(np.float32)

    if embedding is not None:
        identity, score = engine.match_face(embedding, registry)
    else:
        identity, score = "Unknown Person", 0.0

    if identity != "Unknown Person":
        box_color = (0, 255, 0)      # Green
        label_text = f"{identity} (Sim: {score:.2f})"
    else:
        box_color = (0, 0, 255)      # Red
        label_text = f"Unknown Person (Sim: {score:.2f})"

    return (x1, y1, x2, y2), box_color, label_text, identity


def recognize_faces(engine: FaceEngine, registry: Dict[str, np.ndarray]) -> None:
    """Run real-time facial recognition on webcam with tolerant detection for lower-quality inputs.

    Args:
        engine: FaceEngine instance.
        registry: In-memory dictionary mapping names to embedding vectors.
    """
    if not registry:
        print("\n[Notice] No registered people found.")
        print(f"Please register at least one face (Option 1) so it is saved to '{config.FACES_JSON_PATH.name}'.")
        return

    engine.ensure_ready()

    cap = open_camera()
    if cap is None:
        return

    print("\n[Camera] Live recognition started.")
    print("Press [Q] in the camera window to stop recognition and return to the menu.\n")

    frame_count = 0
    cached_results: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int], str]] = []

    # Real-time processing FPS tracker
    t_prev_frame = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Error] Failed to read frame from webcam.")
                break

            display_frame = cv2.flip(frame, 1)
            h, w = display_frame.shape[:2]
            frame_count += 1

            # Tolerant recognition detection (detects smaller/low-light faces)
            if frame_count % config.PROCESS_EVERY_N_FRAMES == 0 or not cached_results:
                new_results = []
                faces = engine.detect_faces(
                    display_frame,
                    scale=config.INFERENCE_SCALE,
                    min_score=config.REC_MIN_DET_SCORE,
                    min_size=(config.REC_MIN_FACE_WIDTH, config.REC_MIN_FACE_HEIGHT),
                )

                for face in faces:
                    bbox, box_color, label_text, _ = identify_face(engine, face, registry)
                    new_results.append((bbox, box_color, label_text))

                cached_results = new_results

            for bbox, box_color, label_text in cached_results:
                draw_face_annotation(display_frame, bbox, box_color, label_text)

            # Measure frame processing FPS
            t_now = time.perf_counter()
            dt_frame = t_now - t_prev_frame
            t_prev_frame = t_now
            inst_fps = 1.0 / dt_frame if dt_frame > 0 else 0.0
            fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps if fps_smooth > 0 else inst_fps

            # Draw top info banner with actual processing FPS
            cv2.rectangle(display_frame, (0, 0), (w, 32), (30, 30, 30), -1)
            cv2.putText(
                display_frame,
                f"FaceMatch Live | Enrolled: {len(registry)} | Proc: {fps_smooth:.1f} FPS | Press [Q] to Exit",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(config.WINDOW_TITLE, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Recognition session stopped.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def recognize_from_video_file(engine: FaceEngine, registry: Dict[str, np.ndarray]) -> None:
    """Run face recognition on a recorded video file with tolerant settings and accurate FPS reporting.

    Args:
        engine: FaceEngine instance.
        registry: In-memory dictionary mapping names to embedding vectors.
    """
    if not registry:
        print("\n[Notice] No registered people found.")
        print(f"Please register at least one face (Option 1) so it is saved to '{config.FACES_JSON_PATH.name}'.")
        return

    print("\n--- Recognize Faces from Recorded Video File ---")
    raw_path = input("Enter video file path: ").strip().strip('"').strip("'")

    if not raw_path:
        print("[Error] Video path cannot be empty.")
        return

    video_path = Path(raw_path)
    if not video_path.exists() or not video_path.is_file():
        print(f"[Error] File not found: '{video_path}'")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video file '{video_path.name}'. Check format and codec.")
        return

    engine.ensure_ready()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000.0 / source_fps))

    print(f"\n[Video] Playing: {video_path.name}")
    print(f"[Video] Frames: {total_frames} | Source FPS: {source_fps:.1f}")
    print("Controls:")
    print(" - [SPACE] : Pause / Resume playback")
    print(" - [Q] / [ESC] : Exit and return to menu\n")

    frame_idx = 0
    paused = False
    cached_results: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int], str]] = []
    total_detections = 0
    total_inferences = 0
    match_counts: Dict[str, int] = {}

    t_video_start = time.perf_counter()
    fps_smooth = source_fps

    try:
        while True:
            t_frame_start = time.perf_counter()

            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[Video] Reached end of video file.")
                    break
                frame_idx += 1

                # Full-resolution or scaled tolerant detection for recorded video analysis
                if frame_idx % config.PROCESS_EVERY_N_FRAMES == 0 or not cached_results:
                    total_inferences += 1
                    new_results = []
                    faces = engine.detect_faces(
                        frame,
                        scale=config.REC_INFERENCE_SCALE,
                        min_score=config.REC_MIN_DET_SCORE,
                        min_size=(config.REC_MIN_FACE_WIDTH, config.REC_MIN_FACE_HEIGHT),
                    )

                    for face in faces:
                        total_detections += 1
                        bbox, box_color, label_text, identity = identify_face(engine, face, registry)
                        match_counts[identity] = match_counts.get(identity, 0) + 1
                        new_results.append((bbox, box_color, label_text))

                    cached_results = new_results

            display_frame = frame
            w = display_frame.shape[1]
            for bbox, box_color, label_text in cached_results:
                draw_face_annotation(display_frame, bbox, box_color, label_text)

            # Compute processing time and smooth FPS
            proc_time_ms = (time.perf_counter() - t_frame_start) * 1000.0
            inst_fps = 1000.0 / proc_time_ms if proc_time_ms > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

            # Top banner distinguishing source FPS from processing FPS
            cv2.rectangle(display_frame, (0, 0), (w, 32), (30, 30, 30), -1)
            state_str = "[PAUSED]" if paused else "[PLAYING]"
            cv2.putText(
                display_frame,
                f"{video_path.name} | Frame {frame_idx}/{total_frames} | Src: {source_fps:.1f} FPS | Proc: {fps_smooth:.1f} FPS {state_str} | [SPACE] Pause | [Q] Exit",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(config.WINDOW_TITLE, display_frame)

            # Pace playback smoothly according to video's native delay
            wait_time = max(1, int(delay_ms - proc_time_ms)) if not paused else 50
            key = cv2.waitKey(wait_time) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Video recognition stopped by user.")
                break
            elif key == 32:  # SPACE to pause/resume
                paused = not paused

        # Summary report
        total_time_elapsed = max(0.001, time.perf_counter() - t_video_start)
        avg_processing_fps = frame_idx / total_time_elapsed

        print("\n================ Video Recognition Summary ================")
        print(f" File: {video_path.name}")
        print(f" Source Video Frame Rate: {source_fps:.1f} FPS")
        print(f" Average Processing Rate: {avg_processing_fps:.1f} FPS")
        print(f" Total Processing Time: {total_time_elapsed:.2f} s")
        print(f" Frames Processed: {frame_idx}/{total_frames}")
        print(f" Inferences Executed: {total_inferences} (Stride: every {config.PROCESS_EVERY_N_FRAMES} frames)")
        print(f" Total Face Detections: {total_detections}")
        if match_counts:
            print(" Identities Detected:")
            for ident, cnt in sorted(match_counts.items(), key=lambda x: -x[1]):
                print(f"  - {ident:<22}: {cnt} detections")
        else:
            print(" No faces detected in video.")
        print("===========================================================\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def list_registered_identities(registry: Dict[str, np.ndarray]) -> None:
    """Print all currently enrolled identities from faces.json to the console.

    Args:
        registry: In-memory dictionary mapping names to embedding vectors.
    """
    print("\n================ Registered Identities ================")
    if not registry:
        print(" No identities registered yet.")
        print(" Use Option 1 or Option 2 to register a face.")
    else:
        print(f" Source: {config.FACES_JSON_PATH}")
        print(f" Total enrolled: {len(registry)}")
        print(" ---------------------------------------------------")
        for idx, (name, emb) in enumerate(registry.items(), start=1):
            print(f" {idx}. {name:<25} [Vector dim: {emb.shape[0]}]")
        print(" ---------------------------------------------------")
    print("=======================================================\n")


def delete_registered_identity(engine: FaceEngine, registry: Dict[str, np.ndarray]) -> None:
    """Delete a selected registered identity and update faces.json immediately.

    Args:
        engine: FaceEngine instance.
        registry: In-memory dictionary mapping names to embedding vectors.
    """
    print("\n--- Delete Registered Person ---")
    if not registry:
        print("No registered identities found to delete.")
        return

    identities = list(registry.keys())
    print("Currently registered people:")
    for idx, name in enumerate(identities, start=1):
        print(f"  {idx}. {name}")

    user_input = input("\nEnter name or number to delete (or 'c' to cancel): ").strip()
    if user_input.lower() in ("c", "cancel", ""):
        print("Deletion cancelled.")
        return

    target_name: Optional[str] = None
    if user_input.isdigit():
        num = int(user_input)
        if 1 <= num <= len(identities):
            target_name = identities[num - 1]
    else:
        for name in identities:
            if name.lower() == user_input.lower():
                target_name = name
                break

    if target_name is None:
        print(f"[Error] Person '{user_input}' not found in registry.")
        return

    confirm = input(f"Are you sure you want to delete '{target_name}' from '{config.FACES_JSON_PATH.name}'? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Deletion cancelled.")
        return

    deleted_emb = registry.pop(target_name)
    try:
        save_faces_to_json(registry, config.FACES_JSON_PATH)
        engine.set_registry(registry)
        print(f"[Success] Identity '{target_name}' deleted successfully from '{config.FACES_JSON_PATH.name}'.")
    except Exception as exc:
        registry[target_name] = deleted_emb
        print(f"[Error] Failed to update storage file: {exc}")


def display_menu() -> None:
    """Print the main console navigation menu."""
    print("\n==========================================")
    print("         FaceMatch Recognition           ")
    print("==========================================")
    print("1. Register Face (Live Webcam)")
    print("2. Register Face from Image")
    print("3. Recognize Face (Live Webcam)")
    print("4. Recognize Face from Video File")
    print("5. List Registered People")
    print("6. Delete Registered Person")
    print("7. Exit")
    print("==========================================")


def main() -> None:
    """Application entry point: instant startup with resilient model pre-warming."""
    print("=" * 55)
    print("        Starting FaceMatch Face Recognition       ")
    print("=" * 55)

    # 1. Load existing registrations from local JSON file (instant)
    try:
        registry = load_faces_from_json(config.FACES_JSON_PATH)
        if registry:
            print(f"[Storage] Loaded {len(registry)} identity/identities from '{config.FACES_JSON_PATH.name}'.")
        else:
            print(f"[Storage] Ready. '{config.FACES_JSON_PATH.name}' will be created upon first registration.")
    except Exception as exc:
        print(f"\n[Fatal Storage Error] {exc}")
        print("FaceMatch aborted to protect data integrity. Please resolve the issue above.")
        sys.exit(1)

    # 2. Initialize FaceEngine with thread-safe pre-warming
    engine = FaceEngine()
    warmup_thread = threading.Thread(target=engine.initialize, daemon=True)
    warmup_thread.start()

    while True:
        try:
            display_menu()
            choice = input("Enter choice (1-7): ").strip()

            if choice == "1":
                register_face(engine, registry)
            elif choice == "2":
                register_face_from_image(engine, registry)
            elif choice == "3":
                recognize_faces(engine, registry)
            elif choice == "4":
                recognize_from_video_file(engine, registry)
            elif choice == "5":
                list_registered_identities(registry)
            elif choice == "6":
                delete_registered_identity(engine, registry)
            elif choice == "7":
                print("\nExiting FaceMatch. Goodbye!")
                break
            else:
                print("\n[Invalid Option] Please enter a number between 1 and 7.")
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user (Ctrl+C). Exiting FaceMatch...")
            break
        except Exception as exc:
            print(f"\n[Unexpected Error] An error occurred: {exc}")


if __name__ == "__main__":
    main()
