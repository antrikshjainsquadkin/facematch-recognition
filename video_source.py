"""Common Video-Source Abstraction and Hybrid Backends for FaceMatch.

Provides a unified interface for video ingestion:
- BaseVideoSource: Common abstract interface (open, read, is_opened, release).
- OpenCVVideoSource: High-performance local webcam and video file decoding.
- PyAVVideoSource: Direct FFmpeg-based RTSP/network stream decoding with low latency.
- ThreadedFrameGrabber: Bounded background worker eliminating buffer lag for live feeds.
- VideoSourceFactory: Intelligent backend routing and transparent fallback.
"""

import os
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

import config

# Conditional import of PyAV with informative capability check
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    av = None
    PYAV_AVAILABLE = False


class BaseVideoSource(ABC):
    """Abstract Base Class for all FaceMatch video ingestion backends.

    Guarantees that both OpenCV and PyAV backends return standardized
    NumPy arrays in BGR format (H, W, 3) uint8 so the FaceEngine pipeline
    operates identically regardless of decoding backend.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open the video stream or device.

        Returns:
            True if opened successfully, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Read the next video frame.

        Returns:
            Tuple of:
            - ret (bool): True if frame read succeeded.
            - frame (Optional[np.ndarray]): Decoded BGR image array.
            - metadata (Dict[str, Any]): Stream information (PTS, timestamp, backend).
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if video source is active and opened.

        Returns:
            True if active, False otherwise.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release all hardware, network, and decoder resources."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Return (width, height) resolution of the stream."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Return frame rate (FPS) of the video source."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return identifier string of the active backend."""
        pass

    def __enter__(self) -> "BaseVideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class OpenCVVideoSource(BaseVideoSource):
    """OpenCV-based video source for webcams, recorded video files, and fallback RTSP."""

    def __init__(
        self,
        source: Union[int, str, Path],
        transport: str = config.RTSP_TRANSPORT,
        buffer_size: int = 1,
    ) -> None:
        """Initialize OpenCV video source.

        Args:
            source: Camera index (int), file path (str/Path), or RTSP URL.
            transport: Transport protocol for RTSP ('tcp' or 'udp').
            buffer_size: OpenCV internal queue buffer depth.
        """
        self.source = source
        self.transport = transport
        self.buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._fps: float = 25.0
        self._width: int = 640
        self._height: int = 480

    def open(self) -> bool:
        """Open source using cv2.VideoCapture with DirectShow on Windows for webcams."""
        self.release()
        self._frame_idx = 0

        # Handle webcam index
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            cam_idx = int(self.source)
            try:
                self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap.release()
                    self._cap = None
            except Exception:
                self._cap = None

            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(cam_idx)

            if self._cap is not None and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
                self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                return True
            return False

        # Handle file or network stream
        src_str = str(self.source)
        if src_str.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.transport}|stimeout;5000000"
            )
            self._cap = cv2.VideoCapture(src_str, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(src_str)
        else:
            self._cap = cv2.VideoCapture(src_str)

        if self._cap is not None and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            return True

        if self._cap:
            self._cap.release()
            self._cap = None
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Read next frame from OpenCV."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, {"backend": self.backend_name, "error": "Not opened"}

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None, {"backend": self.backend_name, "frame_idx": self._frame_idx, "eof": True}

        self._frame_idx += 1
        metadata = {
            "backend": self.backend_name,
            "frame_idx": self._frame_idx,
            "timestamp_sec": time.time(),
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
        return True, frame, metadata

    def is_opened(self) -> bool:
        """Check if capture is open."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Release OpenCV capture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return "OpenCV"


class PyAVVideoSource(BaseVideoSource):
    """PyAV-based video source for RTSP network streams and media container demuxing."""

    def __init__(
        self,
        source: Union[str, Path],
        transport: str = config.PYAV_RTSP_TRANSPORT,
        timeout_sec: float = config.PYAV_TIMEOUT_SEC,
        low_delay: bool = config.PYAV_LOW_DELAY,
        nobuffer: bool = config.PYAV_NOBUFFER,
    ) -> None:
        """Initialize PyAV video source.

        Args:
            source: Video file path or RTSP URL.
            transport: RTSP protocol ('tcp' or 'udp').
            timeout_sec: Socket/demuxer timeout in seconds.
            low_delay: Enable FFmpeg low_delay flags.
            nobuffer: Enable FFmpeg nobuffer flags for live streaming.
        """
        if not PYAV_AVAILABLE:
            raise ImportError(
                "PyAV ('av') is not installed. Install via: pip install av"
            )

        self.source = str(source)
        self.transport = transport
        self.timeout_sec = timeout_sec
        self.low_delay = low_delay
        self.nobuffer = nobuffer

        self._container: Optional[Any] = None
        self._video_stream: Optional[Any] = None
        self._packet_iter: Optional[Any] = None
        self._frame_idx: int = 0
        self._fps: float = 25.0
        self._width: int = 640
        self._height: int = 480
        self._is_opened: bool = False

    def open(self) -> bool:
        """Open source using PyAV container with direct FFmpeg options."""
        self.release()
        self._frame_idx = 0

        options: Dict[str, str] = {}
        is_rtsp = self.source.lower().startswith("rtsp://") or self.source.lower().startswith("rtsps://")

        if is_rtsp:
            options["rtsp_transport"] = self.transport
            options["stimeout"] = str(int(self.timeout_sec * 1000000))  # microseconds
            options["max_delay"] = "500000"
            if self.nobuffer:
                options["fflags"] = "nobuffer"
            if self.low_delay:
                options["flags"] = "low_delay"

        try:
            self._container = av.open(
                self.source,
                options=options if options else None,
                timeout=self.timeout_sec,
            )
            # Find primary video stream
            self._video_stream = next(
                (s for s in self._container.streams if s.type == "video"),
                None,
            )

            if self._video_stream is None:
                self.release()
                return False

            # Configure video stream low-latency decoding
            if is_rtsp and self.low_delay:
                self._video_stream.codec_context.flags = "LOW_DELAY"
                self._video_stream.codec_context.thread_type = "AUTO"

            self._width = self._video_stream.width or 640
            self._height = self._video_stream.height or 480
            # Extract FPS safely
            rate = getattr(self._video_stream, "average_rate", None) or getattr(self._video_stream, "base_rate", None)
            if rate is not None and rate > 0:
                self._fps = float(rate)
            else:
                self._fps = 25.0

            # Create demux generator
            self._packet_iter = self._container.demux(self._video_stream)
            self._is_opened = True
            return True

        except Exception as exc:
            self.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Decode and read next frame, converting directly to BGR numpy array."""
        if not self._is_opened or self._container is None or self._packet_iter is None:
            return False, None, {"backend": self.backend_name, "error": "Not opened"}

        try:
            for packet in self._packet_iter:
                # Discard corrupted / incomplete packets
                if packet.dts is None and packet.size == 0:
                    continue

                for frame in packet.decode():
                    # Direct FFmpeg colorspace conversion to standard BGR24 NumPy array
                    bgr_array = frame.to_ndarray(format="bgr24")
                    self._frame_idx += 1

                    metadata = {
                        "backend": self.backend_name,
                        "frame_idx": self._frame_idx,
                        "pts": frame.pts,
                        "dts": frame.dts,
                        "time_sec": float(frame.time) if frame.time is not None else time.time(),
                        "is_keyframe": bool(getattr(frame, "key_frame", False)),
                        "width": bgr_array.shape[1],
                        "height": bgr_array.shape[0],
                    }
                    return True, bgr_array, metadata

            return False, None, {"backend": self.backend_name, "eof": True}

        except Exception as exc:
            return False, None, {"backend": self.backend_name, "error": str(exc)}

    def is_opened(self) -> bool:
        """Check if PyAV container is active."""
        return self._is_opened and self._container is not None

    def release(self) -> None:
        """Cleanly close PyAV container and decoder."""
        self._is_opened = False
        self._packet_iter = None
        self._video_stream = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return "PyAV"


class ThreadedFrameGrabber(BaseVideoSource):
    """Threaded wrapper with bounded queue (buffer depth 1) eliminating latency accumulation.

    Drains the underlying source (OpenCV or PyAV) in a dedicated background worker,
    ensuring that the FaceEngine consumer always reads the freshest live frame with 0 ms lag.
    Also provides automatic reconnection state machine for network streams.
    """

    def __init__(
        self,
        source: BaseVideoSource,
        timeout_sec: float = config.STREAM_TIMEOUT_SEC,
        reconnect_interval_sec: float = config.STREAM_RECONNECT_INTERVAL_SEC,
        max_reconnect_attempts: int = config.STREAM_MAX_RECONNECT_ATTEMPTS,
        masked_url: str = "",
    ) -> None:
        """Initialize the threaded grabber.

        Args:
            source: Underlying BaseVideoSource instance.
            timeout_sec: Seconds without frame before triggering reconnection.
            reconnect_interval_sec: Delay between reconnection attempts.
            max_reconnect_attempts: Max consecutive connection retries.
            masked_url: Sanitized stream URL for safe display and logging.
        """
        self.underlying_source = source
        self.timeout_sec = timeout_sec
        self.reconnect_interval_sec = reconnect_interval_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.masked_url = masked_url

        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_metadata: Dict[str, Any] = {}
        self._last_frame_time: float = 0.0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        self.is_connected: bool = False
        self.is_reconnecting: bool = False
        self.status_message: str = "Connecting..."

    def open(self) -> bool:
        """Start background grabber thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True

    def _worker(self) -> None:
        """Continuous background grab loop with reconnection and bounded buffer."""
        reconnect_count = 0

        while self._running:
            if not self.underlying_source.is_opened():
                self.is_connected = False
                self.is_reconnecting = True
                self.status_message = f"Connecting via {self.underlying_source.backend_name}..."

                opened = self.underlying_source.open()
                if opened and self.underlying_source.is_opened():
                    self.is_connected = True
                    self.is_reconnecting = False
                    self.status_message = f"Connected ({self.underlying_source.backend_name})"
                    self._last_frame_time = time.time()
                    reconnect_count = 0
                else:
                    reconnect_count += 1
                    if reconnect_count > self.max_reconnect_attempts:
                        self.status_message = (
                            f"Unable to connect after {self.max_reconnect_attempts} attempts."
                        )
                        self.is_reconnecting = False
                        break
                    self.status_message = (
                        f"Connection lost; retrying ({reconnect_count}/{self.max_reconnect_attempts})..."
                    )
                    time.sleep(self.reconnect_interval_sec)
                    continue

            # Grab next frame from underlying source
            ret, frame, metadata = self.underlying_source.read()
            now = time.time()

            if ret and frame is not None and frame.size > 0:
                with self._lock:
                    self._last_frame = frame
                    self._last_metadata = metadata
                    self._last_frame_time = now
                self.is_connected = True
                self.is_reconnecting = False
                self.status_message = f"Connected ({self.underlying_source.backend_name})"
            else:
                # Frame freeze / drop detection
                if now - self._last_frame_time > self.timeout_sec and self._last_frame_time > 0:
                    self.is_connected = False
                    self.is_reconnecting = True
                    self.status_message = "Stream frozen. Reconnecting..."
                    self.underlying_source.release()
                    time.sleep(1.0)
                else:
                    time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Fetch freshest live frame with zero buffer lag (bounded latest-frame strategy)."""
        with self._lock:
            if self._last_frame is not None:
                meta = self._last_metadata.copy()
                meta["status"] = self.status_message
                return True, self._last_frame.copy(), meta
            return False, None, {"status": self.status_message, "backend": self.backend_name}

    def is_opened(self) -> bool:
        """Check if background worker is running and connected."""
        return self._running and self.is_connected

    def release(self) -> None:
        """Stop worker thread and release underlying source."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self.underlying_source.release()
        with self._lock:
            self._last_frame = None
            self._last_metadata.clear()
        self.is_connected = False
        self.status_message = "Disconnected"

    def disconnect(self) -> None:
        """Alias for clean release."""
        self.release()

    def get_resolution(self) -> Tuple[int, int]:
        return self.underlying_source.get_resolution()

    def get_fps(self) -> float:
        return self.underlying_source.get_fps()

    @property
    def backend_name(self) -> str:
        return f"{self.underlying_source.backend_name} (Threaded)"


class VideoSourceFactory:
    """Factory selecting the most suitable video backend and managing fallbacks.

    Design philosophy:
    - Webcams: OpenCV (superior DirectShow integration and device controls on Windows).
    - Local video files: OpenCV (low overhead, native seek) or PyAV.
    - RTSP / CCTV Streams: PyAV (direct FFmpeg demuxing, nobuffer, low_delay, explicit timeouts)
      with graceful automatic fallback to OpenCV if PyAV fails or is uninstalled.
    """

    @staticmethod
    def is_rtsp_source(source: Union[int, str, Path]) -> bool:
        """Check if source URI is an RTSP or HTTP network stream."""
        if isinstance(source, (int, Path)):
            return False
        src_lower = str(source).lower().strip()
        return src_lower.startswith(("rtsp://", "rtsps://", "http://", "https://"))

    @staticmethod
    def is_webcam_source(source: Union[int, str, Path]) -> bool:
        """Check if source is a local webcam device."""
        if isinstance(source, int):
            return True
        src_str = str(source).strip()
        return src_str.isdigit()

    @classmethod
    def create_source(
        cls,
        source: Union[int, str, Path],
        backend: str = config.DEFAULT_VIDEO_BACKEND,
        is_live: bool = False,
        timeout_sec: float = config.STREAM_TIMEOUT_SEC,
        masked_url: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> BaseVideoSource:
        """Instantiate the optimal BaseVideoSource based on input type and configuration.

        Args:
            source: Input device index, file path, or RTSP URL.
            backend: "auto", "pyav", or "opencv".
            is_live: If True, wraps source in ThreadedFrameGrabber for zero lag.
            timeout_sec: Network / read timeout in seconds.
            masked_url: Sanitized URL for safe logging/display.
            options: Optional backend parameter dictionary.

        Returns:
            Configured BaseVideoSource instance.
        """
        backend_choice = (backend or "auto").lower().strip()
        opts = options or {}

        # 1. Determine optimal backend
        selected_backend = "opencv"

        if backend_choice == "pyav":
            if PYAV_AVAILABLE:
                selected_backend = "pyav"
            else:
                print("[VideoSourceFactory] Notice: PyAV requested but not installed. Falling back to OpenCV.")
                selected_backend = "opencv"

        elif backend_choice == "opencv":
            selected_backend = "opencv"

        else:  # "auto"
            if cls.is_webcam_source(source):
                selected_backend = "opencv"
            elif cls.is_rtsp_source(source):
                if PYAV_AVAILABLE:
                    selected_backend = "pyav"
                else:
                    selected_backend = "opencv"
            else:
                # Video file
                selected_backend = "opencv"

        # 2. Instantiate selected backend
        base_source: BaseVideoSource
        if selected_backend == "pyav":
            base_source = PyAVVideoSource(
                source=str(source),
                transport=opts.get("transport", config.PYAV_RTSP_TRANSPORT),
                timeout_sec=timeout_sec,
                low_delay=opts.get("low_delay", config.PYAV_LOW_DELAY),
                nobuffer=opts.get("nobuffer", config.PYAV_NOBUFFER),
            )
        else:
            base_source = OpenCVVideoSource(
                source=source,
                transport=opts.get("transport", config.RTSP_TRANSPORT),
                buffer_size=opts.get("buffer_size", 1),
            )

        # 3. If live stream, wrap in ThreadedFrameGrabber
        if is_live:
            return ThreadedFrameGrabber(
                source=base_source,
                timeout_sec=timeout_sec,
                reconnect_interval_sec=config.STREAM_RECONNECT_INTERVAL_SEC,
                max_reconnect_attempts=config.STREAM_MAX_RECONNECT_ATTEMPTS,
                masked_url=masked_url or str(source),
            )

        return base_source
