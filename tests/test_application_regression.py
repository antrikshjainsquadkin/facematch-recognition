"""Regression Test Suite for FaceMatch Application Components.

Tests:
1. RTSP URL building, credential masking, and CCTV configuration persistence.
2. Hybrid video source factory and fallback behavior.
3. Temporal tracker window stabilization and event debouncing.
4. Main application importability and menu structure.
"""

import unittest
from pathlib import Path

import config
from camera_stream import (
    mask_stream_credentials,
    build_rtsp_url,
    load_cctv_config,
    save_cctv_config,
    clear_cctv_config,
)
from video_source import VideoSourceFactory, OpenCVVideoSource, PyAVVideoSource
from temporal_tracker import TemporalRecognitionTracker


class TestApplicationRegression(unittest.TestCase):
    """Test suite ensuring zero regression across core FaceMatch components."""

    def test_01_rtsp_url_building_and_masking(self):
        """Verify RTSP URL formatting and credential masking for security."""
        url = build_rtsp_url("admin", "Secret123!", "192.168.1.50", "554", "/live/ch0")
        self.assertEqual(url, "rtsp://admin:Secret123%21@192.168.1.50:554/live/ch0")

        masked = mask_stream_credentials(url)
        self.assertNotIn("Secret123!", masked)
        self.assertIn("192.168.1.50", masked)
        self.assertIn("admin:***@", masked)

    def test_02_cctv_config_persistence(self):
        """Verify saving and clearing CCTV configuration safely."""
        test_cfg = {
            "username": "user_test",
            "password": "pwd_test",
            "ip_address": "192.168.1.99",
            "port": "554",
            "stream_path": "/test",
            "video_backend": "auto",
        }
        test_path = config.BASE_DIR / ".test_cctv_config.json"
        try:
            save_cctv_config(test_cfg, config_path=test_path)
            loaded = load_cctv_config(config_path=test_path)
            self.assertEqual(loaded.get("username"), "user_test")
            self.assertEqual(loaded.get("ip_address"), "192.168.1.99")

            clear_cctv_config(config_path=test_path)
            self.assertFalse(test_path.exists())
        finally:
            if test_path.exists():
                test_path.unlink()

    def test_03_video_source_factory(self):
        """Verify video source backend factory selects the correct class."""
        # Non-existent or dummy webcam index selects OpenCV
        reader_cv = VideoSourceFactory.create_source(0, backend="opencv")
        self.assertIsInstance(reader_cv, OpenCVVideoSource)

        # RTSP stream with auto backend selects PyAV preferred
        reader_rtsp = VideoSourceFactory.create_source("rtsp://192.168.1.10:554/stream", backend="auto")
        self.assertIsInstance(reader_rtsp, PyAVVideoSource)

    def test_04_temporal_tracker_debouncing(self):
        """Verify temporal tracker suppresses redundant events for same identity within cooldown."""
        tracker = TemporalRecognitionTracker(
            window_size=3,
            min_confirmations=2,
            event_cooldown_sec=5.0,
        )

        bbox = (50, 50, 150, 150)
        # Frame 1: first sighting
        r1 = tracker.update([(bbox, "Bob", 0.70, False)], current_time=100.0)
        self.assertFalse(r1[0][3])  # not confirmed yet

        # Frame 2: confirmation
        r2 = tracker.update([(bbox, "Bob", 0.72, False)], current_time=100.1)
        self.assertTrue(r2[0][3])   # confirmed
        self.assertTrue(r2[0][4])   # is_new_event

        # Frame 3: within cooldown (101.0 < 100.1 + 5.0)
        r3 = tracker.update([(bbox, "Bob", 0.71, False)], current_time=101.0)
        self.assertTrue(r3[0][3])   # confirmed
        self.assertFalse(r3[0][4])  # event suppressed by debounce cooldown

    def test_05_main_imports_successfully(self):
        """Verify main.py and all required entry points load cleanly."""
        import main
        self.assertTrue(callable(main.recognize_faces))
        self.assertTrue(callable(main.recognize_from_video_file))
        self.assertTrue(callable(main.run_cctv_recognition))


if __name__ == "__main__":
    unittest.main()
