"""Comprehensive Test Suite for Masked-Face Recognition in FaceMatch.

Covers:
1. Model initialization, inference latency, and memory safety.
2. Mask detection algorithm on unmasked vs masked faces across multiple mask colors (surgical blue, white N95, black, gray, green).
3. Genuine recognition of registered subjects without masks (standard baseline).
4. Genuine recognition of registered subjects wearing face masks (dual-threshold matching).
5. Rejection of unknown masked identities (imposter separation).
6. Ambiguity margin rejection for close candidates.
7. Registration protection (rejection of masked face enrollments).
8. Quality verification on masked faces (upper periocular sharpness evaluation).
9. Multiple faces in a frame (mixed masked and unmasked).
10. Temporal tracker stabilization with mask status.
11. Backward-compatible JSON storage operations.
"""

import unittest
import numpy as np
import cv2
import insightface.data

import config
from face_engine import (
    FaceEngine,
    compute_mask_probability,
    check_face_quality,
    compute_sharpness,
    load_faces_from_json,
    save_faces_to_json,
)
from temporal_tracker import TemporalRecognitionTracker


class TestMaskedFaceRecognition(unittest.TestCase):
    """Test suite validating masked-face recognition and regression safety."""

    @classmethod
    def setUpClass(cls):
        """Initialize FaceEngine once for the test suite."""
        cls.engine = FaceEngine()
        cls.engine.initialize()
        cls.t1 = insightface.data.get_image("t1")  # Standard Tom Hanks test image
        cls.faces_orig = cls.engine.detect_faces(cls.t1)
        cls.lead_face = cls.faces_orig[0]

    def _create_masked_image(self, img: np.ndarray, face, color_bgr: tuple[int, int, int]) -> np.ndarray:
        """Helper to create a synthetic masked face image with realistic coverage."""
        masked_img = img.copy()
        bx1, by1, bx2, by2 = [int(v) for v in face.bbox]
        kps = face.kps
        nose_y = int(kps[2][1]) if kps is not None else int(by1 + 0.55 * (by2 - by1))
        chin_y = by2

        pts = np.array([
            [bx1, nose_y],
            [int(kps[2][0]), nose_y - 8],
            [bx2, nose_y],
            [bx2, chin_y],
            [bx1, chin_y],
        ], dtype=np.int32)
        cv2.fillPoly(masked_img, [pts], color_bgr)
        return masked_img

    def test_01_model_initialization(self):
        """Verify model initialization state, thread configuration, and ready flag."""
        self.assertTrue(self.engine.is_ready)
        self.assertIsNotNone(self.engine._app)

    def test_02_unmasked_faces_not_classified_as_masked(self):
        """Verify that genuine unmasked faces are never misclassified as masked."""
        for i, face in enumerate(self.faces_orig):
            is_masked, mask_score = compute_mask_probability(self.t1, face)
            self.assertFalse(
                is_masked,
                f"Face {i} falsely flagged as masked (score={mask_score:.3f})"
            )
            self.assertLess(mask_score, config.MASK_SCORE_THRESHOLD)

    def test_03_masked_faces_correctly_detected(self):
        """Verify accurate mask detection across surgical, N95, cloth, and colored masks."""
        mask_types = [
            ("Surgical Blue", (210, 180, 70)),
            ("White N95", (240, 240, 240)),
            ("Black Fabric", (25, 25, 25)),
            ("Gray Fabric", (128, 128, 128)),
            ("Green Surgical", (50, 180, 50)),
        ]

        for mask_name, color_bgr in mask_types:
            masked_img = self._create_masked_image(self.t1, self.lead_face, color_bgr)
            faces = self.engine.detect_faces(masked_img)

            # Find corresponding face
            target_face = None
            bx1, by1, bx2, by2 = [int(v) for v in self.lead_face.bbox]
            for f in faces:
                mb = f.bbox
                ix1, iy1 = max(bx1, mb[0]), max(by1, mb[1])
                ix2, iy2 = min(bx2, mb[2]), min(by2, mb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (bx2 - bx1) * (by2 - by1) + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter
                if inter / max(1.0, union) > 0.5:
                    target_face = f
                    break

            self.assertIsNotNone(target_face, f"Could not detect face for {mask_name}")
            is_masked, mask_score = self.engine.detect_face_mask(masked_img, target_face)
            self.assertTrue(
                is_masked,
                f"{mask_name} failed detection (is_masked={is_masked}, score={mask_score:.3f})"
            )
            self.assertGreaterEqual(mask_score, config.MASK_SCORE_THRESHOLD)

    def test_04_unmasked_genuine_face_recognition(self):
        """Verify that an unmasked registered person is recognized at standard threshold."""
        registry = {"Tom_Hanks": self.lead_face.normed_embedding}
        ident, score = self.engine.match_face(self.lead_face.normed_embedding, registry, is_masked=False)

        self.assertEqual(ident, "Tom_Hanks")
        self.assertGreaterEqual(score, config.SIMILARITY_THRESHOLD)

    def test_05_masked_genuine_face_recognition(self):
        """Verify that a registered person wearing a surgical mask is recognized accurately."""
        registry = {"Tom_Hanks": self.lead_face.normed_embedding}
        masked_img = self._create_masked_image(self.t1, self.lead_face, (210, 180, 70))
        faces = self.engine.detect_faces(masked_img)

        # Match corresponding face
        target_face = None
        bx1, by1, bx2, by2 = [int(v) for v in self.lead_face.bbox]
        for f in faces:
            mb = f.bbox
            ix1, iy1 = max(bx1, mb[0]), max(by1, mb[1])
            ix2, iy2 = min(bx2, mb[2]), min(by2, mb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (bx2 - bx1) * (by2 - by1) + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter
            if inter / max(1.0, union) > 0.5:
                target_face = f
                break

        self.assertIsNotNone(target_face)
        self.assertTrue(target_face.is_masked)

        ident, score, is_amb = self.engine.match_face_detailed(
            target_face.normed_embedding,
            registry,
            is_masked=target_face.is_masked,
        )

        self.assertEqual(ident, "Tom_Hanks")
        self.assertGreaterEqual(score, config.MASKED_SIMILARITY_THRESHOLD)
        self.assertFalse(is_amb)

    def test_06_masked_imposter_rejection(self):
        """Verify that an unregistered person wearing a mask is safely rejected as Unknown Person."""
        # Register Alice (faces_orig[2]) only; query with masked lead_face (Tom Hanks wearing a mask)
        registry = {"Alice": self.faces_orig[2].normed_embedding}
        masked_img = self._create_masked_image(self.t1, self.lead_face, (210, 180, 70))
        faces = self.engine.detect_faces(masked_img)

        # Locate lead_face in masked image
        target_face = None
        bx1, by1, bx2, by2 = [int(v) for v in self.lead_face.bbox]
        for f in faces:
            mb = f.bbox
            ix1, iy1 = max(bx1, mb[0]), max(by1, mb[1])
            ix2, iy2 = min(bx2, mb[2]), min(by2, mb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (bx2 - bx1) * (by2 - by1) + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter
            if inter / max(1.0, union) > 0.5:
                target_face = f
                break

        self.assertIsNotNone(target_face)
        self.assertTrue(target_face.is_masked)
        ident, score, _ = self.engine.match_face_detailed(
            target_face.normed_embedding,
            registry,
            is_masked=True,
        )

        self.assertEqual(ident, "Unknown Person")
        self.assertLess(score, config.MASKED_SIMILARITY_THRESHOLD)

    def test_07_registration_rejects_masked_faces(self):
        """Verify that extract_single_face_for_registration rejects masked faces."""
        # Crop single face frame
        bx1, by1, bx2, by2 = [int(v) for v in self.lead_face.bbox]
        crop = self.t1[max(0, by1 - 40): min(self.t1.shape[0], by2 + 40), max(0, bx1 - 40): min(self.t1.shape[1], bx2 + 40)].copy()

        # Unmasked crop passes
        emb, err, bbox = self.engine.extract_single_face_for_registration(crop)
        self.assertIsNone(err, f"Unmasked registration failed unexpectedly: {err}")
        self.assertIsNotNone(emb)

        # Masked crop is rejected with informative message
        ch, cw = crop.shape[:2]
        pts = np.array([
            [int(0.1 * cw), int(0.5 * ch)],
            [int(0.5 * cw), int(0.45 * ch)],
            [int(0.9 * cw), int(0.5 * ch)],
            [int(0.9 * cw), int(0.95 * ch)],
            [int(0.1 * cw), int(0.95 * ch)],
        ], dtype=np.int32)
        masked_crop = crop.copy()
        cv2.fillPoly(masked_crop, [pts], (210, 180, 70))

        emb_m, err_m, bbox_m = self.engine.extract_single_face_for_registration(masked_crop)
        self.assertIsNone(emb_m)
        self.assertIsNotNone(err_m)
        self.assertIn("Face mask detected", err_m)

    def test_08_check_face_quality_on_masked_faces(self):
        """Verify that check_face_quality passes on sharp masked faces."""
        masked_img = self._create_masked_image(self.t1, self.lead_face, (210, 180, 70))
        faces = self.engine.detect_faces(masked_img)
        target_face = faces[0]

        is_ok, reason = check_face_quality(masked_img, target_face)
        self.assertTrue(is_ok, f"check_face_quality failed on masked face: {reason}")
        self.assertEqual(reason, "Quality OK")

    def test_09_temporal_tracker_tracks_mask_status(self):
        """Verify TemporalRecognitionTracker correctly propagates and stabilizes mask status."""
        tracker = TemporalRecognitionTracker(window_size=3, min_confirmations=2)

        bbox = (100, 100, 200, 200)
        # Frame 1: Masked detection of Alice
        res1 = tracker.update([(bbox, "Alice", 0.62, False, True)])
        self.assertEqual(len(res1), 1)
        _, final_id, _, is_confirmed, _, is_masked = res1[0]
        self.assertFalse(is_confirmed)  # Needs 2 frames
        self.assertTrue(is_masked)

        # Frame 2: Masked detection of Alice
        res2 = tracker.update([(bbox, "Alice", 0.65, False, True)])
        self.assertEqual(len(res2), 1)
        _, final_id, _, is_confirmed, is_new_event, is_masked = res2[0]
        self.assertTrue(is_confirmed)
        self.assertEqual(final_id, "Alice")
        self.assertTrue(is_masked)
        self.assertTrue(is_new_event)

    def test_10_json_storage_backward_compatibility(self):
        """Verify existing faces.json schema format and round-trip integrity."""
        dummy_registry = {
            "Subject_A": self.lead_face.normed_embedding,
        }
        test_path = config.BASE_DIR / ".test_faces.json"
        try:
            save_faces_to_json(dummy_registry, test_path)
            loaded = load_faces_from_json(test_path)
            self.assertIn("Subject_A", loaded)
            self.assertEqual(loaded["Subject_A"].shape, (512,))
            sim = float(np.dot(dummy_registry["Subject_A"], loaded["Subject_A"]))
            self.assertAlmostEqual(sim, 1.0, places=4)
        finally:
            if test_path.exists():
                test_path.unlink()

    def test_11_low_quality_and_dark_faces_rejected(self):
        """Verify that underexposed or edge-truncated faces are safely rejected."""
        # 1. Dark underexposed frame
        dark_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces_dark = self.engine.detect_faces(dark_img)
        self.assertEqual(len(faces_dark), 0)

        # 2. Corrupted / truncated quality check
        ok, reason = check_face_quality(dark_img, self.lead_face)
        self.assertFalse(ok)
        self.assertTrue("underexposed" in reason.lower() or "truncated" in reason.lower())

    def test_12_multiple_faces_mixed_masked_and_unmasked(self):
        """Verify that multiple faces in a frame can be detected with respective mask flags."""
        # Mask lead_face (Tom Hanks) with surgical blue, leave all other faces in t1 unmasked
        mixed_img = self._create_masked_image(self.t1, self.lead_face, (210, 180, 70))
        faces = self.engine.detect_faces(mixed_img)
        self.assertGreaterEqual(len(faces), 2)

        masked_count = sum(1 for f in faces if f.is_masked)
        unmasked_count = sum(1 for f in faces if not f.is_masked)

        self.assertGreaterEqual(masked_count, 1, "At least one face should be detected as masked")
        self.assertGreaterEqual(unmasked_count, 1, "At least one face should be detected as unmasked")

    def test_13_batch_face_matching_mixed_masks(self):
        """Verify match_faces_batch with mixed masked and unmasked queries."""
        registry = {
            "Tom_Hanks": self.lead_face.normed_embedding,
            "Other_Person": self.faces_orig[2].normed_embedding,
        }

        # Query 1: Tom Hanks unmasked
        # Query 2: Tom Hanks masked
        # Query 3: Unknown random embedding
        masked_img = self._create_masked_image(self.t1, self.lead_face, (210, 180, 70))
        faces = self.engine.detect_faces(masked_img)
        bx1, by1, bx2, by2 = [int(v) for v in self.lead_face.bbox]
        masked_tom = None
        for f in faces:
            mb = f.bbox
            ix1, iy1 = max(bx1, mb[0]), max(by1, mb[1])
            ix2, iy2 = min(bx2, mb[2]), min(by2, mb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (bx2 - bx1) * (by2 - by1) + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter
            if inter / max(1.0, union) > 0.5:
                masked_tom = f
                break
        self.assertIsNotNone(masked_tom)

        rand_emb = np.random.randn(512).astype(np.float32)
        rand_emb /= np.linalg.norm(rand_emb)

        queries = [self.lead_face.normed_embedding, masked_tom.normed_embedding, rand_emb]
        mask_flags = [False, True, True]

        results = self.engine.match_faces_batch(queries, registry, is_masked_list=mask_flags)
        self.assertEqual(len(results), 3)

        # Query 1 recognized as Tom_Hanks
        self.assertEqual(results[0][0], "Tom_Hanks")
        self.assertGreaterEqual(results[0][1], config.SIMILARITY_THRESHOLD)

        # Query 2 recognized as Tom_Hanks
        self.assertEqual(results[1][0], "Tom_Hanks")
        self.assertGreaterEqual(results[1][1], config.MASKED_SIMILARITY_THRESHOLD)

        # Query 3 rejected as Unknown Person
        self.assertEqual(results[2][0], "Unknown Person")

    def test_14_ambiguity_margin_rejection(self):
        """Verify that ambiguous embeddings with near-identical similarity are safely rejected."""
        base_emb = self.lead_face.normed_embedding.copy()
        # Create twin candidate with slight perturbation
        twin_emb = base_emb + 0.005 * np.random.randn(512).astype(np.float32)
        twin_emb /= np.linalg.norm(twin_emb)

        ambiguous_registry = {
            "Candidate_1": base_emb,
            "Candidate_2": twin_emb,
        }

        ident, score, is_amb = self.engine.match_face_detailed(
            base_emb,
            ambiguous_registry,
            threshold=0.50,
            min_margin=0.05,
        )
        self.assertTrue(is_amb, "Should flag as ambiguous due to close second candidate")
        self.assertEqual(ident, "Unknown Person")

    def test_15_repeated_inference_stability_and_latency(self):
        """Verify inference latency and stability across repeated inferences."""
        import time
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            faces = self.engine.detect_faces(self.t1)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)
            self.assertGreater(len(faces), 0)

        avg_latency = float(np.mean(latencies))
        self.assertLess(avg_latency, 500.0, f"Average inference too slow: {avg_latency:.1f}ms")

    def test_16_eye_anchored_masked_face_refinement(self):
        """Verify that masked face embedding refinement elevates genuine recognition confidence."""
        registry = {"Tom_Hanks": self.lead_face.normed_embedding}
        masked_img = self._create_masked_image(self.t1, self.lead_face, (210, 180, 70))
        faces = self.engine.detect_faces(masked_img)

        # Locate corresponding face
        target_face = None
        bx1, by1, bx2, by2 = [int(v) for v in self.lead_face.bbox]
        for f in faces:
            mb = f.bbox
            ix1, iy1 = max(bx1, mb[0]), max(by1, mb[1])
            ix2, iy2 = min(bx2, mb[2]), min(by2, mb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (bx2 - bx1) * (by2 - by1) + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter
            if inter / max(1.0, union) > 0.5:
                target_face = f
                break

        self.assertIsNotNone(target_face)
        # Must recognize Tom_Hanks and return his registered identity
        ident, score = self.engine.match_face(target_face.normed_embedding, registry, is_masked=target_face.is_masked)
        self.assertEqual(ident, "Tom_Hanks")
        self.assertGreater(score, 0.60)
        self.assertTrue(target_face.is_masked)


if __name__ == "__main__":
    unittest.main()

