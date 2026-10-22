"""Temporal Recognition Tracker and Event Debouncing Module for FaceMatch.

Stabilizes real-time facial recognition across video frames by:
- Associating detected face bounding boxes across frames using spatial IoU tracking.
- Maintaining a sliding window of recent identity predictions and similarity scores.
- Requiring temporal confirmation (e.g. M matches out of N frames) to prevent transient false acceptances.
- Suppressing duplicate recognition events for the same person during continuous appearance.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np

import config


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two bounding boxes (x1, y1, x2, y2).

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        IoU overlap ratio in range [0.0, 1.0].
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


@dataclass
class TrackState:
    """State for a single tracked face across consecutive frames."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen_time: float
    history: Deque[Tuple[str, float]] = field(default_factory=deque)
    confirmed_identity: Optional[str] = None
    event_emitted: bool = False
    last_event_time: float = 0.0
    last_is_masked: bool = False
    mask_history: Deque[bool] = field(default_factory=deque)
    consecutive_unknown_count: int = 0


class TemporalRecognitionTracker:
    """Tracks detected faces across frames to provide temporal confirmation and debounce events."""

    def __init__(
        self,
        window_size: int = config.TEMPORAL_WINDOW_SIZE,
        min_confirmations: int = config.TEMPORAL_CONFIRMATION_FRAMES,
        iou_threshold: float = 0.20,
        track_ttl_sec: float = 2.0,
        event_cooldown_sec: float = config.EVENT_DEBOUNCE_COOLDOWN_SEC,
    ) -> None:
        """Initialize the temporal tracker.

        Args:
            window_size: Length of the rolling observation window.
            min_confirmations: Number of positive matches required within the window.
            iou_threshold: Minimum IoU to associate detections with existing tracks (calibrated for small distant faces).
            track_ttl_sec: Max seconds to retain track without any detection.
            event_cooldown_sec: Minimum seconds before a duplicate event can fire for the same person.
        """
        self.window_size = window_size
        self.min_confirmations = min_confirmations
        self.iou_threshold = iou_threshold
        self.track_ttl_sec = track_ttl_sec
        self.event_cooldown_sec = event_cooldown_sec

        self._next_track_id: int = 1
        self._tracks: Dict[int, TrackState] = {}
        self._active_identities: Set[str] = set()

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], str, float, bool]],
        current_time: Optional[float] = None,
    ) -> List[Tuple[Tuple[int, int, int, int], str, float, bool, bool]]:
        """Update tracker with raw detections and return stabilized results.

        Args:
            detections: List of (bbox, raw_identity, raw_score, is_ambiguous).
            current_time: Timestamp in seconds (defaults to time.perf_counter()).

        Returns:
            List of (bbox, stabilized_identity, display_score, is_confirmed, is_new_event).
        """
        now = time.perf_counter() if current_time is None else current_time

        # 1. Prune expired tracks
        expired_ids = [
            tid for tid, trk in self._tracks.items()
            if (now - trk.last_seen_time) > self.track_ttl_sec
        ]
        for tid in expired_ids:
            old_trk = self._tracks.pop(tid)
            if old_trk.confirmed_identity and old_trk.confirmed_identity in self._active_identities:
                self._active_identities.discard(old_trk.confirmed_identity)

        if not detections:
            return []

        # 2. Greedy IoU matching between detections and existing tracks
        det_bboxes = [d[0] for d in detections]
        track_ids = list(self._tracks.keys())
        assigned_track_for_det: Dict[int, int] = {}
        used_tracks: Set[int] = set()

        if track_ids:
            # Build cost/iou matrix
            iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)
            for d_idx, bbox in enumerate(det_bboxes):
                for t_idx, tid in enumerate(track_ids):
                    iou_matrix[d_idx, t_idx] = compute_iou(bbox, self._tracks[tid].bbox)

            # Assign greedily by highest IoU
            while True:
                max_iou = float(np.max(iou_matrix)) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break
                d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[t_idx]
                assigned_track_for_det[d_idx] = tid
                used_tracks.add(tid)
                # Invalidate matched row and column
                iou_matrix[d_idx, :] = -1.0
                iou_matrix[:, t_idx] = -1.0

            # 2b. Secondary spatial proximity matching for remaining unassigned detections (distant faces moving naturally)
            unassigned_dets = [i for i in range(len(detections)) if i not in assigned_track_for_det]
            remaining_tracks = [tid for tid in track_ids if tid not in used_tracks]
            for d_idx in unassigned_dets:
                bbox = det_bboxes[d_idx]
                dcx = (bbox[0] + bbox[2]) * 0.5
                dcy = (bbox[1] + bbox[3]) * 0.5
                dw = max(1, bbox[2] - bbox[0])
                dh = max(1, bbox[3] - bbox[1])
                diag = max(dw, dh)

                best_tid = None
                best_dist = float("inf")
                for tid in remaining_tracks:
                    tb = self._tracks[tid].bbox
                    tcx = (tb[0] + tb[2]) * 0.5
                    tcy = (tb[1] + tb[3]) * 0.5
                    tw = max(1, tb[2] - tb[0])
                    th = max(1, tb[3] - tb[1])

                    # Scale compatibility
                    scale_ratio = max(dw, dh) / max(tw, th)
                    if not (0.5 <= scale_ratio <= 2.0):
                        continue

                    dist = float(np.hypot(dcx - tcx, dcy - tcy))
                    if dist <= diag * 0.85 and dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None:
                    assigned_track_for_det[d_idx] = best_tid
                    remaining_tracks.remove(best_tid)
                    used_tracks.add(best_tid)

        # 3. Process each detection with assigned or new track
        has_mask_info = any(len(d) >= 5 for d in detections)
        results = []
        for d_idx, d in enumerate(detections):
            bbox = d[0]
            raw_id = d[1]
            score = d[2]
            is_ambiguous = d[3]
            is_masked = bool(d[4]) if len(d) >= 5 else False

            if d_idx in assigned_track_for_det:
                tid = assigned_track_for_det[d_idx]
                trk = self._tracks[tid]
                trk.bbox = bbox
                trk.last_seen_time = now
                trk.last_is_masked = is_masked
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                trk = TrackState(
                    track_id=tid,
                    bbox=bbox,
                    last_seen_time=now,
                    last_is_masked=is_masked,
                )
                self._tracks[tid] = trk

            # Track mask state history for stabilized status
            trk.mask_history.append(is_masked)
            if len(trk.mask_history) > self.window_size:
                trk.mask_history.popleft()
            stable_is_masked = (sum(trk.mask_history) > (len(trk.mask_history) // 2)) if trk.mask_history else is_masked

            # Append to history window (skip uninformative quality failure frames where score <= 0.0)
            if score > 0.0 or trk.confirmed_identity is None:
                if is_ambiguous:
                    trk.history.append(("Unknown Person", score))
                else:
                    trk.history.append((raw_id, score))

                if len(trk.history) > self.window_size:
                    trk.history.popleft()

            # Analyze history window for temporal confirmation
            id_counts: Dict[str, int] = {}
            id_score_sums: Dict[str, float] = {}
            for hist_id, hist_score in trk.history:
                id_counts[hist_id] = id_counts.get(hist_id, 0) + 1
                id_score_sums[hist_id] = id_score_sums.get(hist_id, 0.0) + hist_score

            # Find top non-unknown identity candidate
            known_candidates = [
                (ident, count, id_score_sums[ident] / count)
                for ident, count in id_counts.items()
                if not ident.startswith("Unknown Person")
            ]
            known_candidates.sort(key=lambda x: -x[1])

            is_confirmed = False
            is_new_event = False
            final_identity = "Unknown Person"
            final_score = score

            if known_candidates:
                top_id, count, avg_score = known_candidates[0]
                if count >= self.min_confirmations:
                    final_identity = top_id
                    final_score = avg_score
                    is_confirmed = True
                    trk.confirmed_identity = top_id
                    trk.consecutive_unknown_count = 0

                    # Check for duplicate event suppression
                    time_since_last_event = now - trk.last_event_time
                    if (not trk.event_emitted) or (time_since_last_event > self.event_cooldown_sec):
                        is_new_event = True
                        trk.event_emitted = True
                        trk.last_event_time = now
                        self._active_identities.add(top_id)
                elif trk.confirmed_identity == top_id and trk.consecutive_unknown_count <= 2:
                    # Maintain confirmed identity across transient 1-2 frame dip on the same verified track
                    final_identity = top_id
                    final_score = avg_score
                    is_confirmed = True
                else:
                    # Still verifying across frames
                    final_identity = f"Verifying ({top_id})..."
                    final_score = avg_score
            elif trk.confirmed_identity is not None and not is_ambiguous and trk.consecutive_unknown_count <= 2:
                # Maintain confirmed identity across transient 1-2 frame dip on the same verified track
                trk.consecutive_unknown_count += 1
                final_identity = trk.confirmed_identity
                is_confirmed = True
            else:
                trk.confirmed_identity = None
                trk.consecutive_unknown_count = 0

            if has_mask_info:
                results.append((bbox, final_identity, final_score, is_confirmed, is_new_event, stable_is_masked))
            else:
                results.append((bbox, final_identity, final_score, is_confirmed, is_new_event))

        return results

    def reset(self) -> None:
        """Clear all active tracks and event state."""
        self._tracks.clear()
        self._active_identities.clear()
        self._next_track_id = 1
