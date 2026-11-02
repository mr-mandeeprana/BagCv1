"""
==========================================================
FillPac AI
Production Tracker (built-in IoU tracker)
==========================================================

Purpose
-------
Camera-specific object tracking for FillPac AI.

Architecture
------------
ONE shared YOLO Detector
        |
        v
InferenceManager
        |
        +---- Pipeline Camera 1 ---- Tracker 1
        +---- Pipeline Camera 2 ---- Tracker 2
        +---- Pipeline Camera 3 ---- Tracker 3

Important
---------
- Every Pipeline must create its OWN Tracker instance.
- Tracker state is never shared between cameras.
- Only Bag class (class_id = 0) is tracked.
- Track IDs are used for temporal association.
- Physical counting is handled separately by Counter.
- Motion history is maintained independently per camera.
==========================================================
"""

import math

import numpy as np


class _IoUAssociator:
    """
    Lightweight two-stage IoU tracker (replaces supervision.ByteTrack).

    - Stage 1: all tracks vs HIGH-confidence detections
    - Stage 2: tracks seen last frame vs LOW-confidence detections
    - New tracks start from unmatched HIGH detections
    - Constant-velocity prediction keeps fast bags matched
    - Lost tracks are kept for `max_lost` frames, then dropped
    No Kalman filter, no scipy, no supervision.
    """

    def __init__(self, high, low, new, buffer, match_thresh, frame_rate):

        self.high = float(high)
        self.low = min(float(low), self.high)
        self.new = float(new)
        self.max_lost = max(int(frame_rate / 30.0 * buffer), 1)
        self.min_iou = max(1.0 - float(match_thresh), 0.01)

        self.tracks = []
        self._next_id = 1

    # ------------------------------------------------------

    @staticmethod
    def _iou(a, b):

        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])

        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        union = area_a[:, None] + area_b[None, :] - inter

        return inter / np.maximum(union, 1e-6)

    def _match(self, tracks, boxes):
        """Greedy highest-IoU-first matching. Returns (pairs, free_tracks, free_dets)."""

        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))

        predicted = np.asarray(
            [t["box"] + np.tile(t["vel"], 2) * (min(t["lost"], 5) + 1) for t in tracks],
            dtype=np.float32,
        )

        iou = self._iou(predicted, boxes)

        rows, cols = np.nonzero(iou >= self.min_iou)
        order = np.argsort(-iou[rows, cols])

        used_r, used_c, pairs = set(), set(), []

        for k in order:
            r, c = int(rows[k]), int(cols[k])
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))

        return (
            pairs,
            [r for r in range(len(tracks)) if r not in used_r],
            [c for c in range(len(boxes)) if c not in used_c],
        )

    @staticmethod
    def _apply(track, box, score):

        old = track["box"]
        gap = track["lost"] + 1

        delta = (
            np.array(
                [
                    ((box[0] + box[2]) - (old[0] + old[2])) / 2.0,
                    ((box[1] + box[3]) - (old[1] + old[3])) / 2.0,
                ],
                dtype=np.float32,
            )
            / gap
        )

        track["vel"] = 0.5 * track["vel"] + 0.5 * delta
        track["box"] = box.astype(np.float32)
        track["score"] = float(score)
        track["lost"] = 0

    # ------------------------------------------------------

    def update(self, boxes, scores):
        """
        boxes: (N,4) float32 xyxy, scores: (N,) float32.
        Returns [(track_id, box_xyxy, score), ...] for tracks matched
        or created in THIS frame.
        """

        out = []

        high_mask = scores >= self.high
        low_mask = (scores >= self.low) & ~high_mask

        high_idx = np.nonzero(high_mask)[0]
        low_idx = np.nonzero(low_mask)[0]

        high_boxes = boxes[high_idx]
        low_boxes = boxes[low_idx]

        # ---- stage 1 ----
        pairs, free_tracks, free_high = self._match(self.tracks, high_boxes)

        for r, c in pairs:
            det = high_idx[c]
            self._apply(self.tracks[r], boxes[det], scores[det])
            out.append(self.tracks[r])

        # ---- stage 2 ----
        retry = [self.tracks[r] for r in free_tracks if self.tracks[r]["lost"] == 0]

        pairs2, _, _ = self._match(retry, low_boxes)

        matched_ids = set()

        for r, c in pairs2:
            det = low_idx[c]
            self._apply(retry[r], boxes[det], scores[det])
            matched_ids.add(retry[r]["id"])
            out.append(retry[r])

        # ---- lifecycle ----
        seen = {t["id"] for t in out}

        for t in self.tracks:
            if t["id"] not in seen:
                t["lost"] += 1

        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]

        # ---- new tracks ----
        for c in free_high:
            det = high_idx[c]

            if scores[det] < self.new:
                continue

            track = {
                "id": self._next_id,
                "box": boxes[det].astype(np.float32),
                "vel": np.zeros(2, dtype=np.float32),
                "score": float(scores[det]),
                "lost": 0,
            }

            self._next_id += 1
            self.tracks.append(track)
            out.append(track)

        return [(t["id"], t["box"], t["score"]) for t in out]


class Tracker:
    """
    Camera-specific tracker wrapper.

    Adds:
    - Track age filtering
    - Confidence filtering
    - Bounding-box area filtering
    - Relative bounding-box area filtering
    - Motion tracking
    - Speed estimation in pixels/frame
    - Direction estimation
    - Cumulative distance
    - Bounding-box IoU monitoring
    - Motion jump detection
    - Track instability detection
    - Automatic history cleanup
    """

    def __init__(self, tracker_config=None):
        tracker_config = tracker_config or {}

        # ==================================================
        # TRACK QUALITY CONFIGURATION
        # ==================================================

        self.min_track_age = max(int(tracker_config.get("min_track_age", 4)), 1)

        self.min_bbox_area = max(float(tracker_config.get("min_bbox_area", 4000)), 0.0)

        self.min_bbox_area_ratio = max(float(tracker_config.get("min_bbox_area_ratio", 0.0)), 0.0)

        self.min_track_confidence = min(
            max(float(tracker_config.get("min_track_confidence", 0.25)), 0.0), 1.0
        )

        # ==================================================
        # FRAME DIMENSIONS
        #
        # If dimensions are not configured, they can be
        # learned dynamically from detections/frame metadata
        # later using set_frame_size().
        # ==================================================

        self.frame_width = max(int(tracker_config.get("frame_width", 0) or 0), 0)

        self.frame_height = max(int(tracker_config.get("frame_height", 0) or 0), 0)

        # ==================================================
        # MOTION QUALITY CONFIGURATION
        # ==================================================

        self.max_jump_distance = max(float(tracker_config.get("max_jump_distance", 250)), 0.0)

        self.min_iou_warning = min(
            max(float(tracker_config.get("min_iou_warning", 0.10)), 0.0), 1.0
        )

        self.unstable_frame_threshold = max(
            int(tracker_config.get("unstable_frame_threshold", 3)), 1
        )

        self.history_ttl_frames = max(int(tracker_config.get("history_ttl_frames", 120)), 1)

        # ==================================================
        # BYTETRACK CONFIGURATION
        # ==================================================

        track_high_thresh = float(tracker_config.get("track_high_thresh", 0.5))

        track_buffer = max(int(tracker_config.get("track_buffer", 30)), 1)

        match_thresh = float(tracker_config.get("match_thresh", 0.8))

        frame_rate = max(int(tracker_config.get("frame_rate", 30)), 1)

        # ==================================================
        # FRAME INDEX
        # ==================================================

        self.frame_index = 0

        # ==================================================
        # TRACK MEMORY
        #
        # All dictionaries are local to THIS Tracker
        # instance and therefore local to ONE camera.
        # ==================================================

        self.track_age = {}

        self.track_last_seen = {}

        self.track_last_center = {}

        self.track_last_bbox = {}

        self.track_velocity = {}

        self.track_speed = {}

        self.track_direction = {}

        self.track_distance = {}

        self.track_iou = {}

        self.track_unstable = {}

        self.track_unstable_count = {}

        self.track_motion_jump = {}

        # ==================================================
        # CREATE CAMERA-SPECIFIC BYTETRACK
        # ==================================================

        self.associator = _IoUAssociator(
            high=track_high_thresh,
            low=tracker_config.get("track_low_thresh", 0.05),
            new=tracker_config.get("new_track_thresh", track_high_thresh),
            buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

    # ======================================================
    # SET FRAME SIZE
    # ======================================================

    def set_frame_size(self, width, height):
        """
        Set actual camera frame dimensions.

        This allows min_bbox_area_ratio to work correctly.

        Safe to call repeatedly.
        """

        width = int(width or 0)

        height = int(height or 0)

        if width > 0:
            self.frame_width = width

        if height > 0:
            self.frame_height = height

    # ======================================================
    # UPDATE TRACKER
    # ======================================================

    def update(self, detections):
        """
        Update tracker with current bag detections.

        Parameters
        ----------
        detections:
            List of detection dictionaries returned by
            Detector / InferenceManager.

        Returns
        -------
        list
            Stable tracked bag dictionaries.
        """

        self.frame_index += 1

        # ==================================================
        # KEEP ONLY BAG DETECTIONS
        # ==================================================

        bags = [detection for detection in (detections or []) if detection.get("class_id") == 0]

        # ==================================================
        # NO DETECTIONS
        # ==================================================

        if bags:
            xyxy = np.asarray([bag["bbox"] for bag in bags], dtype=np.float32)
            scores = np.asarray([bag.get("confidence", 0.0) for bag in bags], dtype=np.float32)
        else:
            # Empty frame still advances the lost-track lifecycle.
            xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)

        observations = self.associator.update(xyxy, scores)

        if not observations:
            self._trim_history(active_track_ids=set())
            return []

        tracked_bags = []

        active_track_ids = set()

        for track_id, box, confidence_value in observations:

            bbox = tuple(box.astype(int))

            center = self.get_center(bbox)

            active_track_ids.add(track_id)

            # ==================================================
            # TRACK AGE
            #
            # Track age counts tracker observations,
            # regardless of whether the current bounding box
            # passes our downstream quality filters.
            # ==================================================

            self.track_age[track_id] = self.track_age.get(track_id, 0) + 1

            self.track_last_seen[track_id] = self.frame_index

            # ==================================================
            # PREVIOUS MEMORY
            #
            # Capture previous values BEFORE updating.
            # ==================================================

            previous_center = self.track_last_center.get(track_id)

            previous_bbox = self.track_last_bbox.get(track_id)

            # ==================================================
            # IMPORTANT FIX
            #
            # Always update motion memory for a valid tracker
            # observation BEFORE applying our downstream quality
            # filters.
            #
            # Otherwise:
            #
            # Frame 1 valid
            # Frame 2 filtered
            # Frame 3 filtered
            # Frame 4 valid
            #
            # Frame 4 would compare against Frame 1 and could
            # incorrectly appear as a large motion jump.
            # ==================================================

            self._update_track_memory(
                track_id=track_id,
                bbox=bbox,
                center=center,
                previous_center=previous_center,
                previous_bbox=previous_bbox,
            )

            # ==================================================
            # QUALITY FILTERS
            # ==================================================

            if not self._passes_quality_filters(bbox, confidence_value):

                continue

            # ==================================================
            # MINIMUM TRACK AGE
            # ==================================================

            if self.track_age[track_id] < self.min_track_age:

                continue

            # ==================================================
            # BUILD TRACK RESULT
            # ==================================================

            tracked_bags.append(
                {
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": confidence_value,
                    "class_id": 0,
                    "center": center,
                    # Pixels moved since previous
                    # processed tracker observation.
                    "speed": self.track_speed.get(track_id, 0.0),
                    "direction": self.track_direction.get(track_id, "stationary"),
                    # Total cumulative movement in pixels.
                    "distance": self.track_distance.get(track_id, 0.0),
                    "iou": self.track_iou.get(track_id, 1.0),
                    "unstable": self.track_unstable.get(track_id, False),
                    "motion_jump": self.track_motion_jump.get(track_id, False),
                    # Useful for debugging / future dashboard.
                    "track_age": self.track_age.get(track_id, 0),
                }
            )

        # ==================================================
        # CLEAN OLD TRACK MEMORY
        # ==================================================

        self._trim_history(active_track_ids)

        return tracked_bags

    # ======================================================
    # GET CENTER
    # ======================================================

    @staticmethod
    def get_center(bbox):

        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)

        center_y = int((y1 + y2) / 2)

        return (center_x, center_y)

    # ======================================================
    # QUALITY FILTERS
    # ======================================================

    def _passes_quality_filters(self, bbox, confidence):

        # ----------------------------------------------
        # Confidence
        # ----------------------------------------------

        if confidence < self.min_track_confidence:

            return False

        # ----------------------------------------------
        # Bounding Box Area
        # ----------------------------------------------

        x1, y1, x2, y2 = bbox

        width = max(x2 - x1, 0)

        height = max(y2 - y1, 0)

        area = width * height

        minimum_area = self._minimum_bbox_area()

        if area < minimum_area:

            return False

        return True

    # ======================================================
    # MINIMUM BBOX AREA
    # ======================================================

    def _minimum_bbox_area(self):

        # ----------------------------------------------
        # Absolute minimum
        # ----------------------------------------------

        minimum_area = self.min_bbox_area

        # ----------------------------------------------
        # Relative frame-area minimum
        # ----------------------------------------------

        if self.frame_width > 0 and self.frame_height > 0 and self.min_bbox_area_ratio > 0:

            frame_area = self.frame_width * self.frame_height

            relative_area = frame_area * self.min_bbox_area_ratio

            minimum_area = max(minimum_area, relative_area)

        return minimum_area

    # ======================================================
    # MOTION JUMP
    # ======================================================

    def _is_motion_jump(self, previous_center, center):

        if previous_center is None:

            return False

        if self.max_jump_distance <= 0:

            return False

        distance = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])

        return distance > self.max_jump_distance

    # ======================================================
    # UPDATE TRACK MEMORY
    # ======================================================

    def _update_track_memory(self, track_id, bbox, center, previous_center, previous_bbox):

        # ----------------------------------------------
        # Velocity / Distance
        # ----------------------------------------------

        if previous_center is None:

            velocity = (0, 0)

            frame_distance = 0.0

        else:

            velocity = (center[0] - previous_center[0], center[1] - previous_center[1])

            frame_distance = math.hypot(velocity[0], velocity[1])

        # ----------------------------------------------
        # Bounding Box IoU
        # ----------------------------------------------

        iou = self._bbox_iou(previous_bbox, bbox)

        # ----------------------------------------------
        # Motion Jump
        # ----------------------------------------------

        motion_jump = self._is_motion_jump(previous_center, center)

        # ----------------------------------------------
        # Low IoU
        # ----------------------------------------------

        low_iou = previous_bbox is not None and iou < self.min_iou_warning

        # ----------------------------------------------
        # Store Current Memory
        # ----------------------------------------------

        self.track_last_center[track_id] = center

        self.track_last_bbox[track_id] = bbox

        self.track_velocity[track_id] = velocity

        # IMPORTANT:
        #
        # Current speed means pixels moved between
        # consecutive tracker observations.
        #
        # This is pixels/frame, NOT pixels/second.
        self.track_speed[track_id] = frame_distance

        self.track_direction[track_id] = self._motion_direction(velocity)

        self.track_distance[track_id] = self.track_distance.get(track_id, 0.0) + frame_distance

        self.track_iou[track_id] = iou

        self.track_motion_jump[track_id] = motion_jump

        # ==================================================
        # INSTABILITY COUNTER
        #
        # Require consecutive bad observations before
        # declaring a track unstable.
        # ==================================================

        if motion_jump or low_iou:

            self.track_unstable_count[track_id] = self.track_unstable_count.get(track_id, 0) + 1

        else:

            self.track_unstable_count[track_id] = 0

        self.track_unstable[track_id] = (
            self.track_unstable_count.get(track_id, 0) >= self.unstable_frame_threshold
        )

    # ======================================================
    # BOUNDING BOX IOU
    # ======================================================

    @staticmethod
    def _bbox_iou(previous_bbox, bbox):

        if previous_bbox is None:

            return 1.0

        ax1, ay1, ax2, ay2 = previous_bbox

        bx1, by1, bx2, by2 = bbox

        # ----------------------------------------------
        # Intersection
        # ----------------------------------------------

        inter_x1 = max(ax1, bx1)

        inter_y1 = max(ay1, by1)

        inter_x2 = min(ax2, bx2)

        inter_y2 = min(ay2, by2)

        inter_width = max(inter_x2 - inter_x1, 0)

        inter_height = max(inter_y2 - inter_y1, 0)

        intersection = inter_width * inter_height

        # ----------------------------------------------
        # Areas
        # ----------------------------------------------

        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)

        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)

        union = area_a + area_b - intersection

        if union <= 0:

            return 0.0

        return intersection / union

    # ======================================================
    # MOTION DIRECTION
    # ======================================================

    @staticmethod
    def _motion_direction(velocity):

        dx, dy = velocity

        if dx == 0 and dy == 0:

            return "stationary"

        # ----------------------------------------------
        # Vertical movement dominates
        # ----------------------------------------------

        if abs(dy) >= abs(dx):

            if dy > 0:

                return "down"

            return "up"

        # ----------------------------------------------
        # Horizontal movement dominates
        # ----------------------------------------------

        if dx > 0:

            return "right"

        return "left"

    # ======================================================
    # TRIM TRACK HISTORY
    # ======================================================

    def _trim_history(self, active_track_ids):

        # ----------------------------------------------
        # Tracks recently seen by the tracker
        # ----------------------------------------------

        fresh_ids = {
            track_id
            for (track_id, last_seen) in self.track_last_seen.items()
            if (self.frame_index - last_seen <= self.history_ttl_frames)
        }

        retained_ids = set(active_track_ids) | fresh_ids

        # ----------------------------------------------
        # Track Age
        # ----------------------------------------------

        self.track_age = {
            track_id: value
            for (track_id, value) in self.track_age.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Last Seen
        # ----------------------------------------------

        self.track_last_seen = {
            track_id: value
            for (track_id, value) in self.track_last_seen.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Last Center
        # ----------------------------------------------

        self.track_last_center = {
            track_id: value
            for (track_id, value) in self.track_last_center.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Last Bounding Box
        # ----------------------------------------------

        self.track_last_bbox = {
            track_id: value
            for (track_id, value) in self.track_last_bbox.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Velocity
        # ----------------------------------------------

        self.track_velocity = {
            track_id: value
            for (track_id, value) in self.track_velocity.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Speed
        # ----------------------------------------------

        self.track_speed = {
            track_id: value
            for (track_id, value) in self.track_speed.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Direction
        # ----------------------------------------------

        self.track_direction = {
            track_id: value
            for (track_id, value) in self.track_direction.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Distance
        # ----------------------------------------------

        self.track_distance = {
            track_id: value
            for (track_id, value) in self.track_distance.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # IoU
        # ----------------------------------------------

        self.track_iou = {
            track_id: value
            for (track_id, value) in self.track_iou.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Unstable
        # ----------------------------------------------

        self.track_unstable = {
            track_id: value
            for (track_id, value) in self.track_unstable.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Unstable Count
        # ----------------------------------------------

        self.track_unstable_count = {
            track_id: value
            for (track_id, value) in self.track_unstable_count.items()
            if track_id in retained_ids
        }

        # ----------------------------------------------
        # Motion Jump
        # ----------------------------------------------

        self.track_motion_jump = {
            track_id: value
            for (track_id, value) in self.track_motion_jump.items()
            if track_id in retained_ids
        }