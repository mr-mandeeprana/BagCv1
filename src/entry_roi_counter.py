"""
Entry ROI Counter (CPU-optimized)

Counts a bag once per Track ID when the bag CENTER is inside the
configured rectangular ROI and the track is stable
(seen on `min_track_frames` frames).

Bounding-box edges and entry distance are NOT used, so two bags
entering very close together both count.
"""


class EntryROICounter:

    def __init__(
        self,
        roi,
        min_track_frames=4,
        max_history=300,
        duplicate_distance=60,  # accepted for config compatibility, unused
    ):
        if not isinstance(roi, dict):
            raise ValueError("Entry ROI must be a dictionary containing x1, y1, x2, y2.")

        self.x1 = float(roi.get("x1", 0))
        self.y1 = float(roi.get("y1", 0))
        self.x2 = float(roi.get("x2", 0))
        self.y2 = float(roi.get("y2", 0))

        if self.x1 == self.x2 or self.y1 == self.y2:
            raise ValueError("Entry ROI must be a rectangle with non-zero width and height.")

        self.x_min = min(self.x1, self.x2)
        self.x_max = max(self.x1, self.x2)
        self.y_min = min(self.y1, self.y2)
        self.y_max = max(self.y1, self.y2)

        self.min_track_frames = max(int(min_track_frames), 1)
        self.max_history = max(int(max_history), 1)

        self.total_count = 0

        # Per-track state (read by Pipeline)
        self.track_frame_count = {}
        self.track_centers = {}
        self.track_inside_roi = {}
        self.counted_track_ids = set()

        # Bags counted in the CURRENT frame
        self.last_counted_bags = []

    # ------------------------------------------------------

    @staticmethod
    def get_center(bbox):
        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox

        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    # ------------------------------------------------------

    def update(self, tracks):
        """
        tracks: [{"track_id": 7, "class_id": 0, "bbox": [x1,y1,x2,y2]}]
        Returns total_count.
        """

        self.last_counted_bags = []

        if not tracks:
            return self.total_count

        active_track_ids = set()

        frame_count = self.track_frame_count
        centers = self.track_centers
        inside_map = self.track_inside_roi
        counted = self.counted_track_ids

        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        min_frames = self.min_track_frames

        for track in tracks:

            if not isinstance(track, dict) or track.get("class_id") != 0:
                continue

            track_id = track.get("track_id")

            if track_id is None:
                continue

            active_track_ids.add(track_id)

            bbox = track.get("bbox")
            center = self.get_center(bbox)

            if center is None:
                continue

            frames_seen = frame_count.get(track_id, 0) + 1
            frame_count[track_id] = frames_seen

            inside = x_min <= center[0] <= x_max and y_min <= center[1] <= y_max

            centers[track_id] = center
            inside_map[track_id] = inside

            if frames_seen < min_frames or not inside or track_id in counted:
                continue

            self.total_count += 1
            counted.add(track_id)

            self.last_counted_bags.append({"track_id": track_id, "bbox": bbox, "center": center})

        if len(centers) > self.max_history:
            self._cleanup(active_track_ids)

        return self.total_count

    # ------------------------------------------------------

    def _cleanup(self, active_track_ids):
        """Bound memory in place. Active tracks are never removed."""

        stale = [track_id for track_id in self.track_centers if track_id not in active_track_ids]

        # Keep the newest `max_history` stale IDs (dicts keep insertion order).
        for track_id in stale[: max(len(stale) - self.max_history, 0)]:
            self.track_centers.pop(track_id, None)
            self.track_frame_count.pop(track_id, None)
            self.track_inside_roi.pop(track_id, None)
            self.counted_track_ids.discard(track_id)

    # ------------------------------------------------------

    def reset(self):
        self.total_count = 0
        self.track_frame_count.clear()
        self.track_centers.clear()
        self.track_inside_roi.clear()
        self.counted_track_ids.clear()
        self.last_counted_bags = []

    @property
    def roi(self):
        return {"x1": int(self.x1), "y1": int(self.y1), "x2": int(self.x2), "y2": int(self.y2)}