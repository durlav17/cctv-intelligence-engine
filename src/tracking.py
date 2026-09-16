"""
CCTV Intelligence Engine - Phase 2: track lifecycle bookkeeping.

This module knows NOTHING about YOLO, OpenCV or drawing. It is fed the raw
(tracker_id, class_name, confidence, bbox) tuples that a tracker produced for
one frame, and it maintains the persistent record for each tracked person.

Keeping it separate means:
  - detection code stays detection code (Phase 1 is untouched),
  - the tracker can be swapped (BoT-SORT <-> ByteTrack) without touching this,
  - the records below are already shaped like the rows a future database
    would store (see DATA MODEL note at the bottom).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# Track lifecycle states.
STATE_ACTIVE = "active"          # matched to a detection in the current frame
STATE_LOST = "lost"              # not seen right now, but still inside the grace window
STATE_TERMINATED = "terminated"  # gone for longer than the grace window; ID retired


@dataclass
class Track:
    """Everything we know about one tracked person during one video run.

    `label` is the anonymous display name (P01, P02, ...). `tracker_id` is the
    raw integer the tracker handed us - we keep both so the on-screen label is
    stable and human-readable while still being traceable to tracker output.
    """

    tracker_id: int
    label: str
    class_name: str

    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp: float   # seconds from start of video
    last_seen_timestamp: float

    confidence: float = 0.0       # confidence of the most recent detection
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # most recent x1,y1,x2,y2

    frames_tracked: int = 0       # frames in which this track was actually matched
    state: str = STATE_ACTIVE
    frames_since_seen: int = 0    # how long it has been unmatched

    # Lifecycle history: ("lost", frame) / ("reacquired", frame) / ("terminated", frame).
    # We record failures instead of pretending tracking is perfect.
    events: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock span from first to last sighting, in video time."""
        return self.last_seen_timestamp - self.first_seen_timestamp

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_seconds"] = round(self.duration_seconds, 2)
        return d


class TrackRegistry:
    """Owns all Track objects for one video run and advances their lifecycle.

    Usage, once per frame:

        registry.update(detections, frame_index, timestamp)
        for track in registry.visible_tracks():
            ...draw it...

    `max_lost_frames` is the grace window: how many consecutive frames a track
    may go unmatched (occlusion, a missed detection) before we declare it
    terminated. Too small -> occlusions create new IDs. Too large -> a stale ID
    can be handed to a different person.
    """

    def __init__(self, max_lost_frames: int = 30) -> None:
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, Track] = {}    # tracker_id -> Track (active + lost)
        self.finished: Dict[int, Track] = {}  # tracker_id -> Track (terminated)
        self._next_label_number = 1

    # ---------- labelling ----------

    def _new_label(self) -> str:
        """Anonymous, non-identifying display name: P01, P02, ... P10, P11."""
        label = "P{:02d}".format(self._next_label_number)
        self._next_label_number += 1
        return label

    # ---------- per-frame update ----------

    def update(self, detections: List[dict], frame_index: int, timestamp: float) -> None:
        """Fold one frame's tracker output into the registry.

        Args:
            detections: list of dicts with keys
                tracker_id (int), class_name (str), confidence (float),
                bbox (x1, y1, x2, y2).
            frame_index: 0-based index of the current frame.
            timestamp: seconds from the start of the video for this frame.
        """
        seen_now = set()

        for det in detections:
            tid = det["tracker_id"]
            seen_now.add(tid)
            track = self.tracks.get(tid)

            if track is None:
                # NEW TRACK - first time this tracker_id has ever appeared.
                track = Track(
                    tracker_id=tid,
                    label=self._new_label(),
                    class_name=det["class_name"],
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                    first_seen_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                )
                track.events.append(("created", frame_index))
                self.tracks[tid] = track
            elif track.state == STATE_LOST:
                # REACQUIRED - it was inside the grace window and came back.
                track.events.append(("reacquired", frame_index))

            # ACTIVE TRACK - refresh the mutable fields.
            track.state = STATE_ACTIVE
            track.frames_since_seen = 0
            track.frames_tracked += 1
            track.last_seen_frame = frame_index
            track.last_seen_timestamp = timestamp
            track.confidence = det["confidence"]
            track.bbox = det["bbox"]

        # Anything not matched this frame ages towards termination.
        for tid, track in list(self.tracks.items()):
            if tid in seen_now:
                continue
            track.frames_since_seen += 1
            if track.state == STATE_ACTIVE:
                # TEMPORARILY LOST - occlusion, or the detector simply missed it.
                track.state = STATE_LOST
                track.events.append(("lost", frame_index))
            if track.frames_since_seen > self.max_lost_frames:
                # TERMINATED - retire the ID; it will never be reused.
                track.state = STATE_TERMINATED
                track.events.append(("terminated", frame_index))
                self.finished[tid] = self.tracks.pop(tid)

    def close(self, frame_index: int) -> None:
        """End of video: terminate everything still open, so the summary is complete."""
        for tid, track in list(self.tracks.items()):
            track.state = STATE_TERMINATED
            track.events.append(("terminated_end_of_video", frame_index))
            self.finished[tid] = self.tracks.pop(tid)

    # ---------- queries ----------

    def visible_tracks(self) -> List[Track]:
        """Tracks matched in the current frame - the ones worth drawing."""
        return [t for t in self.tracks.values() if t.state == STATE_ACTIVE]

    def lost_tracks(self) -> List[Track]:
        """Tracks inside the grace window, currently unmatched."""
        return [t for t in self.tracks.values() if t.state == STATE_LOST]

    def all_tracks(self) -> List[Track]:
        """Every track ever created, in first-seen order."""
        combined = list(self.finished.values()) + list(self.tracks.values())
        return sorted(combined, key=lambda t: t.first_seen_frame)

    # ---------- reporting ----------

    def summary_lines(self, min_frames: int = 1) -> List[str]:
        """Human-readable end-of-run summary.

        `min_frames` filters out one-frame flickers, which are almost always
        false detections rather than real people.
        """
        every = self.all_tracks()
        tracks = [t for t in every if t.frames_tracked >= min_frames]

        lines = ["", "TRACKING SUMMARY", ""]
        lines.append("unique track IDs created: {}".format(len(every)))
        lines.append("tracks with >= {} frames: {}".format(min_frames, len(tracks)))
        lines.append("")

        for t in tracks:
            lost_count = sum(1 for name, _ in t.events if name == "lost")
            reacquired = sum(1 for name, _ in t.events if name == "reacquired")
            lines.append("{}:".format(t.label))
            lines.append("  frames tracked: {}".format(t.frames_tracked))
            lines.append("  duration: {:.1f} sec".format(t.duration_seconds))
            lines.append("  frame range: {} -> {}".format(t.first_seen_frame, t.last_seen_frame))
            lines.append("  last confidence: {:.2f}".format(t.confidence))
            lines.append("  times lost: {} (reacquired: {})".format(lost_count, reacquired))
            lines.append("")
        return lines

    def to_records(self) -> List[dict]:
        """Flat dicts, ready to be written to JSON now or to a DB later."""
        return [t.to_dict() for t in self.all_tracks()]


# DATA MODEL NOTE (no database in this phase - this is only the shape)
# -------------------------------------------------------------------
# A Track above maps cleanly onto future tables:
#
#   visitor/session : one processing run over one video / camera stream.
#                     (video_path, started_at, tracker, model, source fps)
#   track           : one row per Track - tracker_id, label, class_name,
#                     first/last_seen_frame, first/last_seen_timestamp,
#                     frames_tracked, confidence.
#   visit           : a contiguous ACTIVE span of a track - derivable from the
#                     created/lost/reacquired events, so a person occluded
#                     twice yields three visit rows under one track row.
#   event           : one row per entry in Track.events -
#                     (track_id, event_type, frame_index).
#
# Deliberately NOT modelled yet: any cross-video or cross-day identity. A track
# row is only meaningful inside its own session row.
