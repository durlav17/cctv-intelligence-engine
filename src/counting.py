"""
CCTV Intelligence Engine - Phase 3: spatial reasoning and line crossing.

Pure geometry and state machine. No OpenCV, no YOLO, no file I/O - so every
rule in here can be unit-tested with plain tuples and integers.

Responsibilities (deliberately narrow):
  - reference_point()      : where on a person we consider them to "be"
  - CameraConfig           : counting line, deadband, entry direction
  - CameraConfig.side_of() : which side of the line a point is on
  - LineCrossingDetector   : per-track side history -> CrossingEvent

What it explicitly does NOT do: occupancy, sessions, drawing, persistence.
Those live in sessions.py and track_people.py.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
Point = Tuple[int, int]

# Which side of the horizontal counting line a reference point sits on.
SIDE_ABOVE = "ABOVE"    # smaller y - image coordinates grow downwards
SIDE_BELOW = "BELOW"    # larger y
SIDE_BUFFER = "BUFFER"  # inside the deadband; deliberately undecided

# Direction of travel across the line.
TOP_TO_BOTTOM = "TOP_TO_BOTTOM"
BOTTOM_TO_TOP = "BOTTOM_TO_TOP"
DIRECTIONS = (TOP_TO_BOTTOM, BOTTOM_TO_TOP)

# Event types produced by a crossing.
EVENT_ENTRY = "ENTRY"
EVENT_EXIT = "EXIT"

# Observation status of a track relative to the counting boundary.
#
# OBSERVATION STATUS / FIRST-SIGHTING POLICY
# -------------------------------------------
# A person can be visible in frame 0, or walk in from the side, without ever
# having crossed the boundary in view. Counting a first sighting as an ENTRY
# would inflate every total - on a busy scene, by more than the real count.
#
# So: a track begins as INITIAL_VISIBLE. Its first decidable side is recorded
# as a *baseline*, never as an event. Only once the track is observed moving
# from one confirmed side to the other does it become OBSERVED_CROSSING and
# start producing events. A track that is INITIAL_VISIBLE at the end of the
# video was simply never seen to cross, which is a fact worth reporting, not a
# missing entry to be guessed at.
STATUS_INITIAL_VISIBLE = "INITIAL_VISIBLE"
STATUS_OBSERVED_CROSSING = "OBSERVED_CROSSING"


def reference_point(bbox: BBox) -> Point:
    """Ground-plane reference point for a person: bottom-center of the box.

        x = (x1 + x2) / 2
        y = y2

    Why bottom-center rather than the box center:

    A counting line drawn on a CCTV image is really a line on the *floor*. The
    only part of a person's bounding box that reliably touches the floor is its
    bottom edge - that is where their feet are. The box center floats at roughly
    chest height, so it crosses a floor line well before the person's feet do,
    and the error grows with how tall the person appears (i.e. how close they
    are to the camera). Two people crossing the same floor line at the same
    moment would be counted at different times if we used centers.

    Bottom-center also degrades more gracefully: when someone is partially
    occluded from the waist up the box shrinks towards the feet, whereas an
    occluded lower body moves the center but takes the feet with it.

    Caveat worth knowing: if a person is cut off by the bottom edge of the
    frame, y2 clamps to the frame height and the reference point stops tracking
    their feet. Put counting lines away from the frame border.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


@dataclass(frozen=True)
class CameraConfig:
    """Per-camera counting geometry.

    Kept as a small frozen dataclass rather than a config framework: it is one
    camera's worth of numbers, and a future multi-camera setup is a list of
    these, loaded from wherever, without changing any logic below.

    Attributes:
        line_y: image row of the horizontal counting line.
        deadband: half-height of the ignored buffer band around the line, in
            pixels. 0 disables hysteresis.
        entry_direction: which direction of travel counts as ENTRY. The other
            direction is EXIT. Not hard-coded, because a camera facing the
            other way inverts the meaning.
        camera_id: identifier for the camera this geometry belongs to. Carried
            into every event and session so records from several cameras stay
            distinguishable once they share a store.
        short_visit_seconds: visits shorter than this are *reported* as
            possible line-hover artefacts. Reporting only - nothing is
            suppressed or discarded on this basis.
    """

    line_y: int
    deadband: int = 10
    entry_direction: str = TOP_TO_BOTTOM
    camera_id: str = "camera_01"
    short_visit_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.line_y < 0:
            raise ValueError("line_y must be >= 0, got {}".format(self.line_y))
        if self.deadband < 0:
            raise ValueError("deadband must be >= 0, got {}".format(self.deadband))
        if self.entry_direction not in DIRECTIONS:
            raise ValueError(
                "entry_direction must be one of {}, got {!r}".format(
                    DIRECTIONS, self.entry_direction))
        if not self.camera_id:
            raise ValueError("camera_id must not be empty")

    # ---------- loading ----------

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        """Build a config from a plain dict (typically parsed JSON).

        Unknown keys are rejected rather than ignored: a typo like
        "counting_lines" would otherwise silently leave the line at its
        default and quietly produce wrong counts.
        """
        known = {"camera_id", "counting_line", "entry_direction", "deadband",
                 "short_visit_seconds"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                "unknown camera config key(s): {}. Known keys: {}".format(
                    ", ".join(sorted(unknown)), ", ".join(sorted(known))))
        if "counting_line" not in data:
            raise ValueError("camera config must define 'counting_line'")

        return cls(
            line_y=int(data["counting_line"]),
            deadband=int(data.get("deadband", 10)),
            entry_direction=str(data.get("entry_direction", TOP_TO_BOTTOM)),
            camera_id=str(data.get("camera_id", "camera_01")),
            short_visit_seconds=float(data.get("short_visit_seconds", 1.0)),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "CameraConfig":
        """Load a camera config from JSON, e.g. config/cameras/camera_01.json.

        Keeping geometry in a per-camera file is what makes the pipeline
        video-agnostic: the same code runs any video, and the camera file says
        where that camera's boundary is. Nothing here guesses a line for an
        unknown scene - a human still has to look at a frame and decide.
        """
        import json

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise ValueError("camera config not found: {}".format(path)) from None
        except json.JSONDecodeError as exc:
            raise ValueError(
                "camera config {} is not valid JSON: {}".format(path, exc)) from None
        if not isinstance(data, dict):
            raise ValueError(
                "camera config {} must contain a JSON object".format(path))
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError("in camera config {}: {}".format(path, exc)) from exc

    def to_dict(self) -> dict:
        """Round-trips through from_dict()."""
        return {
            "camera_id": self.camera_id,
            "counting_line": self.line_y,
            "entry_direction": self.entry_direction,
            "deadband": self.deadband,
            "short_visit_seconds": self.short_visit_seconds,
        }

    @property
    def exit_direction(self) -> str:
        """Whichever direction entry_direction is not."""
        return BOTTOM_TO_TOP if self.entry_direction == TOP_TO_BOTTOM else TOP_TO_BOTTOM

    @property
    def upper_bound(self) -> int:
        """Rows above this are unambiguously ABOVE."""
        return self.line_y - self.deadband

    @property
    def lower_bound(self) -> int:
        """Rows below this are unambiguously BELOW."""
        return self.line_y + self.deadband

    def side_of(self, point: Point) -> str:
        """Classify a reference point as ABOVE / BELOW / BUFFER.

        BUFFER is a real answer, not a failure: it means "too close to the line
        to commit", and the crossing detector treats it as "no new information".
        """
        y = point[1]
        if y < self.upper_bound:
            return SIDE_ABOVE
        if y > self.lower_bound:
            return SIDE_BELOW
        return SIDE_BUFFER

    def event_type_for(self, direction: str) -> str:
        """Map a direction of travel to ENTRY or EXIT for this camera."""
        if direction not in DIRECTIONS:
            raise ValueError("unknown direction {!r}".format(direction))
        return EVENT_ENTRY if direction == self.entry_direction else EVENT_EXIT


@dataclass
class CrossingEvent:
    """One person crossing the counting line once.

    Structured data, not a printed string - this is the row a future `event`
    table stores, and the object sessions.py consumes.
    """

    event_id: str
    event_type: str        # ENTRY | EXIT
    track_id: str          # anonymous per-run label, e.g. "P01". NOT a person.
    tracker_id: int        # raw tracker integer, for debugging
    frame_number: int
    timestamp: float       # seconds on the video timeline
    direction: str         # TOP_TO_BOTTOM | BOTTOM_TO_TOP
    reference_point: Point
    camera: str = "camera-1"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reference_point"] = list(self.reference_point)
        d["timestamp"] = round(self.timestamp, 3)
        return d


@dataclass
class TrackSideState:
    """What the detector remembers about one track's relationship to the line.

    `confirmed_side` is the last side the track was *unambiguously* on. Frames
    spent in the buffer band do not overwrite it - that is the whole hysteresis
    mechanism.

    `status` makes the first-sighting policy explicit rather than implied:
    every track starts INITIAL_VISIBLE and stays there until it is actually
    seen to cross. See the OBSERVATION STATUS note below.
    """

    confirmed_side: Optional[str] = None
    current_side: Optional[str] = None
    previous_side: Optional[str] = None
    crossings: int = 0
    status: str = "INITIAL_VISIBLE"
    first_seen_side: Optional[str] = None


class LineCrossingDetector:
    """Turns a stream of per-track reference points into CrossingEvents.

    One instance per camera. Feed it every visible track, every frame:

        event = detector.update("P01", bbox, frame_number, timestamp)

    Returns a CrossingEvent only on the frame where a confirmed side flip
    happens, and None otherwise - so it cannot emit duplicates while a person
    stands still, and cannot fire at all while they loiter inside the band.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.states: Dict[str, TrackSideState] = {}
        self._event_counter = 0

    # ---------- per-frame update ----------

    def update(
        self,
        track_id: str,
        bbox: BBox,
        frame_number: int,
        timestamp: float,
        tracker_id: int = -1,
    ) -> Optional[CrossingEvent]:
        """Feed one track's current box; return a CrossingEvent if it just crossed."""
        point = reference_point(bbox)
        side = self.config.side_of(point)

        state = self.states.get(track_id)
        if state is None:
            # First sighting: record where the track started, never count it as
            # a crossing. A person who appears below the line was not observed
            # crossing it, and inventing an event here would inflate every count.
            state = TrackSideState(
                confirmed_side=None if side == SIDE_BUFFER else side,
                current_side=side,
                status=STATUS_INITIAL_VISIBLE,
                first_seen_side=side,
            )
            self.states[track_id] = state
            return None

        state.previous_side = state.current_side
        state.current_side = side

        if side == SIDE_BUFFER:
            # No information. Keep the last confirmed side untouched, which is
            # what stops jitter around the line from producing ENTRY ENTRY ENTRY.
            return None

        if state.confirmed_side is None:
            # Track first became decidable now (it appeared inside the band).
            # Adopt this side as the baseline rather than treating it as a flip.
            state.confirmed_side = side
            state.first_seen_side = side
            return None

        if side == state.confirmed_side:
            return None

        # Confirmed side flipped: a real crossing, in a known direction.
        direction = (TOP_TO_BOTTOM if state.confirmed_side == SIDE_ABOVE
                     else BOTTOM_TO_TOP)
        state.confirmed_side = side
        state.crossings += 1
        state.status = STATUS_OBSERVED_CROSSING
        return self._make_event(track_id, tracker_id, frame_number, timestamp,
                                direction, point)

    def _make_event(
        self,
        track_id: str,
        tracker_id: int,
        frame_number: int,
        timestamp: float,
        direction: str,
        point: Point,
    ) -> CrossingEvent:
        self._event_counter += 1
        return CrossingEvent(
            event_id="E{:04d}".format(self._event_counter),
            event_type=self.config.event_type_for(direction),
            track_id=track_id,
            tracker_id=tracker_id,
            frame_number=frame_number,
            timestamp=timestamp,
            direction=direction,
            reference_point=point,
            camera=self.config.camera_id,
        )

    # ---------- queries (used by the debug overlay) ----------

    def state_for(self, track_id: str) -> Optional[TrackSideState]:
        return self.states.get(track_id)

    def status_counts(self) -> Dict[str, int]:
        """How many live tracks have been seen to cross versus never crossed."""
        counts = {STATUS_INITIAL_VISIBLE: 0, STATUS_OBSERVED_CROSSING: 0}
        for state in self.states.values():
            counts[state.status] = counts.get(state.status, 0) + 1
        return counts

    def forget(self, track_id: str) -> None:
        """Drop a terminated track's side history.

        Safe because labels are never reused within a run. Keeps memory flat on
        long videos.
        """
        self.states.pop(track_id, None)


# HYSTERESIS: WHY A DEADBAND, AND WHAT IT COSTS
# ---------------------------------------------
# The problem: a detector's box edges jitter by a few pixels every frame. A
# person standing on the line therefore oscillates ABOVE/BELOW/ABOVE/BELOW and a
# naive "side changed" rule emits an event per frame - dozens of phantom
# entries from one stationary person.
#
# This implementation: a band of +/- `deadband` pixels around the line is
# neither side. A crossing is only emitted when the track goes from one
# *confirmed* side to the other, so a person must traverse the entire band to
# be counted, and can never be counted while inside it.
#
# Tradeoffs:
#   + Jitter smaller than the band is completely invisible.
#   + Fast walkers who skip the band entirely in one frame are still counted
#     correctly, because the confirmed side flips.
#   - A genuine shallow crossing - stepping just over the line and back - is
#     not counted. That is usually desirable, but it IS an under-count.
#   - The band is a fixed pixel height, while perspective means a pixel near
#     the top of the frame covers far more floor than one near the bottom. A
#     band tuned for the foreground is effectively much wider in the distance.
#   - Too large a band on a small frame can swallow the crossing region.
#
# Alternatives considered:
#   - Temporal debounce (require N consecutive frames on the new side): also
#     works, but delays every event by N frames and still miscounts a person
#     who genuinely lingers. It is complementary, not better; it could be added
#     on top of this if field data shows a need.
#   - Segment intersection (test the line against the segment between the
#     previous and current reference points): more precise for very fast
#     targets that skip the band, and immune to the band's under-count, but on
#     its own it re-introduces the jitter problem, since a jittering point
#     crosses the line repeatedly. It would be the natural upgrade *combined*
#     with the deadband.
