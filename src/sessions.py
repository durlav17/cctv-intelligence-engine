"""
CCTV Intelligence Engine - Phase 3: occupancy and visit sessions.

Consumes CrossingEvent objects from counting.py and maintains two pieces of
state that outlive any single frame:

  OccupancyState  - entries, exits, current occupancy
  SessionManager  - VisitSession per (track, visit), with an explicit status

No OpenCV, no YOLO, no file I/O. Unit-testable with hand-built events.

IDENTITY MODEL - read before extending this file
------------------------------------------------
`track_id` here is an anonymous, per-run tracking label such as "P01". It means
"the tracker believed these boxes were one moving object during this run". It
does NOT identify a person, and it is not comparable across videos, cameras or
days. A future anonymous-ReID layer might group sessions under an anonymous
visitor entity (V001 -> S001, S002, ...) - that layer does not exist, and
nothing in this file should be read as if it does. No names, no biometrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from counting import CrossingEvent, EVENT_ENTRY, EVENT_EXIT

# Session lifecycle statuses.
STATUS_ACTIVE = "ACTIVE"                  # entered, not yet seen to leave
STATUS_COMPLETED = "COMPLETED"            # entered and exited across the line
STATUS_OPEN_AT_END = "OPEN_AT_END"        # video ended mid-visit
STATUS_OPEN_TRACK_LOST = "OPEN_TRACK_LOST"  # tracker gave up; no exit observed
STATUS_UNMATCHED_EXIT = "UNMATCHED_EXIT"  # exit with no recorded entry


@dataclass
class VisitSession:
    """One visit: from an observed ENTRY crossing to an observed EXIT crossing.

    Timestamps are seconds on the *video* timeline, never wall-clock, so a
    session means the same thing whether the file is processed live, at 10x, or
    a year later.

    exit_timestamp and duration stay None unless an exit was actually observed.
    We never fabricate one to make the record look tidy.
    """

    session_id: str
    track_id: str
    status: str
    entry_timestamp: Optional[float] = None
    exit_timestamp: Optional[float] = None
    entry_frame: Optional[int] = None
    exit_frame: Optional[int] = None
    entry_direction: Optional[str] = None
    exit_direction: Optional[str] = None
    closed_reason: str = ""
    camera: str = "camera-1"

    @property
    def duration(self) -> Optional[float]:
        """Seconds between entry and exit, or None if the visit never closed."""
        if self.entry_timestamp is None or self.exit_timestamp is None:
            return None
        return self.exit_timestamp - self.entry_timestamp

    @property
    def is_open(self) -> bool:
        return self.status in (STATUS_ACTIVE, STATUS_OPEN_AT_END,
                               STATUS_OPEN_TRACK_LOST)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = None if self.duration is None else round(self.duration, 3)
        for key in ("entry_timestamp", "exit_timestamp"):
            if d[key] is not None:
                d[key] = round(d[key], 3)
        return d


class OccupancyState:
    """Running entry/exit tallies derived from crossing events.

    WHY EVENT-DERIVED OCCUPANCY IS NOT "PEOPLE VISIBLE IN THE FRAME"
    ----------------------------------------------------------------
    Counting boxes in the current frame answers "how many people can the
    detector see right now" - which is not the same question. It:
      - misses everyone inside the monitored space but out of camera view
        (behind a shelf, in a side room, occluded by a pillar),
      - counts people who are visible but on the *other* side of the line and
        so were never admitted,
      - drops to zero the moment the detector has a bad frame, then jumps back,
      - cannot distinguish someone walking past the doorway from someone who
        went through it.
    Event-derived occupancy instead accumulates observed transitions, so it
    survives occlusion and people leaving the field of view. Its own failure
    mode is different and worth stating: errors *accumulate*. One missed exit
    inflates occupancy for the rest of the run, whereas a per-frame count is
    wrong only for that frame. Neither is ground truth.
    """

    def __init__(self) -> None:
        self.total_entries = 0
        self.total_exits = 0
        # Exits we could not subtract because occupancy was already 0. Kept
        # instead of allowing a negative count, so the discrepancy is visible.
        self.unmatched_exits = 0
        self._occupancy = 0

    @property
    def current_occupancy(self) -> int:
        return self._occupancy

    def apply(self, event: CrossingEvent) -> None:
        """Fold one crossing event into the tallies."""
        if event.event_type == EVENT_ENTRY:
            self.total_entries += 1
            self._occupancy += 1
        elif event.event_type == EVENT_EXIT:
            self.total_exits += 1
            if self._occupancy > 0:
                self._occupancy -= 1
            else:
                # Someone left who we never saw arrive - they were already
                # inside when the video started, or we missed their entry.
                # Clamping at zero keeps occupancy physically meaningful;
                # the counter records that we know it happened.
                self.unmatched_exits += 1
        else:
            raise ValueError("unknown event_type {!r}".format(event.event_type))

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "current_occupancy": self.current_occupancy,
            "unmatched_exits": self.unmatched_exits,
        }


class SessionManager:
    """Creates and closes VisitSessions in response to crossing events.

    TRACK LOSS IS NOT AN EXIT
    -------------------------
    The single most important rule in this file. A tracker terminates a track
    for many reasons - occlusion, a detector miss, the person walking out of
    frame, the video ending - and none of them are evidence that the person
    crossed the exit boundary. An EXIT is only ever recorded when a crossing
    event says so. When a track dies mid-visit we close the *record* with an
    honest status (OPEN_TRACK_LOST) and no exit timestamp, and we deliberately
    do NOT decrement occupancy: as far as we observed, that person is still
    inside.
    """

    def __init__(self, camera: str = "camera-1") -> None:
        self.camera = camera
        self.sessions: List[VisitSession] = []
        self._open_by_track: Dict[str, VisitSession] = {}
        self._session_counter = 0

    def _new_session_id(self) -> str:
        self._session_counter += 1
        return "S{:03d}".format(self._session_counter)

    # ---------- event handling ----------

    def apply(self, event: CrossingEvent) -> VisitSession:
        """Fold one crossing event into session state; returns the touched session."""
        if event.event_type == EVENT_ENTRY:
            return self._handle_entry(event)
        if event.event_type == EVENT_EXIT:
            return self._handle_exit(event)
        raise ValueError("unknown event_type {!r}".format(event.event_type))

    def _handle_entry(self, event: CrossingEvent) -> VisitSession:
        existing = self._open_by_track.get(event.track_id)
        if existing is not None:
            # Entry while already inside: we missed the exit in between. Close
            # the stale record rather than overwriting it silently, then open a
            # fresh one so the new visit is still measured.
            existing.status = STATUS_OPEN_TRACK_LOST
            existing.closed_reason = "re-entry observed without an intervening exit"

        session = VisitSession(
            session_id=self._new_session_id(),
            track_id=event.track_id,
            status=STATUS_ACTIVE,
            entry_timestamp=event.timestamp,
            entry_frame=event.frame_number,
            entry_direction=event.direction,
            camera=self.camera,
        )
        self.sessions.append(session)
        self._open_by_track[event.track_id] = session
        return session

    def _handle_exit(self, event: CrossingEvent) -> VisitSession:
        session = self._open_by_track.pop(event.track_id, None)
        if session is None:
            # EXIT with no ENTRY. Real and common: the person was already
            # inside when the video started, or their entry happened during an
            # occlusion. Recorded as its own status - never dropped, and never
            # back-dated into a fake entry.
            session = VisitSession(
                session_id=self._new_session_id(),
                track_id=event.track_id,
                status=STATUS_UNMATCHED_EXIT,
                exit_timestamp=event.timestamp,
                exit_frame=event.frame_number,
                exit_direction=event.direction,
                closed_reason="exit observed with no recorded entry",
                camera=self.camera,
            )
            self.sessions.append(session)
            return session

        session.status = STATUS_COMPLETED
        session.exit_timestamp = event.timestamp
        session.exit_frame = event.frame_number
        session.exit_direction = event.direction
        return session

    # ---------- lifecycle hooks ----------

    def on_track_terminated(self, track_id: str, reason: str = "track terminated") -> None:
        """The tracker gave up on this track. NOT an exit - see class docstring."""
        session = self._open_by_track.pop(track_id, None)
        if session is None:
            return
        session.status = STATUS_OPEN_TRACK_LOST
        session.closed_reason = reason

    def close_all_at_end(self, reason: str = "video ended during active visit") -> None:
        """End of video: mark still-open visits OPEN_AT_END, with no exit time."""
        for track_id in list(self._open_by_track):
            session = self._open_by_track.pop(track_id)
            session.status = STATUS_OPEN_AT_END
            session.closed_reason = reason

    # ---------- queries / reporting ----------

    def open_sessions(self) -> List[VisitSession]:
        return [s for s in self.sessions if s.is_open]

    def completed_sessions(self) -> List[VisitSession]:
        return [s for s in self.sessions if s.status == STATUS_COMPLETED]

    def short_visits(self, threshold_seconds: float) -> List[VisitSession]:
        """Completed visits shorter than a threshold - possible line-hover artefacts.

        Reported, never suppressed: a 0.3 s visit may be a counting error or may
        be someone who genuinely stepped in and turned around. We cannot tell
        from geometry alone, so we surface it and let a human decide.
        """
        return [s for s in self.completed_sessions()
                if s.duration is not None and s.duration < threshold_seconds]

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.sessions:
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts

    def to_records(self) -> List[dict]:
        return [s.to_dict() for s in self.sessions]


# SESSION EDGE-CASE POLICIES (all explicit, none silent)
# ------------------------------------------------------
#  1. EXIT without ENTRY        -> STATUS_UNMATCHED_EXIT session; occupancy
#                                  clamps at 0 and OccupancyState.unmatched_exits
#                                  increments. No entry time invented.
#  2. ENTRY then track death    -> STATUS_OPEN_TRACK_LOST. No exit timestamp,
#                                  occupancy NOT decremented.
#  3. Temporary track loss      -> nothing happens. The registry keeps the track
#                                  alive inside its grace window, the label is
#                                  unchanged, the session stays ACTIVE.
#  4. Track reappears in window -> same label, same session continues.
#  5. Hovering near the line    -> the deadband in counting.py means no event is
#                                  generated at all, so no session churn.
#  6. Cross and reverse         -> two real events: ENTRY then EXIT, producing a
#                                  COMPLETED session with a very short duration.
#                                  Surfaced via short_visits(), not suppressed.
#  7. Simultaneous crossings    -> state is per track_id throughout; events are
#                                  independent and order-insensitive.
#  8. Duplicate detections      -> impossible to double-count: the registry
#                                  dedupes by tracker id, and the crossing
#                                  detector only fires on a confirmed side flip.
#  9. Video ends mid-visit      -> STATUS_OPEN_AT_END via close_all_at_end().
# 10. Re-entry without exit     -> stale session marked OPEN_TRACK_LOST with a
#                                  reason, new session opened.
