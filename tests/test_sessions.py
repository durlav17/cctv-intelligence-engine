"""Unit tests for Phase 3 occupancy and visit sessions (src/sessions.py).

No YOLO, no OpenCV, no video.

Run from the project root:
    .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counting import (  # noqa: E402
    BOTTOM_TO_TOP,
    TOP_TO_BOTTOM,
    CameraConfig,
    LineCrossingDetector,
)
from sessions import (  # noqa: E402
    STATUS_ACTIVE,
    STATUS_COMPLETED,
    STATUS_OPEN_AT_END,
    STATUS_OPEN_TRACK_LOST,
    STATUS_UNMATCHED_EXIT,
    OccupancyState,
    SessionManager,
)
from test_counting import FPS, box_at, walk  # noqa: E402


class Phase3Harness:
    """Wires detector -> occupancy -> sessions, the same way the pipeline does.

    Having the wiring in one place keeps the tests about behaviour rather than
    plumbing, and proves the three modules compose without a video.
    """

    def __init__(self, line_y=400, deadband=10, entry_direction=TOP_TO_BOTTOM):
        self.config = CameraConfig(line_y=line_y, deadband=deadband,
                                   entry_direction=entry_direction)
        self.detector = LineCrossingDetector(self.config)
        self.occupancy = OccupancyState()
        self.manager = SessionManager()
        self.events = []

    def feed(self, track_id, ys, start_frame=0):
        for offset, y in enumerate(ys):
            frame = start_frame + offset
            event = self.detector.update(track_id, box_at(y), frame, frame / FPS)
            if event is None:
                continue
            self.events.append(event)
            self.occupancy.apply(event)
            self.manager.apply(event)
        return self.events


class TestOccupancy(unittest.TestCase):
    """TEST 8 (occupancy half) - counts derived from events only."""

    def test_entry_then_exit_returns_to_zero(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500, 300])
        self.assertEqual(h.occupancy.total_entries, 1)
        self.assertEqual(h.occupancy.total_exits, 1)
        self.assertEqual(h.occupancy.current_occupancy, 0)

    def test_occupancy_never_goes_negative(self):
        h = Phase3Harness()
        h.feed("P01", [800, 700, 300])  # starts inside, leaves: exit with no entry
        self.assertEqual(h.occupancy.total_exits, 1)
        self.assertEqual(h.occupancy.current_occupancy, 0)
        self.assertEqual(h.occupancy.unmatched_exits, 1)

    def test_many_unmatched_exits_still_clamp_at_zero(self):
        h = Phase3Harness()
        for i, label in enumerate(["P01", "P02", "P03"]):
            h.feed(label, [800, 300], start_frame=i * 10)
        self.assertEqual(h.occupancy.current_occupancy, 0)
        self.assertEqual(h.occupancy.unmatched_exits, 3)

    def test_concurrent_visitors_accumulate(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        h.feed("P02", [300, 500])
        h.feed("P03", [300, 500])
        self.assertEqual(h.occupancy.current_occupancy, 3)
        h.feed("P02", [500, 300], start_frame=10)
        self.assertEqual(h.occupancy.current_occupancy, 2)
        self.assertEqual(h.occupancy.total_entries, 3)
        self.assertEqual(h.occupancy.total_exits, 1)

    def test_no_duplicate_counting_while_standing_still(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500] + [500] * 50)
        self.assertEqual(h.occupancy.total_entries, 1)


class TestSessionLifecycle(unittest.TestCase):
    """TEST 5, 6 - the normal visit."""

    def test_enter_then_exit_produces_one_completed_session(self):
        h = Phase3Harness()
        h.feed("P01", [300, 350, 500, 550, 600])       # enter at frame 2
        h.feed("P01", [500, 400, 300], start_frame=20)  # leave at frame 22

        self.assertEqual([e.event_type for e in h.events], ["ENTRY", "EXIT"])
        self.assertEqual(len(h.manager.sessions), 1)

        session = h.manager.sessions[0]
        self.assertEqual(session.status, STATUS_COMPLETED)
        self.assertEqual(session.track_id, "P01")
        self.assertEqual(session.entry_frame, 2)
        self.assertEqual(session.exit_frame, 22)
        self.assertEqual(session.entry_direction, TOP_TO_BOTTOM)
        self.assertEqual(session.exit_direction, BOTTOM_TO_TOP)
        self.assertAlmostEqual(session.duration, 20 / FPS)
        self.assertFalse(session.is_open)

    def test_timestamps_come_from_the_video_timeline(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        session = h.manager.sessions[0]
        self.assertAlmostEqual(session.entry_timestamp, 1 / FPS)

    def test_re_entry_opens_a_second_session(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500])                        # in
        h.feed("P01", [500, 300], start_frame=10)        # out
        h.feed("P01", [300, 500], start_frame=20)        # in again
        self.assertEqual(len(h.manager.sessions), 2)
        self.assertEqual(h.manager.sessions[0].status, STATUS_COMPLETED)
        self.assertEqual(h.manager.sessions[1].status, STATUS_ACTIVE)
        self.assertEqual(h.occupancy.current_occupancy, 1)

    def test_cross_and_immediately_reverse_is_a_short_completed_visit(self):
        """Edge case 6: counted honestly, then flagged - never silently dropped."""
        h = Phase3Harness()
        h.feed("P01", [300, 500, 300])
        session = h.manager.sessions[0]
        self.assertEqual(session.status, STATUS_COMPLETED)
        self.assertLess(session.duration, 0.2)
        self.assertEqual(len(h.manager.short_visits(1.0)), 1)
        self.assertEqual(len(h.manager.short_visits(0.01)), 0)

    def test_two_tracks_get_separate_sessions(self):
        """TEST 7 at the session layer."""
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        h.feed("P02", [300, 500])
        h.feed("P01", [500, 300], start_frame=10)
        self.assertEqual(len(h.manager.sessions), 2)
        ids = {s.track_id: s.status for s in h.manager.sessions}
        self.assertEqual(ids["P01"], STATUS_COMPLETED)
        self.assertEqual(ids["P02"], STATUS_ACTIVE)


class TestSessionEdgeCases(unittest.TestCase):
    """TEST 8, 9, 10 - the cases that must not be papered over."""

    # TEST 8
    def test_track_lost_after_entry_does_not_fabricate_an_exit(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500, 550])
        h.manager.on_track_terminated("P01", reason="tracker terminated (occlusion)")

        session = h.manager.sessions[0]
        self.assertEqual(session.status, STATUS_OPEN_TRACK_LOST)
        self.assertIsNone(session.exit_timestamp)
        self.assertIsNone(session.exit_frame)
        self.assertIsNone(session.duration)
        self.assertIn("occlusion", session.closed_reason)
        self.assertTrue(session.is_open)

    def test_track_loss_does_not_reduce_occupancy(self):
        """The core Phase 3 rule: losing a track is not evidence of leaving."""
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        self.assertEqual(h.occupancy.current_occupancy, 1)
        h.manager.on_track_terminated("P01")
        self.assertEqual(h.occupancy.current_occupancy, 1)
        self.assertEqual(h.occupancy.total_exits, 0)

    def test_terminating_a_track_with_no_session_is_harmless(self):
        h = Phase3Harness()
        h.manager.on_track_terminated("P99")
        self.assertEqual(h.manager.sessions, [])

    def test_temporary_loss_then_reappearance_keeps_one_session(self):
        """Edge cases 3 and 4: the registry keeps the label, so nothing happens."""
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        # ... frames where the track is unmatched; no calls reach Phase 3 ...
        h.feed("P01", [520, 540], start_frame=30)
        self.assertEqual(len(h.manager.sessions), 1)
        self.assertEqual(h.manager.sessions[0].status, STATUS_ACTIVE)

    # TEST 9
    def test_video_ending_mid_visit_gives_open_at_end(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500, 600])
        h.manager.close_all_at_end()

        session = h.manager.sessions[0]
        self.assertEqual(session.status, STATUS_OPEN_AT_END)
        self.assertIsNone(session.exit_timestamp)
        self.assertIsNone(session.duration)
        self.assertEqual(len(h.manager.open_sessions()), 1)
        self.assertEqual(len(h.manager.completed_sessions()), 0)

    def test_close_all_at_end_leaves_completed_sessions_alone(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        h.feed("P01", [500, 300], start_frame=10)
        h.feed("P02", [300, 500], start_frame=20)
        h.manager.close_all_at_end()
        statuses = h.manager.status_counts()
        self.assertEqual(statuses[STATUS_COMPLETED], 1)
        self.assertEqual(statuses[STATUS_OPEN_AT_END], 1)

    # TEST 10
    def test_exit_without_entry_is_recorded_explicitly(self):
        h = Phase3Harness()
        h.feed("P01", [800, 700, 300])

        self.assertEqual(len(h.manager.sessions), 1)
        session = h.manager.sessions[0]
        self.assertEqual(session.status, STATUS_UNMATCHED_EXIT)
        self.assertIsNone(session.entry_timestamp)
        self.assertIsNotNone(session.exit_timestamp)
        self.assertIsNone(session.duration, "duration must stay None, not be faked")
        self.assertEqual(h.occupancy.unmatched_exits, 1)

    def test_entry_while_already_inside_closes_the_stale_session(self):
        """A second ENTRY with no exit between means we missed something."""
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        # Force a second entry without an intervening exit by resetting the
        # detector's memory of which side P01 was confirmed on.
        h.detector.forget("P01")
        h.feed("P01", [300, 500], start_frame=50)

        self.assertEqual(len(h.manager.sessions), 2)
        self.assertEqual(h.manager.sessions[0].status, STATUS_OPEN_TRACK_LOST)
        self.assertIn("re-entry", h.manager.sessions[0].closed_reason)
        self.assertEqual(h.manager.sessions[1].status, STATUS_ACTIVE)
        self.assertIsNone(h.manager.sessions[0].exit_timestamp)


class TestSerialisation(unittest.TestCase):
    def test_sessions_are_json_ready(self):
        import json

        h = Phase3Harness()
        h.feed("P01", [300, 500])
        h.feed("P01", [500, 300], start_frame=10)
        h.feed("P02", [300, 500], start_frame=20)
        h.manager.close_all_at_end()

        records = h.manager.to_records()
        json.dumps(records)  # must not raise
        self.assertEqual(len(records), 2)
        for key in ("session_id", "track_id", "entry_timestamp", "exit_timestamp",
                    "duration", "entry_direction", "exit_direction", "status"):
            self.assertIn(key, records[0])

    def test_occupancy_dict(self):
        h = Phase3Harness()
        h.feed("P01", [300, 500])
        self.assertEqual(h.occupancy.to_dict(), {
            "total_entries": 1,
            "total_exits": 0,
            "current_occupancy": 1,
            "unmatched_exits": 0,
        })


if __name__ == "__main__":
    unittest.main()
