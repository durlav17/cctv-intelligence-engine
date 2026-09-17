"""Unit tests for Phase 3 spatial logic (src/counting.py).

No YOLO, no OpenCV, no video - just boxes and integers.

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
    EVENT_ENTRY,
    EVENT_EXIT,
    SIDE_ABOVE,
    SIDE_BELOW,
    SIDE_BUFFER,
    CameraConfig,
    LineCrossingDetector,
    reference_point,
)

FPS = 25.0


def box_at(y: int, x: int = 100, width: int = 40, height: int = 100):
    """A box whose bottom edge - the reference point - sits exactly at y."""
    return (x - width // 2, y - height, x + width // 2, y)


def walk(detector, track_id, ys, start_frame=0, tracker_id=1):
    """Feed a track through a sequence of reference-point rows.

    Returns the list of CrossingEvents produced along the way.
    """
    events = []
    for offset, y in enumerate(ys):
        frame = start_frame + offset
        event = detector.update(track_id, box_at(y), frame, frame / FPS, tracker_id)
        if event is not None:
            events.append(event)
    return events


class TestReferencePoint(unittest.TestCase):
    def test_bottom_center(self):
        self.assertEqual(reference_point((10, 20, 30, 80)), (20, 80))

    def test_is_bottom_edge_not_center(self):
        point = reference_point((0, 0, 100, 400))
        self.assertEqual(point[1], 400, "y must be the bottom edge, not the mid-height")


class TestCameraConfig(unittest.TestCase):
    def setUp(self):
        self.config = CameraConfig(line_y=400, deadband=10)

    def test_side_classification(self):
        self.assertEqual(self.config.side_of((0, 200)), SIDE_ABOVE)
        self.assertEqual(self.config.side_of((0, 389)), SIDE_ABOVE)
        self.assertEqual(self.config.side_of((0, 390)), SIDE_BUFFER)
        self.assertEqual(self.config.side_of((0, 400)), SIDE_BUFFER)
        self.assertEqual(self.config.side_of((0, 410)), SIDE_BUFFER)
        self.assertEqual(self.config.side_of((0, 411)), SIDE_BELOW)
        self.assertEqual(self.config.side_of((0, 900)), SIDE_BELOW)

    def test_zero_deadband_leaves_no_buffer(self):
        strict = CameraConfig(line_y=400, deadband=0)
        self.assertEqual(strict.side_of((0, 399)), SIDE_ABOVE)
        self.assertEqual(strict.side_of((0, 400)), SIDE_BUFFER)
        self.assertEqual(strict.side_of((0, 401)), SIDE_BELOW)

    def test_entry_direction_is_configurable(self):
        downward = CameraConfig(line_y=400, entry_direction=TOP_TO_BOTTOM)
        self.assertEqual(downward.event_type_for(TOP_TO_BOTTOM), EVENT_ENTRY)
        self.assertEqual(downward.event_type_for(BOTTOM_TO_TOP), EVENT_EXIT)

        upward = CameraConfig(line_y=400, entry_direction=BOTTOM_TO_TOP)
        self.assertEqual(upward.event_type_for(BOTTOM_TO_TOP), EVENT_ENTRY)
        self.assertEqual(upward.event_type_for(TOP_TO_BOTTOM), EVENT_EXIT)
        self.assertEqual(upward.exit_direction, TOP_TO_BOTTOM)

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            CameraConfig(line_y=400, deadband=-1)
        with self.assertRaises(ValueError):
            CameraConfig(line_y=400, entry_direction="SIDEWAYS")
        with self.assertRaises(ValueError):
            CameraConfig(line_y=-5)


class TestCrossingDetection(unittest.TestCase):
    """TEST 1, 2, 3, 4, 6, 7 from the Phase 3 spec."""

    def setUp(self):
        self.config = CameraConfig(line_y=400, deadband=10,
                                   entry_direction=TOP_TO_BOTTOM)
        self.detector = LineCrossingDetector(self.config)

    # TEST 1
    def test_person_stays_above_line_produces_no_event(self):
        events = walk(self.detector, "P01", [100, 150, 200, 250, 300, 340, 370])
        self.assertEqual(events, [])

    def test_person_stays_below_line_produces_no_event(self):
        events = walk(self.detector, "P01", [800, 750, 700, 650, 600, 500, 450])
        self.assertEqual(events, [])

    # TEST 2
    def test_above_to_below_is_entry_when_configured(self):
        events = walk(self.detector, "P01", [300, 350, 395, 405, 450, 500])
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, EVENT_ENTRY)
        self.assertEqual(event.direction, TOP_TO_BOTTOM)
        self.assertEqual(event.track_id, "P01")
        self.assertEqual(event.frame_number, 4)  # first frame confirmed BELOW
        self.assertAlmostEqual(event.timestamp, 4 / FPS)
        self.assertEqual(event.reference_point[1], 450)

    # TEST 3
    def test_below_to_above_is_exit_when_configured(self):
        events = walk(self.detector, "P01", [600, 500, 420, 400, 380, 300])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_EXIT)
        self.assertEqual(events[0].direction, BOTTOM_TO_TOP)

    def test_direction_meaning_flips_with_config(self):
        inverted = LineCrossingDetector(
            CameraConfig(line_y=400, deadband=10, entry_direction=BOTTOM_TO_TOP))
        events = walk(inverted, "P01", [300, 350, 450, 500])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_EXIT)
        self.assertEqual(events[0].direction, TOP_TO_BOTTOM)

    # TEST 4
    def test_jitter_inside_deadband_produces_no_events(self):
        jitter = [398, 402, 396, 404, 399, 401, 395, 405, 400, 397, 403]
        events = walk(self.detector, "P01", jitter)
        self.assertEqual(events, [], "deadband must absorb oscillation around the line")

    def test_jitter_after_a_real_crossing_does_not_re_fire(self):
        # Cross properly, then hover on the far edge of the band.
        ys = [300, 350, 450, 500, 430, 405, 395, 402, 408, 412, 460]
        events = walk(self.detector, "P01", ys)
        self.assertEqual(len(events), 1, "one traversal must yield exactly one event")
        self.assertEqual(events[0].event_type, EVENT_ENTRY)

    def test_walking_far_below_does_not_repeat_the_event(self):
        events = walk(self.detector, "P01", [300, 450, 600, 700, 800, 900, 1000])
        self.assertEqual(len(events), 1)

    # TEST 6
    def test_crossing_back_produces_the_opposite_event(self):
        events = walk(self.detector, "P01", [300, 450, 500, 450, 300, 250])
        self.assertEqual([e.event_type for e in events], [EVENT_ENTRY, EVENT_EXIT])
        self.assertEqual([e.direction for e in events],
                         [TOP_TO_BOTTOM, BOTTOM_TO_TOP])

    def test_multiple_traversals_all_counted(self):
        events = walk(self.detector, "P01", [300, 500, 300, 500, 300])
        self.assertEqual([e.event_type for e in events],
                         [EVENT_ENTRY, EVENT_EXIT, EVENT_ENTRY, EVENT_EXIT])

    # TEST 7
    def test_two_tracks_are_independent(self):
        walk(self.detector, "P01", [300, 350, 450])          # P01 enters
        walk(self.detector, "P02", [600, 550, 350])          # P02 exits
        p01 = walk(self.detector, "P01", [500, 550], start_frame=10)
        p02 = walk(self.detector, "P02", [300, 250], start_frame=10)
        self.assertEqual(p01, [], "P01 already confirmed BELOW; no new event")
        self.assertEqual(p02, [], "P02 already confirmed ABOVE; no new event")
        self.assertEqual(self.detector.state_for("P01").confirmed_side, SIDE_BELOW)
        self.assertEqual(self.detector.state_for("P02").confirmed_side, SIDE_ABOVE)

    def test_simultaneous_crossings_get_distinct_event_ids(self):
        a = self.detector.update("P01", box_at(300), 0, 0.0)
        b = self.detector.update("P02", box_at(300), 0, 0.0)
        self.assertIsNone(a)
        self.assertIsNone(b)
        a = self.detector.update("P01", box_at(500), 1, 0.04)
        b = self.detector.update("P02", box_at(500), 1, 0.04)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a.event_id, b.event_id)
        self.assertEqual(a.frame_number, b.frame_number)


class TestFirstSighting(unittest.TestCase):
    def setUp(self):
        self.detector = LineCrossingDetector(CameraConfig(line_y=400, deadband=10))

    def test_appearing_below_the_line_is_not_a_crossing(self):
        """A person who walks into frame already inside must not be counted."""
        events = walk(self.detector, "P01", [800, 820, 850])
        self.assertEqual(events, [])

    def test_appearing_inside_the_band_adopts_first_decidable_side(self):
        events = walk(self.detector, "P01", [400, 398, 402, 460, 500])
        self.assertEqual(events, [], "no confirmed side existed before, so no flip")
        self.assertEqual(self.detector.state_for("P01").confirmed_side, SIDE_BELOW)

    def test_appearing_in_band_then_leaving_upward_then_crossing_down(self):
        events = walk(self.detector, "P01", [400, 300, 500])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, TOP_TO_BOTTOM)

    def test_forget_drops_state(self):
        walk(self.detector, "P01", [300])
        self.assertIsNotNone(self.detector.state_for("P01"))
        self.detector.forget("P01")
        self.assertIsNone(self.detector.state_for("P01"))


class TestEventSerialisation(unittest.TestCase):
    def test_event_dict_is_json_ready(self):
        import json

        detector = LineCrossingDetector(CameraConfig(line_y=400, deadband=10))
        walk(detector, "P01", [300])
        event = detector.update("P01", box_at(500), 12, 12 / FPS, tracker_id=7)
        payload = event.to_dict()
        json.dumps(payload)  # must not raise
        for key in ("event_id", "event_type", "track_id", "frame_number",
                    "timestamp", "direction", "reference_point"):
            self.assertIn(key, payload)
        self.assertEqual(payload["tracker_id"], 7)
        self.assertEqual(payload["reference_point"], [100, 500])


if __name__ == "__main__":
    unittest.main()
