"""Unit tests for camera configuration loading and the first-sighting policy.

No YOLO, no OpenCV, no video.

Run from the project root:
    .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counting import (  # noqa: E402
    BOTTOM_TO_TOP,
    TOP_TO_BOTTOM,
    SIDE_ABOVE,
    SIDE_BELOW,
    STATUS_INITIAL_VISIBLE,
    STATUS_OBSERVED_CROSSING,
    CameraConfig,
    LineCrossingDetector,
)
from test_counting import FPS, box_at, walk  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestCameraConfigFromDict(unittest.TestCase):
    def test_minimal_config(self):
        config = CameraConfig.from_dict({"counting_line": 540})
        self.assertEqual(config.line_y, 540)
        self.assertEqual(config.deadband, 10)
        self.assertEqual(config.entry_direction, TOP_TO_BOTTOM)
        self.assertEqual(config.camera_id, "camera_01")

    def test_full_config(self):
        config = CameraConfig.from_dict({
            "camera_id": "north_door",
            "counting_line": 700,
            "entry_direction": BOTTOM_TO_TOP,
            "deadband": 25,
            "short_visit_seconds": 2.5,
        })
        self.assertEqual(config.camera_id, "north_door")
        self.assertEqual(config.line_y, 700)
        self.assertEqual(config.entry_direction, BOTTOM_TO_TOP)
        self.assertEqual(config.exit_direction, TOP_TO_BOTTOM)
        self.assertEqual(config.deadband, 25)
        self.assertEqual(config.short_visit_seconds, 2.5)

    def test_missing_counting_line_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            CameraConfig.from_dict({"camera_id": "x"})
        self.assertIn("counting_line", str(ctx.exception))

    def test_unknown_key_is_rejected_not_ignored(self):
        """A typo must fail loudly rather than silently use the default line."""
        with self.assertRaises(ValueError) as ctx:
            CameraConfig.from_dict({"counting_line": 540, "counting_lines": 600})
        self.assertIn("counting_lines", str(ctx.exception))

    def test_bad_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            CameraConfig.from_dict({"counting_line": 540,
                                    "entry_direction": "LEFT_TO_RIGHT"})

    def test_round_trip(self):
        original = CameraConfig(line_y=400, deadband=15,
                                entry_direction=BOTTOM_TO_TOP,
                                camera_id="cam_x", short_visit_seconds=3.0)
        self.assertEqual(CameraConfig.from_dict(original.to_dict()), original)


class TestCameraConfigFromFile(unittest.TestCase):
    def test_loads_the_committed_camera_01(self):
        path = os.path.join(REPO_ROOT, "config", "cameras", "camera_01.json")
        config = CameraConfig.from_json_file(path)
        self.assertEqual(config.camera_id, "camera_01")
        self.assertEqual(config.line_y, 540)
        self.assertEqual(config.entry_direction, TOP_TO_BOTTOM)

    def test_error_message_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"camera_id": "no_line"}, fh)
            with self.assertRaises(ValueError) as ctx:
                CameraConfig.from_json_file(path)
            self.assertIn("broken.json", str(ctx.exception))

    def test_non_object_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([1, 2, 3], fh)
            with self.assertRaises(ValueError):
                CameraConfig.from_json_file(path)

    def test_camera_id_reaches_the_event(self):
        config = CameraConfig.from_dict({"camera_id": "north_door",
                                         "counting_line": 400})
        detector = LineCrossingDetector(config)
        walk(detector, "P01", [300])
        event = detector.update("P01", box_at(500), 5, 5 / FPS)
        self.assertEqual(event.camera, "north_door")


class TestFirstSightingPolicy(unittest.TestCase):
    """Step 11: a person visible at frame 0 must not be counted as an ENTRY."""

    def setUp(self):
        self.detector = LineCrossingDetector(
            CameraConfig(line_y=400, deadband=10))

    def test_track_starts_initial_visible(self):
        walk(self.detector, "P01", [800])
        state = self.detector.state_for("P01")
        self.assertEqual(state.status, STATUS_INITIAL_VISIBLE)
        self.assertEqual(state.first_seen_side, SIDE_BELOW)

    def test_person_present_at_frame_zero_below_line_is_not_an_entry(self):
        events = walk(self.detector, "P01", [800, 810, 820, 830])
        self.assertEqual(events, [])
        self.assertEqual(self.detector.state_for("P01").status,
                         STATUS_INITIAL_VISIBLE)

    def test_person_present_at_frame_zero_above_line_is_not_an_exit(self):
        events = walk(self.detector, "P01", [100, 110, 120])
        self.assertEqual(events, [])

    def test_status_becomes_observed_crossing_only_after_a_real_crossing(self):
        walk(self.detector, "P01", [300, 350])
        self.assertEqual(self.detector.state_for("P01").status,
                         STATUS_INITIAL_VISIBLE)
        walk(self.detector, "P01", [500], start_frame=10)
        self.assertEqual(self.detector.state_for("P01").status,
                         STATUS_OBSERVED_CROSSING)

    def test_first_seen_side_survives_later_crossings(self):
        walk(self.detector, "P01", [300, 500, 300])
        state = self.detector.state_for("P01")
        self.assertEqual(state.first_seen_side, SIDE_ABOVE)
        self.assertEqual(state.confirmed_side, SIDE_ABOVE)
        self.assertEqual(state.crossings, 2)

    def test_status_counts(self):
        walk(self.detector, "P01", [300, 500])   # crossed
        walk(self.detector, "P02", [800, 850])   # never crossed
        counts = self.detector.status_counts()
        self.assertEqual(counts[STATUS_OBSERVED_CROSSING], 1)
        self.assertEqual(counts[STATUS_INITIAL_VISIBLE], 1)


if __name__ == "__main__":
    unittest.main()
