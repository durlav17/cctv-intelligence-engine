# Changelog

## [Phase 3] - 2026-09-16

### Added
- `src/counting.py`: spatial reasoning, no OpenCV/YOLO
  - `reference_point()` - bottom-center of the box (feet on the ground plane)
  - `CameraConfig` - counting line, deadband, entry direction, per camera
  - `LineCrossingDetector` - per-track side state machine -> `CrossingEvent`
  - Hysteresis: an ABOVE/BUFFER/BELOW band so jitter around the line cannot
    emit repeated events; one traversal produces exactly one event
- `src/sessions.py`: state that outlives a frame, no OpenCV/YOLO
  - `OccupancyState` - entries, exits, occupancy (clamped at 0), unmatched exits
  - `VisitSession` / `SessionManager` - ACTIVE, COMPLETED, OPEN_AT_END,
    OPEN_TRACK_LOST, UNMATCHED_EXIT, with documented policies for all ten
    edge cases
  - Track loss is explicitly NOT treated as an exit
- `tests/test_counting.py`, `tests/test_sessions.py`: 43 stdlib unittest cases
  covering every spatial and session rule without YOLO, OpenCV or a video
- `docs/experiments.md`: Phase 3 section with measured results

### Changed
- `src/track_people.py`: wires the new modules into the existing frame loop;
  draws the counting line, reference points and crossing banners; HUD gains
  Entries / Exits / Occupancy; prints a COUNTING SUMMARY
- New flags: `--line`, `--deadband`, `--entry-direction`, `--short-visit`,
  `--events`, `--sessions`, `--debug`, `--no-counting`
- `run_tracking()` now returns a dict instead of the bare registry

### Unchanged
- `src/tracking.py` and `src/detect.py` - not modified. Phase 3 is a read-only
  consumer of Track objects, so tracking behaviour is unaffected
  (verified: 88 unique track IDs with and without counting)

### Measured (1920x1080 @ 25fps, 341 frames, line y=540, deadband 10)
- 18 crossing events: 10 entries, 8 exits, final occupancy 4, 2 unmatched exits
- 0 completed sessions - the sample video is a thoroughfare, not a doorway
- Phase 2 baseline 12.37 fps vs Phase 3 12.23 fps: ~1% difference, within
  run-to-run noise
- No ground truth exists for this footage, so no accuracy metric is claimed
- Detail: `docs/experiments.md`

### Added later in Phase 3
- `config/cameras/camera_01.json` + `config/cameras/README.md`: per-camera
  geometry as data, loaded with `--camera-config`. Unknown keys are rejected,
  CLI flags override the file.
- `CameraConfig.from_dict()` / `.from_json_file()` / `.to_dict()`
- `CameraConfig.name` renamed to `camera_id`; carried into every event and
  session record (JSON key `camera_id` in the session header)
- Explicit first-sighting policy: `INITIAL_VISIBLE` -> `OBSERVED_CROSSING`
  on `TrackSideState`, with `first_seen_side` and `status_counts()`
- `--video` as an alternative to the positional video argument
- `tests/test_camera_config.py`: 16 more tests (59 total)

### Still not implemented
- Zones, dwell time, ReID, cross-day identity, visitor categories,
  PostgreSQL, FastAPI, dashboard, face recognition, biometrics

## [Phase 2] - 2026-09-16

### Added
- `src/track_people.py`: detection + multi-object tracking pipeline
  - Uses `model.track(persist=True, tracker="botsort.yaml", classes=[0])`
    so YOLO detections are associated across frames into persistent track IDs
  - Anonymous labels (P01, P02, ...) drawn with per-track colour + confidence
  - HUD: active tracked people, temporarily lost tracks, processing FPS
  - `--tracker` switches between BoT-SORT and ByteTrack
  - `--records` dumps per-track metadata as JSON
  - End-of-run `TRACKING SUMMARY`
- `src/tracking.py`: tracker-agnostic lifecycle bookkeeping
  - `Track` dataclass: track_id, class_name, confidence, first/last_seen_frame,
    first/last_seen_timestamp, frames_tracked, bbox, state, event history
  - `TrackRegistry`: new / active / lost / terminated state machine with a
    configurable grace window; records losses instead of hiding them
  - Records are shaped for a future session/track/visit/event schema
    (no database in this phase)
- `docs/experiments.md`: measured Phase 2 results

### Unchanged
- `src/detect.py` (Phase 1) — not modified, still runs standalone

### Measured (CPU-only, YOLOv8n, 1920x1080 @ 25fps, 341 frames)
- BoT-SORT: 12.78 avg processing FPS, 88 unique track IDs,
  95.9% consecutive-frame ID continuation
- ByteTrack: 16.67 avg processing FPS, 89 unique track IDs
- Single-person crop: one ID held 306/341 frames across 11 dropouts
- Full detail and known failure modes: `docs/experiments.md`

### Still not implemented
- PostgreSQL, FastAPI, dashboard, face recognition, cross-day identity,
  category classification, ReID appearance matching

## [Phase 1] - 2026-09-16

### Added
- `src/detect.py`: video -> OpenCV -> YOLOv8n -> person detection pipeline
  - Filters detections to COCO class 0 ("person") above a confidence threshold
  - Draws bounding boxes + confidence labels
  - Overlays live FPS and per-frame person count
  - `--headless` mode: writes annotated output video instead of opening a
    window (for servers/CI without a display)
- `requirements.txt`: opencv-python, ultralytics, numpy
- Project skeleton: `src/`, `data/`, `models/`, `docs/`

### Tested
- Ran against a real CCTV-style test clip (person walking indoors,
  768x432 @ 12fps). Verified visually: correct bounding box placement,
  confidence score displayed, FPS overlay updating, person count correct.

### Not yet implemented (future phases)
- Tracking / persistent IDs (Phase 2)
- Entry/exit counting, zones, dwell-time, heatmaps (Phases 3-7)
- Event detection, database, API, dashboard (Phases 8-11)
- Historical analytics, forecasting, benchmarking, Docker (Phases 12-15)
