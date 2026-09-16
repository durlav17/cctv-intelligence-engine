# Changelog

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
