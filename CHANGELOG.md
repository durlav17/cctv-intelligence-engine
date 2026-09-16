# Changelog

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
