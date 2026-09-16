# Architecture

## Current pipeline (Phase 1)

```
┌──────────┐    ┌────────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────┐
│ Video    │ -> │ OpenCV     │ -> │ YOLOv8n  │ -> │ Filter class │ -> │ Draw +  │
│ file/RTSP│    │ VideoCapture│   │ inference│    │ == "person"  │    │ display │
└──────────┘    └────────────┘    └──────────┘    └──────────────┘    └─────────┘
```

- **OpenCV (`cv2.VideoCapture`)** decodes the video and hands us one frame
  (a NumPy array, shape `HxWx3`, BGR) at a time.
- **YOLOv8n** (Ultralytics) is a single-stage object detector: one forward
  pass through a CNN produces boxes + class IDs + confidence scores for
  every object it recognizes in the frame. "n" = nano, the smallest/fastest
  variant, trained on COCO (80 classes). We only keep class `0` ("person").
- **FPS** is measured as wall-clock time between the start of processing
  one frame and the start of the next — this reflects the full pipeline
  cost (decode + inference + draw), not just model inference time.

## Design decisions

- **Anonymous IDs by default**: no facial recognition anywhere in the
  architecture. Later phases (tracking) will assign IDs like `P17` based
  on appearance/motion continuity (ByteTrack/BoT-SORT), not identity.
- **`--headless` flag**: `cv2.imshow` requires a display/X server. Servers,
  CI runners, and sandboxes don't have one, so the script can write the
  annotated video to a file instead — same detection code path either way.
- **No premature abstraction**: single script, single function for the
  core loop. We'll split into modules (`detector.py`, `tracker.py`, etc.)
  once there's an actual second consumer of that logic (Phase 2).

## Planned phase-by-phase evolution

| Phase | Adds |
|---|---|
| 1 ✅ | Detection |
| 2 | Multi-object tracking (ByteTrack/BoT-SORT) + anonymous persistent IDs |
| 3 | Entry/exit line-crossing counting |
| 4 | Occupancy estimation (net count in a zone) |
| 5 | Zone definitions (polygons) |
| 6 | Dwell-time per person per zone |
| 7 | Heatmaps / movement analytics |
| 8 | Event detection (crowding, restricted zone, stationary person) |
| 9 | PostgreSQL persistence |
| 10 | FastAPI backend |
| 11 | React/Next.js dashboard |
| 12 | Historical analytics queries |
| 13 | Occupancy forecasting |
| 14 | Benchmarking (FPS, latency, precision/recall, ID switches) |
| 15 | Docker / deployment |
| 16 | Docs + portfolio polish |

Each phase is additive — earlier phases keep working as later ones are
layered on top.
