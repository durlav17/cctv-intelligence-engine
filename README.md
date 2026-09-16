# CCTV Intelligence Engine

Real-time computer vision system for CCTV/video analytics, built milestone by milestone.

**Current status: Phase 2 — Multi-Object Tracking** ✅

## What the pipeline does

Phase 1 — detection only (`src/detect.py`, still runnable and unchanged):

```
video file -> OpenCV (read frames) -> YOLO (detect) -> filter "person" class
-> draw boxes + confidence -> overlay FPS -> display (or save to file)
```

Phase 2 — detection + tracking (`src/track_people.py`):

```
video file -> OpenCV -> YOLO detection -> BoT-SORT tracking
-> persistent track IDs (P01, P02, ...) -> track lifecycle + metadata
-> annotated output -> TRACKING SUMMARY
```

Track IDs are **anonymous and per-run**. P01 in one video has nothing to do
with P01 in another video or on another day — see `docs/experiments.md`.

No database, no API, no dashboard, no face recognition — those come later
(or, for face recognition, not at all by default).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

On your own machine (with a display), this opens a live window:

```bash
python src/detect.py path/to/video.mp4
```

Press `q` to quit the window early.

Options:

```bash
python src/detect.py path/to/video.mp4 --conf 0.5           # confidence threshold
python src/detect.py path/to/video.mp4 --model yolov8s.pt   # bigger/more accurate model
python src/detect.py path/to/video.mp4 --headless --out annotated.mp4   # no window, save to file
```

`--headless` is for servers / CI / sandboxes without a display — it writes the
annotated video to disk instead of opening a window.

The first run downloads `yolov8n.pt` (~6MB) automatically from Ultralytics.

## Project layout

```
cctv-intelligence-engine/
├── src/
│   ├── detect.py       # Phase 1: person detection
│   ├── track_people.py # Phase 2: detection + tracking pipeline (entry point)
│   └── tracking.py     # Phase 2: track lifecycle + metadata (no CV code)
├── data/                # put your test videos here (gitignored)
├── models/              # downloaded YOLO weights land here if you move them
├── docs/
│   ├── architecture.md
│   └── experiments.md  # measured results per phase
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Tracking (Phase 2)

```bash
python src/track_people.py data/people-walking.mp4
```

Options:

```bash
--tracker bytetrack.yaml    # ByteTrack instead of the BoT-SORT default
--conf 0.5                  # detection confidence threshold
--max-lost 45               # frames a track may be unmatched before termination
--headless --out out.mp4    # no window; write annotated video
--records tracks.json       # dump per-track metadata as JSON
--max-frames 200            # stop early (useful while iterating)
```

Each tracked person is drawn with a colour-coded box, an anonymous label and
the detection confidence (`P07 0.55`). The HUD shows active tracked people,
temporarily lost tracks and processing FPS. At the end, a `TRACKING SUMMARY`
lists every track with its frame count, duration, frame range and how often it
was lost and reacquired.

### Track lifecycle

| State | Meaning |
| --- | --- |
| new | a tracker ID seen for the first time; gets the next `Pnn` label |
| active | matched to a detection in the current frame |
| lost | unmatched, but still inside the `--max-lost` grace window |
| terminated | unmatched past the grace window; the ID is retired, never reused |

Losses and reacquisitions are recorded per track rather than hidden — tracking
is not perfect and the summary says so.

## Roadmap

See `docs/architecture.md` for the full 16-phase plan.

- [x] **Phase 1 — Person detection** (`src/detect.py`)
- [x] **Phase 2 — Multi-object tracking** with persistent anonymous IDs
      (`src/track_people.py`, BoT-SORT, ReID off)
- [ ] Phase 3+ — entry/exit counting, zones, dwell time, heatmaps, events,
      database, API, dashboard. Not started; measured results for each
      completed phase go in `docs/experiments.md`.
