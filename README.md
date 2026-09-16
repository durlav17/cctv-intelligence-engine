# CCTV Intelligence Engine

Real-time computer vision system for CCTV/video analytics, built milestone by milestone.

**Current status: Phase 1 — Person Detection** ✅

## What Phase 1 does

```
video file -> OpenCV (read frames) -> YOLO (detect) -> filter "person" class
-> draw boxes + confidence -> overlay FPS -> display (or save to file)
```

No tracking, no database, no API yet — those come in later phases.

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
│   └── detect.py       # Phase 1: person detection
├── data/                # put your test videos here (gitignored)
├── models/              # downloaded YOLO weights land here if you move them
├── docs/
│   └── architecture.md
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Roadmap

See `docs/architecture.md` for the full 16-phase plan. Phase 2 (multi-object
tracking with persistent anonymous IDs) is next, on request.
