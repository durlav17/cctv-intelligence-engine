# CCTV Intelligence Engine

Real-time computer vision system for CCTV/video analytics, built milestone by milestone.

**Current status: Phase 3 — Entry/Exit Counting, Occupancy & Visit Sessions** ✅

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

Phase 3 — counting on top of tracking (same entry point):

```
... tracking -> bottom-center reference point -> side of counting line
-> ENTRY / EXIT crossing events -> occupancy -> visit sessions
-> annotated output -> COUNTING SUMMARY
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
│   ├── track_people.py # Phase 2+3: pipeline + drawing + CLI (entry point)
│   ├── tracking.py     # Phase 2: track lifecycle + metadata (no CV code)
│   ├── counting.py     # Phase 3: reference point, line crossing (no CV code)
│   └── sessions.py     # Phase 3: occupancy + visit sessions (no CV code)
├── data/                # put your test videos here (gitignored)
├── models/              # downloaded YOLO weights land here if you move them
├── config/
│   └── cameras/        # one JSON per camera: line, direction, deadband
├── tests/              # stdlib unittest; no YOLO/OpenCV needed
│   ├── test_counting.py
│   ├── test_sessions.py
│   └── test_camera_config.py
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

## Counting, occupancy and visits (Phase 3)

Counting is **on by default**, using a horizontal line across the middle of the
frame. Everything about it is configurable, and `--no-counting` gives you plain
Phase 2 behaviour.

```bash
python src/track_people.py data/people-walking.mp4 --line 700 --deadband 12
```

| Flag | Meaning |
| --- | --- |
| `--video FILE` | input video (alternative to the positional form) |
| `--camera-config FILE` | camera geometry JSON, e.g. `config/cameras/camera_01.json` |
| `--line Y` | counting line row in pixels (overrides the camera config) |
| `--deadband N` | half-height of the jitter buffer around the line (default: 10) |
| `--entry-direction` | `TOP_TO_BOTTOM` (default) or `BOTTOM_TO_TOP` |
| `--short-visit S` | completed visits below S seconds get flagged in the summary |
| `--events FILE` | write crossing events as JSON |
| `--sessions FILE` | write visit sessions + occupancy as JSON |
| `--debug` | draw each track's current/previous side next to its reference point |
| `--no-counting` | disable Phase 3 entirely |

### How a crossing is decided

1. **Reference point** — each person is reduced to the **bottom-center** of
   their box (`x=(x1+x2)/2`, `y=y2`). A counting line on a CCTV image is really
   a line on the floor, and the bottom edge is where the feet are. A box center
   floats at chest height and crosses a floor line too early, by an amount that
   grows with how close the person is to the camera.
2. **Side** — the point is classified `ABOVE`, `BELOW`, or `BUFFER`.
3. **Deadband** — with `--line 400 --deadband 10`: `y < 390` is ABOVE,
   `390..410` is BUFFER, `y > 410` is BELOW. The buffer is deliberately
   undecided, so box jitter around the line cannot produce
   `ENTRY ENTRY ENTRY`. A person must traverse the whole band to be counted,
   and can never be counted while standing inside it.
4. **Crossing** — fires only when a track's last *confirmed* side flips.
   One traversal produces exactly one event.
5. **Direction → meaning** — `--entry-direction` decides which way is ENTRY;
   the other way is EXIT. Nothing is hard-coded, because a camera mounted
   facing the other way inverts it.

A person first seen already below the line is **not** counted — we did not
observe them cross.

### Camera configuration

Counting geometry belongs to the *camera*, not to the code or the filename.
One JSON file per camera, in `config/cameras/`:

```json
{
  "camera_id": "camera_01",
  "counting_line": 540,
  "entry_direction": "TOP_TO_BOTTOM",
  "deadband": 10,
  "short_visit_seconds": 1.0
}
```

```bash
python src/track_people.py --video data/people-walking.mp4 \
    --camera-config config/cameras/camera_01.json
```

`camera_id` is carried into every event and session record, so rows from
different cameras stay distinguishable once they share a store. CLI flags
override the file, so you can sweep a line position without editing it.
Unknown keys are rejected rather than ignored — a typo like `counting_lines`
would otherwise leave the line at its default and quietly produce wrong counts.

Nothing infers a line for an unknown scene, by design. See
`config/cameras/README.md` for how to choose one for a new camera.

### First-sighting policy

A person can be visible in frame 0, or walk in from the side, without ever
having been seen to cross the boundary. Counting that as an ENTRY would inflate
every total.

So each track starts as **`INITIAL_VISIBLE`**. Its first decidable side is
recorded as a *baseline*, never as an event. Only when the track is observed
moving from one confirmed side to the other does it become
`OBSERVED_CROSSING` and begin producing events. A track still `INITIAL_VISIBLE`
at the end of the video was simply never seen to cross — a fact worth
reporting, not a missing entry to guess at.

### Occupancy

```
occupancy = entries - exits      (clamped at 0)
```

An exit with nobody inside increments `unmatched_exits` instead of going
negative, so the discrepancy stays visible.

**This is not the same as counting people in the frame.** A per-frame headcount
misses everyone inside but out of view, counts people who never came in, and
drops to zero on one bad detection frame. Event-derived occupancy survives
occlusion and people leaving view — but its errors *accumulate*: one missed
exit inflates the count for the rest of the run. Neither is ground truth.

### Visit sessions

An ENTRY opens a session; a matching EXIT closes it. Timestamps come from the
video timeline (`frame / fps`), never wall-clock.

| Status | When |
| --- | --- |
| `ACTIVE` | entered, still inside |
| `COMPLETED` | entered and exited; has a duration |
| `OPEN_AT_END` | video ended mid-visit |
| `OPEN_TRACK_LOST` | tracker gave up; **no exit was observed** |
| `UNMATCHED_EXIT` | exit with no recorded entry (already inside at start) |

**Losing a track is not an exit.** A tracker terminates on occlusion, detector
failure, someone walking out of frame, or the file ending — none of which is
evidence that the person crossed the exit boundary. When that happens the
session becomes `OPEN_TRACK_LOST`, gets **no** exit timestamp, and occupancy is
**not** decremented. An EXIT is recorded only when a crossing says so.

### Identity model

`P01` is an anonymous, per-run tracking label. It is not a person, and it means
nothing across videos, cameras or days. No face recognition, no biometrics, no
names — by design, not by omission.

### JSON schema

`--events` writes `{"session": {...}, "events": [...]}`, where each event is:

```json
{
  "event_id": "E0002", "event_type": "ENTRY", "track_id": "P05",
  "tracker_id": 12, "frame_number": 187, "timestamp": 7.48,
  "direction": "TOP_TO_BOTTOM", "reference_point": [460, 354],
  "camera": "camera-1"
}
```

`--sessions` writes `{"session": {...}, "occupancy": {...}, "visits": [...]}`,
where each visit is:

```json
{
  "session_id": "S001", "track_id": "P07", "status": "UNMATCHED_EXIT",
  "entry_timestamp": null, "exit_timestamp": 0.84,
  "entry_frame": null, "exit_frame": 21,
  "entry_direction": null, "exit_direction": "BOTTOM_TO_TOP",
  "duration": null, "closed_reason": "exit observed with no recorded entry",
  "camera": "camera-1"
}
```

The shared `session` block (video, model, tracker, resolution, line, deadband,
directions) is the run-level record every other row would key off in a future
database. Nulls are meaningful: they mean "not observed", never "zero".

### Tests

```bash
python -m unittest discover -s tests -v
```

59 tests, no YOLO/OpenCV/video required — all spatial, session and config
logic is pure functions over tuples and dicts.

### Known limitations

- Horizontal lines only; no angled or polygonal boundaries yet.
- The deadband is a fixed pixel height, so perspective makes it effectively
  much wider in the distance than in the foreground.
- People already present when the video starts are invisible to the count —
  the baseline is assumed to be zero.
- A detector miss at the line is a counting miss; distant small figures are the
  usual culprits.
- **`data/people-walking.mp4` is a thoroughfare, not a doorway.** A line across
  it measures traversals, not entries to a bounded space. See
  `docs/experiments.md`.
- **The IPID clips in `data/datasets/ipid/` are moving-camera dashcam footage**,
  not CCTV. Detection and tracking run on them, but a fixed counting line is
  meaningless when the camera itself is moving. See `docs/experiments.md`.

### Recording a proper counting test video

The repository has no clip with real entry/exit geometry, so no counting
accuracy figure can be computed from it. To get one:

1. Mount the camera so a **single boundary** (doorway, gate, corridor mouth)
   crosses the frame roughly horizontally, ideally in the middle third — not
   near the frame edge, where bounding boxes get clipped and the bottom-center
   reference point stops tracking the feet.
2. Record 2–5 minutes with a **known script**: e.g. person A in, person B in,
   person A out, two people in together, someone who walks up to the line and
   turns back. Write the script down as you shoot — that written list is your
   ground truth.
3. Include at least one deliberate awkward case: two people crossing abreast,
   someone pausing on the line, someone partially occluded while crossing.
4. Save as `data/line_crossing_test.mp4`, find the boundary's pixel row (open
   one frame in any image viewer), and run with `--line <that row> --debug
   --events data/test_events.json`.
5. Compare the event list against your written script. That difference — not
   anything in this repo today — is the counting accuracy.

## Roadmap

See `docs/architecture.md` for the full 16-phase plan.

- [x] **Phase 1 — Person detection** (`src/detect.py`)
- [x] **Phase 2 — Multi-object tracking** with persistent anonymous IDs
      (`src/track_people.py`, BoT-SORT, ReID off)
- [x] **Phase 3 — Entry/exit counting, occupancy, visit sessions**
      (`src/counting.py`, `src/sessions.py`)
- [ ] Phase 4+ — zones, dwell time, heatmaps, event analytics, database, API,
      dashboard. Not started; measured results for each completed phase go in
      `docs/experiments.md`.
