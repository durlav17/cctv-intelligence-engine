# Experiments

Measured results only. Every number below came from an actual run on this
machine — nothing here is copied from a paper or a benchmark table.

**Hardware / environment for all runs below**

| Item | Value |
| --- | --- |
| CPU | Intel64 Family 6 Model 183 (13th-gen class), Windows 11 |
| GPU | none — `torch.cuda.is_available() == False` |
| torch | 2.14.0+cpu |
| ultralytics | 8.4.153 |
| Detector | YOLOv8n (`yolov8n.pt`), conf threshold 0.4, `classes=[0]` |

All runs used `--headless`, so the measured FPS includes video decode,
inference, tracking, drawing and mp4 encoding.

---

## Phase 2 — Multi-object tracking (BoT-SORT / ByteTrack)

Date: 2026-09-16
Code: `src/track_people.py` + `src/tracking.py`

### Test 1 — Multiple people (crowded scene)

| Item | Value |
| --- | --- |
| Video | `data/people-walking.mp4` |
| Resolution | 1920x1080 |
| Source FPS | 25.0 |
| Frames | 341 (13.6 s of video) |
| Scene | Overhead-ish view of a busy transit hall, 25–55 people in frame |

Command:

```bash
python src/track_people.py data/people-walking.mp4 --headless \
    --out data/output_tracked_botsort.mp4 --records data/tracks_botsort.json
```

**BoT-SORT (`botsort.yaml`, ReID disabled — Ultralytics default)**

| Metric | Measured |
| --- | --- |
| Wall-clock time | 26.7 s |
| Average processing FPS | **12.78** (repeat runs: 9.11, 12.70, 12.78) |
| Real-time? | No — 12.8 fps against a 25 fps source, ~0.5x real time |
| Unique track IDs created | 88 |
| Tracks lasting ≥ 50 frames (2 s) | 47 |
| Tracks lasting ≥ 200 frames (8 s) | 18 |
| Longest track | 326 of 341 frames (13.0 s) |
| Consecutive-frame ID continuations | 8093 / 8440 = **95.9 %** |
| Gap events (track resumed after ≥1 missing frame) | 347 |

The FPS figure varies run to run (9.1–12.8) because this is a CPU-only box
with other load on it; the 9.11 outlier was the first run, which also paid for
model load and file cache misses.

**ByteTrack (`bytetrack.yaml`), same video, same detector**

| Metric | Measured |
| --- | --- |
| Wall-clock time | 20.5 s |
| Average processing FPS | **16.67** |
| Unique track IDs created | 89 |

ByteTrack was ~30 % faster on this clip and produced a near-identical number of
IDs (89 vs 88). On this footage the camera is static, so BoT-SORT's main
advantage — global camera-motion compensation — buys nothing, and we are paying
for it. **BoT-SORT is kept as the default anyway** because the target
application is real CCTV, where pan/tilt and camera shake are common; the
comparison is recorded here so the choice can be revisited per deployment.

### Test 2 — Single person (isolated crop)

The repository has no genuinely single-person clip, so one was constructed from
the same source by cropping the top-left corner (`[0:340, 0:300]`, upscaled 2x),
a region that one pedestrian occupies for the full clip. Generated as
`data/test_single_person.mp4`. This is a derived clip, not new footage — worth
remembering when reading the numbers.

| Item | Value |
| --- | --- |
| Resolution | 600x680 (2x upscale of a 300x340 crop) |
| Frames | 341 @ 25 fps |
| Average processing FPS | **17.57** (smaller frame, far fewer objects) |
| Unique track IDs created | 8 |
| Dominant track | **P03 — 306 of 341 frames, 13.6 s, lost 11x, reacquired 11/11** |

This is the cleanest evidence that identity persists: one person, one ID, held
across the entire clip, surviving 11 separate short dropouts. The other 7 IDs
are partial bodies clipped by the crop edge, which the detector sees
intermittently.

### Test 3 — Occlusion

No dedicated occlusion clip exists, but the crowded scene supplies natural
occlusion constantly (people crossing paths, passing behind each other, walking
under the overhead structure). Measured from the 347 gap events:

| Gap length | Count |
| --- | --- |
| 1 frame | 125 |
| 2 frames | 62 |
| 3–5 frames | 73 |
| 6–10 frames | 56 |
| 11–25 frames | 31 |

**75 % of all dropouts (260/347) lasted ≤ 5 frames (0.2 s) and the track
recovered its ID.** That is BoT-SORT's Kalman prediction doing its job: the
person is briefly undetected, the filter dead-reckons the box forward, and the
next detection re-matches. Longer gaps (>15 frames) were much rarer and are the
ones most likely to have produced a silent ID change.

### ID switches and failures observed

Reported honestly — these are eyeballed from the annotated output plus the
track statistics, not a computed MOTA/IDF1 score. We have no ground-truth
annotations for this footage, so no such metric is claimed.

1. **Fragmentation is the dominant failure.** 88 IDs were created in a scene
   that plausibly contains ~55–65 distinct people. The excess comes from people
   who left and re-entered under a new ID, and from distant small figures
   flickering in and out of the detector's confidence threshold.
2. **Short-lived junk tracks.** 12 of the 88 tracks lasted under 5 frames.
   These are almost all borderline detections at the far end of the hall, not
   real new people. `summary_lines(min_frames=...)` exists to filter these.
3. **Small/distant people are missed entirely.** In the upper third of the
   frame (people furthest from the camera) YOLOv8n frequently produces no box
   at all, so those people are never tracked. This is a *detector* limitation
   that no tracker can fix — a larger model (yolov8s/m) or tiling would be the
   remedy.
4. **Crossing pedestrians are the risky moment.** When two people of similar
   size and clothing overlap, association is IoU+motion only (ReID is off), so
   a swap is possible. Occurrences were not counted — that requires ground
   truth we don't have.
5. **Not real time on CPU.** 12.8 fps against 25 fps source. A GPU, a smaller
   input size, or frame skipping would be needed for live operation.

### Limitations of this phase

- Track IDs are valid **only within one run over one video**. Restarting the
  process renumbers everything from P01. No cross-video or cross-day identity
  is implemented or implied.
- No ReID / appearance embedding (`with_reid: False`). Identity survives
  occlusion by motion prediction alone.
- No ground truth → no MOTA, IDF1, or ID-switch count. Every "accuracy"
  statement above is qualitative and labelled as such.
- Tracking metadata is held in memory and optionally dumped to JSON
  (`--records`). No database — that is a later phase.

---

## Phase 3 — Entry/exit counting, occupancy and visit sessions

Date: 2026-09-16
Code: `src/counting.py`, `src/sessions.py`, wired into `src/track_people.py`
Same hardware/detector as the Phase 2 runs above.

### Unit tests (no video, no YOLO)

```
.\venv\Scripts\python.exe -m unittest discover -s tests -v
Ran 43 tests in 0.005s -- OK
```

Covers all ten scenarios from the Phase 3 spec plus config validation,
first-sighting handling and JSON serialisation.

### Test 1 — `data/people-walking.mp4`

| Item | Value |
| --- | --- |
| Video | `data/people-walking.mp4` |
| Resolution | 1920x1080 |
| Source FPS | 25.0 |
| Frames | 341 |
| Counting line | y = 540 (mid-frame default) |
| Deadband | +/- 10 px |
| Entry direction | TOP_TO_BOTTOM |
| Exit direction | BOTTOM_TO_TOP |

Measured:

| Metric | Value |
| --- | --- |
| Crossing events | 18 |
| Total entries | 10 |
| Total exits | 8 |
| Final occupancy | 4 |
| Unmatched exits | 2 |
| Completed sessions | **0** |
| Open sessions | 18 (3 OPEN_AT_END, 7 OPEN_TRACK_LOST, 8 UNMATCHED_EXIT) |

Identical across two consecutive runs.

**These numbers are geometrically correct and operationally meaningless, and
that distinction matters more than the numbers.** The clip is a transit
concourse, not a doorway: people walk through in both directions and never
return. A horizontal line across it measures *line traversals*, not entries to
a bounded space. Zero completed sessions is the proof — nobody crossed
downward and later crossed back up, because there is nothing to enter.

Occupancy of 4 should therefore be read as "10 downward traversals minus 6
upward ones that had a matching prior traversal", not as "4 people are in the
room".

### Test 2 — `data/test_single_person.mp4` (manual verification)

The derived single-person crop from Phase 2 (600x680, 341 frames) was used to
verify the mechanism by eye, because with few people the events can actually be
checked against the pixels.

| Metric | Value |
| --- | --- |
| Counting line | y = 340 (mid-frame default) |
| Crossing events | 3 (1 EXIT, 2 ENTRY) |
| Final occupancy | 2 |

**Hand-verified event E0002** (`P05`, ENTRY, frame 187, t=7.48 s,
TOP_TO_BOTTOM, reference point [460, 354]):

- frame 183: the person's feet are above the line, HUD reads `Entries: 0`
- frame 188: the feet are below the line, the `P05 -> ENTRY` banner is drawn,
  HUD reads `Entries: 1`

That is one event for one traversal, fired on the correct frame, at the
person's feet. This is a single hand-checked case, not an accuracy measurement.

### Performance: Phase 2 vs Phase 3

Same video, same settings, alternating runs. `--no-counting` gives the Phase 2
baseline from the identical code path.

| Run | Phase 2 (`--no-counting`) | Phase 3 (counting on) |
| --- | --- | --- |
| 1 | 11.90 fps | 12.12 fps |
| 2 | 12.84 fps | 12.33 fps |
| mean | **12.37 fps** | **12.23 fps** |

Difference: about **1 %**, smaller than the run-to-run variation on this
machine. Phase 3 adds a handful of integer comparisons per track per frame
against a YOLO forward pass, so no measurable cost is the expected result. No
optimisation was attempted or needed.

**Regression check:** both configurations produced exactly 88 unique track IDs,
confirming the counting layer does not perturb tracking.

### Observed edge cases (from the real runs, not constructed)

| Case | Occurrences | Handling |
| --- | --- | --- |
| EXIT with no prior ENTRY | 8 sessions | `UNMATCHED_EXIT`, no entry time invented, occupancy clamped at 0 and `unmatched_exits` incremented |
| ENTRY then track terminated | 7 sessions | `OPEN_TRACK_LOST`, no exit timestamp, occupancy **not** decremented |
| Video ended mid-visit | 3 sessions | `OPEN_AT_END`, no exit timestamp |
| Line hovering | not observed as events | absorbed by the deadband before reaching the event layer |

The 8 unmatched exits are exactly what a thoroughfare should produce: people
who were already "inside" (below the line) when the clip started and then
walked upward across it.

### False crossings

No phantom repeat-crossings were observed — no track produced a burst of
same-direction events, which is the signature the deadband exists to prevent.
This is an observation over 341 frames, not a guarantee.

**Counted separately and honestly:** some of the 18 events are people merely
walking past on the concourse. Whether an event is "false" depends on an
intent the geometry cannot see, which is why the honest statement is that the
line was traversed 18 times, not that 10 people entered anything.

### No accuracy metrics are claimed

There is **no ground-truth annotation** for this footage: no labelled list of
who crossed when. Precision, recall and counting accuracy are therefore not
computed and not estimated. The one verified event above is a spot check.

To obtain real accuracy figures, a purpose-recorded clip is needed — see
"Recording a proper counting test video" in `README.md`.

### Test 3 — IPID dataset (`data/datasets/ipid/clips/`)

12 clips, all 1920x1080 @ 29.97 fps. Run through the normal pipeline with no
reference to the XML annotations, as intended — inference does not depend on
ground truth.

| Clip | Frames | Line | Proc. FPS | Track IDs | Entries | Exits | Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_002 | 403 | 540 (default) | 16.03 | 13 | 0 | 0 | 0 |
| clip_002 | 403 | 900 | 16.76 | 13 | 2 | 0 | 2 |
| clip_022 | 364 | 900 | 13.23 | 58 | 10 | 0 | 10 |
| clip_029 | 236 | 900 | 14.56 | 23 | 0 | 0 | 0 |

**Detection: works.** People are found and tracked in all three clips.
**Tracking: works.** Persistent IDs are produced throughout.
**Line crossing: configurable and functioning.** Moving the line from 540 to
900 changed clip_002 from 0 events to 2, confirming the geometry is data, not
code.

**The scene requires a different line position — and more importantly, it is
the wrong kind of scene entirely.**

These are **dashcam clips recorded from a moving vehicle**, not fixed CCTV.
Two consequences, both fatal to entry/exit semantics:

1. At the default mid-frame line (y=540) the line sits in sky and treetops.
   No pedestrian's feet ever reach it, hence 0 events. Pedestrians in these
   clips have their reference points around y=800–1000, so a usable line for
   this footage is far lower.
2. **A fixed counting line assumes a fixed camera.** When the camera moves, the
   whole scene flows through image space, and a stationary pedestrian "crosses"
   a line that is itself moving over the ground. The counts below are artefacts
   of ego-motion.

The evidence is in the direction distribution: clip_022 produced **10 entries
and 0 exits** — every single crossing in the same direction, which is what you
get when the camera advances and the entire scene sweeps downward through the
frame. Three of those "entries" (P09, P10, P16) fired within two frames of each
other (frames 63–64): one camera movement, three pedestrians swept across the
line simultaneously. No doorway behaves like that.

clip_022 also produced 58 track IDs in 364 frames, far more than the scene
contains — ego-motion plus a busy roadside is hard on the tracker, and
BoT-SORT's camera-motion compensation is not enough to hold IDs here.

**Conclusion: line-crossing logic verified as functioning; IPID is not suitable
for semantic entry/exit or occupancy validation.** No accuracy claim is made
from these clips, and their counts should not be read as visitor numbers.

For reference, `clip_002.xml` contains 8 annotated `person` tracks while the
pipeline produced 13 track IDs. That is a raw observation, **not an accuracy
metric** — no detection-to-ground-truth association was performed, and the
XML plays no part in inference. Building that comparison properly is a separate
evaluation pipeline, deliberately out of scope here.

### Camera configuration and first-sighting policy

Added after the initial Phase 3 implementation:

- `config/cameras/camera_01.json` holds the geometry as data; loaded with
  `--camera-config`. Verified end-to-end: running `people-walking.mp4` via
  `--video ... --camera-config config/cameras/camera_01.json` reproduced the
  same 18 events / 10 entries / 8 exits / occupancy 4 as the equivalent CLI
  flags, and printed `Camera: camera_01  line: y=540`.
- `camera_id` now appears in every event and in the JSON session header.
- The first-sighting policy is now an explicit named state:
  `INITIAL_VISIBLE` → `OBSERVED_CROSSING`. Behaviour is unchanged (a first
  sighting was never counted) — it is now inspectable rather than implied.

### Unit tests after these additions

```
.\venv\Scripts\python.exe -m unittest discover -s tests -v
Ran 59 tests in 0.041s -- OK
```

16 new tests cover config loading, unknown-key rejection, round-tripping,
`camera_id` propagation, and the first-sighting states.

### Performance: Phase 2 vs Phase 3 (re-measured after the config work)

| Run | Phase 2 (`--no-counting`) | Phase 3 (`--camera-config`) |
| --- | --- | --- |
| 1 | 15.69 fps | 16.03 fps |
| 2 | 16.84 fps | 16.13 fps |
| mean | **16.27 fps** | **16.08 fps** |

Difference ~1.2 %, within run-to-run variation — consistent with the earlier
measurement. (Absolute FPS is higher than the first Phase 3 benchmark because
the machine was less loaded, which is exactly why only paired, alternating runs
are compared.)

Regression: 88 unique track IDs with counting on and off, unchanged from
Phase 2.

### Limitations

- The sample video has no doorway geometry, so it cannot validate entry/exit
  semantics, only the crossing mechanism.
- **A fixed counting line assumes a fixed camera.** On moving-camera footage
  (the IPID clips) crossings are generated by ego-motion and are meaningless.
  Nothing in the code detects this condition — it is the operator's job to
  point this system at a static camera.
- A fixed-pixel deadband ignores perspective: 10 px near the camera is a much
  shorter real distance than 10 px at the far end of the hall.
- A horizontal line only. Angled or polygonal boundaries are not implemented.
- Occupancy errors accumulate: a single missed exit inflates the count for the
  remainder of the run.
- People already inside the space when the video starts are invisible to an
  event-derived count — the baseline is assumed to be zero.
- Detector misses (distant people, from Phase 2) silently become counting
  misses; a person never detected at the line is never counted.
- Cosmetic: when two people cross simultaneously and stand adjacent, their
  `-> ENTRY` banners can overlap and become hard to read.

---

## Phase 1 — Person detection

Recorded retroactively; see `CHANGELOG.md` for the original notes. Phase 1 ran
detection only (`src/detect.py`), with no persistent identity between frames.
It remains runnable and unmodified.
