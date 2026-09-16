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

## Phase 1 — Person detection

Recorded retroactively; see `CHANGELOG.md` for the original notes. Phase 1 ran
detection only (`src/detect.py`), with no persistent identity between frames.
It remains runnable and unmodified.
