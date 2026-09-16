"""
CCTV Intelligence Engine - Phase 2: Multi-Object Tracking

Pipeline:
    video file -> OpenCV (read frames) -> YOLO detection + BoT-SORT tracking
    -> persistent track IDs -> tracking metadata (src/tracking.py)
    -> draw boxes + P01/P02 labels + confidence -> overlay FPS / active count
    -> display or save -> TRACKING SUMMARY

Phase 1 (src/detect.py) is untouched and still runs standalone.

What is new here versus Phase 1: we call model.track(...) instead of model(...).
Ultralytics runs the same YOLO forward pass, then hands the detections to a
tracker (BoT-SORT by default) which associates them with the previous frame's
tracks and returns a persistent integer id per box.

Still NOT in this phase: database, API, dashboard, face recognition,
cross-video identity, category classification.
"""

import argparse
import json
import sys
import time

import cv2
from ultralytics import YOLO

from tracking import TrackRegistry

PERSON_CLASS_ID = 0  # COCO dataset: class 0 == "person"

# Distinct BGR colours so neighbouring track IDs do not look alike.
PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 165, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]


def colour_for(label: str):
    """Stable colour per anonymous label, so P01 keeps its colour all video.

    Keyed off the digits of the label rather than hash(), because Python
    randomises string hashing per process - we want P01 to be the same colour
    in every run, not just within one run.
    """
    number = int("".join(ch for ch in label if ch.isdigit()) or 0)
    return PALETTE[number % len(PALETTE)]


def extract_detections(results, conf_threshold: float):
    """Turn one Ultralytics result into plain dicts for the registry.

    Boxes without an `id` are detections the tracker chose not to confirm as a
    track yet (low confidence, too new). We skip them: a box with no ID cannot
    carry identity, and inventing one here would defeat the point of tracking.
    """
    detections = []
    boxes = results.boxes
    if boxes is None or boxes.id is None:
        return detections

    names = results.names
    for box in boxes:
        if box.id is None:
            continue
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id != PERSON_CLASS_ID or conf < conf_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        detections.append({
            "tracker_id": int(box.id[0]),
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
        })
    return detections


def draw_track(frame, track):
    """Draw one tracked person: box, anonymous ID, confidence."""
    x1, y1, x2, y2 = track.bbox
    colour = colour_for(track.label)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label = "{} {:.2f}".format(track.label, track.confidence)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Filled strip behind the text so it stays readable on a busy background.
    cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, max(y1, th + 8)),
                  colour, -1)
    cv2.putText(frame, label, (x1 + 2, max(y1 - 5, th + 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def run_tracking(
    video_path: str,
    model_path: str = "yolov8n.pt",
    tracker_cfg: str = "botsort.yaml",
    conf_threshold: float = 0.4,
    max_lost_frames: int = 30,
    headless: bool = False,
    output_path: str = "output_tracked.mp4",
    records_path: str = "",
    max_frames: int = 0,
) -> TrackRegistry:
    """Run detection + tracking over a video file, frame by frame."""
    print("Loading model: {}".format(model_path))
    print("Tracker: {}".format(tracker_cfg))
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: could not open video: {}".format(video_path))
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Source: {}x{} @ {:.2f} fps".format(width, height, src_fps))

    writer = None
    if headless:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

    registry = TrackRegistry(max_lost_frames=max_lost_frames)

    frame_index = -1
    prev_time = time.time()
    run_start = prev_time

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if max_frames and frame_index >= max_frames:
            break

        # Video time for this frame, used for first/last_seen_timestamp.
        timestamp = frame_index / src_fps

        # persist=True is the whole trick: it tells Ultralytics to keep the
        # tracker's state between calls instead of starting fresh each frame.
        # classes=[0] restricts the tracker to people, so a chair never
        # competes for a person's ID.
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            classes=[PERSON_CLASS_ID],
            conf=conf_threshold,
            verbose=False,
        )[0]

        detections = extract_detections(results, conf_threshold)
        registry.update(detections, frame_index, timestamp)

        for track in registry.visible_tracks():
            draw_track(frame, track)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        active = len(registry.visible_tracks())
        lost = len(registry.lost_tracks())
        # Dark backdrop so the HUD stays readable when a track label sits under it.
        hud_h = 104 if lost else 72
        cv2.rectangle(frame, (0, 0), (430, hud_h), (0, 0, 0), -1)
        cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Active tracked people: {}".format(active), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if lost:
            cv2.putText(frame, "Temporarily lost: {}".format(lost), (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if headless:
            writer.write(frame)
        else:
            cv2.imshow("CCTV Intelligence Engine - Phase 2 (tracking)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    frames_done = frame_index + 1
    elapsed = time.time() - run_start
    registry.close(frame_index)

    cap.release()
    if writer is not None:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()

    avg_fps = frames_done / elapsed if elapsed > 0 else 0.0
    print("")
    print("Frames processed: {}".format(frames_done))
    print("Wall-clock time: {:.1f} s".format(elapsed))
    print("Average processing FPS: {:.2f}".format(avg_fps))
    if headless:
        print("Annotated video saved to: {}".format(output_path))

    for line in registry.summary_lines():
        print(line)

    if records_path:
        payload = {
            "session": {
                "video_path": video_path,
                "model": model_path,
                "tracker": tracker_cfg,
                "source_fps": src_fps,
                "resolution": "{}x{}".format(width, height),
                "frames_processed": frames_done,
                "avg_processing_fps": round(avg_fps, 2),
            },
            "tracks": registry.to_records(),
        }
        with open(records_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("Track records written to: {}".format(records_path))

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: person detection + multi-object tracking")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--tracker", default="botsort.yaml",
                        choices=["botsort.yaml", "bytetrack.yaml"],
                        help="Ultralytics tracker config (default: botsort.yaml)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--max-lost", type=int, default=30,
                        help="Frames a track may be unmatched before we terminate it")
    parser.add_argument("--headless", action="store_true",
                        help="No display window; save annotated video to --out instead")
    parser.add_argument("--out", default="output_tracked.mp4", help="Output path when --headless is set")
    parser.add_argument("--records", default="", help="Optional JSON file for track metadata")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = whole video)")
    args = parser.parse_args()

    run_tracking(
        video_path=args.video,
        model_path=args.model,
        tracker_cfg=args.tracker,
        conf_threshold=args.conf,
        max_lost_frames=args.max_lost,
        headless=args.headless,
        output_path=args.out,
        records_path=args.records,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
