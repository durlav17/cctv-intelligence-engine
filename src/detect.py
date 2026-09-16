"""
CCTV Intelligence Engine — Phase 1: Person Detection from Video

Pipeline:
    video file -> OpenCV (read frames) -> YOLO (detect) -> filter "person"
    -> draw boxes + confidence -> overlay FPS -> display / save

No tracking, no DB, no API yet — those are later phases.
"""

import argparse
import sys
import time

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO dataset: class 0 == "person"


def run_detection(
    video_path: str,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.4,
    headless: bool = False,
    output_path: str = "output.mp4",
) -> None:
    """Run person detection on a video file, frame by frame.

    Args:
        video_path: path to input video.
        model_path: YOLO weights file (auto-downloaded by ultralytics if not local).
        conf_threshold: minimum confidence to keep a detection.
        headless: if True, no GUI window — write annotated frames to output_path.
        output_path: output video path, used only when headless=True.
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if headless:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

    prev_time = time.time()
    frame_count = 0
    total_person_detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Single forward pass -> boxes, classes, confidences for the whole frame
        results = model(frame, verbose=False)[0]

        person_count = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != PERSON_CLASS_ID or conf < conf_threshold:
                continue

            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {conf:.2f}"
            cv2.putText(
                frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        total_person_detections += person_count

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"People: {person_count}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if headless:
            writer.write(frame)
        else:
            cv2.imshow("CCTV Intelligence Engine - Phase 1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Done. Frames processed: {frame_count}")
    print(f"Total person detections across all frames: {total_person_detections}")
    if headless:
        print(f"Annotated video saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Person detection on a video file")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--headless", action="store_true",
                         help="No display window; save annotated video to --out instead")
    parser.add_argument("--out", default="output.mp4", help="Output path when --headless is set")
    args = parser.parse_args()

    run_detection(args.video, args.model, args.conf, args.headless, args.out)


if __name__ == "__main__":
    main()
