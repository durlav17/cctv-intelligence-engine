"""
CCTV Intelligence Engine - Phases 2 + 3: tracking, counting, visit sessions.

Pipeline:
    video file -> OpenCV (read frames) -> YOLO detection + BoT-SORT tracking
    -> persistent track IDs          (Phase 2, src/tracking.py)
    -> bottom-center reference point (Phase 3, src/counting.py)
    -> line crossing -> ENTRY/EXIT events
    -> occupancy + visit sessions    (Phase 3, src/sessions.py)
    -> annotated output -> TRACKING / COUNTING summaries

Phase 1 (src/detect.py) is untouched and still runs standalone.

What Phase 2 added over Phase 1: model.track(...) instead of model(...), so
Ultralytics keeps tracker state between frames and returns a persistent id
per box.

What Phase 3 adds over Phase 2: this file reads Track objects and derives
spatial events from them. It never mutates tracking state, so `--no-counting`
reproduces Phase 2 output exactly.

Still NOT in this phase: database, API, dashboard, face recognition, ReID,
cross-video identity, zones, dwell time, category classification.
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from counting import (
    BOTTOM_TO_TOP,
    TOP_TO_BOTTOM,
    EVENT_ENTRY,
    SIDE_ABOVE,
    SIDE_BELOW,
    CameraConfig,
    CrossingEvent,
    LineCrossingDetector,
    reference_point,
)
from sessions import OccupancyState, SessionManager
from tracking import Track, TrackRegistry

PERSON_CLASS_ID = 0  # COCO dataset: class 0 == "person"

# How long a "P01 -> ENTRY" banner stays on screen after a crossing, in frames.
FLASH_FRAMES = 20

# CLI defaults. Named so build_camera_config() can tell "user passed the
# default" from "user passed nothing" when merging with a camera config file.
DEFAULT_DEADBAND = 10
DEFAULT_SHORT_VISIT = 1.0

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


def draw_counting_line(frame, config: CameraConfig, width: int) -> None:
    """Draw the counting line plus its deadband edges.

    The band is drawn as thin dashed-looking edges rather than a filled overlay
    so it does not obscure the people walking through it.
    """
    cv2.line(frame, (0, config.line_y), (width, config.line_y), (0, 255, 255), 2)
    if config.deadband > 0:
        for y in (config.upper_bound, config.lower_bound):
            for x in range(0, width, 30):
                cv2.line(frame, (x, y), (x + 15, y), (0, 180, 180), 1)

    arrow = "v ENTRY" if config.entry_direction == TOP_TO_BOTTOM else "^ ENTRY"
    cv2.putText(frame, arrow, (width - 130, config.line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_reference_point(frame, track: Track, side: Optional[str],
                         previous_side: Optional[str], debug: bool) -> None:
    """Mark the point the counting logic actually uses.

    Green when confidently on a side, amber inside the buffer band, so it is
    obvious at a glance why a crossing did or did not fire. With --debug the
    current and previous sides are printed next to it.
    """
    point = reference_point(track.bbox)
    colour = (0, 255, 0) if side in (SIDE_ABOVE, SIDE_BELOW) else (0, 200, 255)
    cv2.circle(frame, point, 4, colour, -1)
    if debug:
        text = "{}".format(side or "?")
        if previous_side and previous_side != side:
            text = "{}<-{}".format(side, previous_side)
        cv2.putText(frame, text, (point[0] + 6, point[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


def draw_event_flash(frame, track: Track, event_type: str) -> None:
    """Temporary banner over a person who just crossed: 'P01 -> ENTRY'."""
    x1, y1, x2, y2 = track.bbox
    colour = (0, 220, 0) if event_type == EVENT_ENTRY else (0, 0, 255)
    text = "{} -> {}".format(track.label, event_type)
    cv2.putText(frame, text, (x1, min(y2 + 22, frame.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def build_camera_config(
    camera_config_path: str,
    line_y: int,
    deadband: int,
    entry_direction: str,
    short_visit_seconds: float,
    frame_height: int,
) -> CameraConfig:
    """Resolve the camera geometry from a config file and/or CLI flags.

    Precedence: an explicit CLI flag beats the config file, which beats the
    built-in default. That order lets you sweep a line position from the
    command line without editing a camera's committed configuration.
    """
    base = (CameraConfig.from_json_file(camera_config_path)
            if camera_config_path else None)

    if base is None:
        return CameraConfig(
            line_y=resolve_line_y(line_y, frame_height),
            deadband=deadband,
            entry_direction=entry_direction,
            short_visit_seconds=short_visit_seconds,
        )

    data = base.to_dict()
    if line_y > 0:
        data["counting_line"] = line_y
    if deadband != DEFAULT_DEADBAND:
        data["deadband"] = deadband
    if entry_direction != TOP_TO_BOTTOM:
        data["entry_direction"] = entry_direction
    if short_visit_seconds != DEFAULT_SHORT_VISIT:
        data["short_visit_seconds"] = short_visit_seconds

    config = CameraConfig.from_dict(data)
    if not 0 < config.line_y < frame_height:
        raise ValueError(
            "camera {} puts the counting line at y={}, outside this video "
            "(height {})".format(config.camera_id, config.line_y, frame_height))
    return config


def resolve_line_y(requested: int, frame_height: int) -> int:
    """Pick the counting line row, defaulting to mid-frame.

    Mid-frame is a neutral default for an unknown camera, not a claim that it
    is the right place for any particular scene - set --line per camera.
    """
    line_y = requested if requested > 0 else frame_height // 2
    if not 0 < line_y < frame_height:
        raise ValueError(
            "counting line y={} is outside the frame (height {})".format(
                line_y, frame_height))
    return line_y


def session_header(video_path, model_path, tracker_cfg, src_fps, width, height,
                   frames_done, avg_fps, config: Optional[CameraConfig]) -> dict:
    """Run-level metadata shared by every JSON output.

    This is the `session` row a future database would key the tracks, events and
    visits off - which is why every JSON file carries the same one.
    """
    header = {
        "video_path": video_path,
        "model": model_path,
        "tracker": tracker_cfg,
        "source_fps": src_fps,
        "resolution": "{}x{}".format(width, height),
        "frames_processed": frames_done,
        "avg_processing_fps": round(avg_fps, 2),
    }
    if config is not None:
        header["camera_id"] = config.camera_id
        header["counting_line_y"] = config.line_y
        header["deadband"] = config.deadband
        header["entry_direction"] = config.entry_direction
        header["exit_direction"] = config.exit_direction
    return header


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def counting_summary_lines(config: CameraConfig, occupancy: OccupancyState,
                           manager: SessionManager,
                           events: List[CrossingEvent]) -> List[str]:
    """End-of-run counting report, printed after the tracking summary."""
    lines = ["", "COUNTING SUMMARY", ""]
    lines.append("counting line: y={} (deadband +/-{})".format(
        config.line_y, config.deadband))
    lines.append("entry direction: {}   exit direction: {}".format(
        config.entry_direction, config.exit_direction))
    lines.append("")
    lines.append("total entries:     {}".format(occupancy.total_entries))
    lines.append("total exits:       {}".format(occupancy.total_exits))
    lines.append("final occupancy:   {}".format(occupancy.current_occupancy))
    if occupancy.unmatched_exits:
        lines.append("unmatched exits:   {}  (left without an observed entry)".format(
            occupancy.unmatched_exits))
    lines.append("crossing events:   {}".format(len(events)))
    lines.append("")

    lines.append("VISIT SESSIONS")
    lines.append("")
    counts = manager.status_counts()
    if not counts:
        lines.append("no visits recorded")
    for status in sorted(counts):
        lines.append("  {:<18} {}".format(status + ":", counts[status]))

    completed = manager.completed_sessions()
    if completed:
        durations = [s.duration for s in completed if s.duration is not None]
        lines.append("")
        lines.append("  completed visit duration: min {:.1f}s / mean {:.1f}s / max {:.1f}s"
                     .format(min(durations), sum(durations) / len(durations),
                             max(durations)))

    short = manager.short_visits(config.short_visit_seconds)
    if short:
        lines.append("")
        lines.append("  {} completed visits shorter than {:.1f}s - possible".format(
            len(short), config.short_visit_seconds))
        lines.append("  line-hover artefacts, reported not suppressed: {}".format(
            ", ".join(s.track_id for s in short[:10])))

    open_sessions = manager.open_sessions()
    if open_sessions:
        lines.append("")
        lines.append("  {} visits never closed. No exit was observed for these,".format(
            len(open_sessions)))
        lines.append("  so no exit time is recorded and occupancy still counts them.")
    lines.append("")
    return lines


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
    counting: bool = True,
    line_y: int = 0,
    deadband: int = DEFAULT_DEADBAND,
    entry_direction: str = TOP_TO_BOTTOM,
    short_visit_seconds: float = DEFAULT_SHORT_VISIT,
    camera_config_path: str = "",
    events_path: str = "",
    sessions_path: str = "",
    debug: bool = False,
) -> dict:
    """Run detection + tracking (+ optional counting) over a video, frame by frame.

    Returns a dict with the registry and, when counting is enabled, the
    occupancy state, session manager and the list of crossing events.
    """
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

    # --- Phase 3 state. All optional: with counting=False this block is inert
    # and the run is byte-for-byte Phase 2 behaviour. ---
    config: Optional[CameraConfig] = None
    detector: Optional[LineCrossingDetector] = None
    occupancy: Optional[OccupancyState] = None
    manager: Optional[SessionManager] = None
    events: List[CrossingEvent] = []
    # label -> (event_type, frame the flash should stop) for the on-screen banner
    flashes: Dict[str, Tuple[str, int]] = {}
    # tracker_ids we have already told the session manager about
    terminated_seen = set()

    if counting:
        config = build_camera_config(
            camera_config_path=camera_config_path,
            line_y=line_y,
            deadband=deadband,
            entry_direction=entry_direction,
            short_visit_seconds=short_visit_seconds,
            frame_height=height,
        )
        detector = LineCrossingDetector(config)
        occupancy = OccupancyState()
        manager = SessionManager(camera=config.camera_id)
        print("Camera: {}  line: y={}  deadband=+/-{}  entry={}".format(
            config.camera_id, config.line_y, config.deadband,
            config.entry_direction))

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

        # --- Phase 3: spatial reasoning over the tracks this frame ---
        if counting:
            # A track the registry has just terminated is NOT an exit. Tell the
            # session manager so it can close the record honestly.
            for tracker_id, finished in registry.finished.items():
                if tracker_id in terminated_seen:
                    continue
                terminated_seen.add(tracker_id)
                manager.on_track_terminated(
                    finished.label,
                    reason="tracker terminated at frame {} (no exit crossing observed)"
                           .format(frame_index),
                )
                detector.forget(finished.label)

            for track in registry.visible_tracks():
                event = detector.update(track.label, track.bbox, frame_index,
                                        timestamp, track.tracker_id)
                if event is None:
                    continue
                events.append(event)
                occupancy.apply(event)
                manager.apply(event)
                flashes[track.label] = (event.event_type, frame_index + FLASH_FRAMES)

            draw_counting_line(frame, config, width)

        for track in registry.visible_tracks():
            draw_track(frame, track)
            if not counting:
                continue
            state = detector.state_for(track.label)
            draw_reference_point(
                frame, track,
                state.current_side if state else None,
                state.previous_side if state else None,
                debug,
            )
            flash = flashes.get(track.label)
            if flash and frame_index < flash[1]:
                draw_event_flash(frame, track, flash[0])

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        active = len(registry.visible_tracks())
        lost = len(registry.lost_tracks())

        hud_lines = [
            ("FPS: {:.1f}".format(fps), (0, 0, 255)),
            ("Active tracked people: {}".format(active), (0, 0, 255)),
        ]
        if lost:
            hud_lines.append(("Temporarily lost: {}".format(lost), (0, 200, 255)))
        if counting:
            hud_lines.append(("Entries: {}".format(occupancy.total_entries),
                              (0, 255, 0)))
            hud_lines.append(("Exits: {}".format(occupancy.total_exits),
                              (0, 128, 255)))
            hud_lines.append(("Occupancy: {}".format(occupancy.current_occupancy),
                              (0, 255, 255)))

        # Dark backdrop so the HUD stays readable when a track label sits under it.
        cv2.rectangle(frame, (0, 0), (430, 12 + 32 * len(hud_lines)), (0, 0, 0), -1)
        for row, (text, colour) in enumerate(hud_lines):
            cv2.putText(frame, text, (10, 28 + 32 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        if headless:
            writer.write(frame)
        else:
            title = ("CCTV Intelligence Engine - Phase 3 (counting)" if counting
                     else "CCTV Intelligence Engine - Phase 2 (tracking)")
            cv2.imshow(title, frame)
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

    if counting:
        # Video ended: any visit still open is OPEN_AT_END. We do not invent an
        # exit timestamp for it, and we do not treat the end of file as an exit.
        manager.close_all_at_end()
        for line in counting_summary_lines(config, occupancy, manager, events):
            print(line)
        if events_path:
            write_json(events_path, {
                "session": session_header(video_path, model_path, tracker_cfg,
                                          src_fps, width, height, frames_done,
                                          avg_fps, config),
                "events": [e.to_dict() for e in events],
            })
            print("Crossing events written to: {}".format(events_path))
        if sessions_path:
            write_json(sessions_path, {
                "session": session_header(video_path, model_path, tracker_cfg,
                                          src_fps, width, height, frames_done,
                                          avg_fps, config),
                "occupancy": occupancy.to_dict(),
                "visits": manager.to_records(),
            })
            print("Visit sessions written to: {}".format(sessions_path))

    if records_path:
        write_json(records_path, {
            "session": session_header(video_path, model_path, tracker_cfg,
                                      src_fps, width, height, frames_done,
                                      avg_fps, config),
            "tracks": registry.to_records(),
        })
        print("Track records written to: {}".format(records_path))

    return {
        "registry": registry,
        "occupancy": occupancy,
        "sessions": manager,
        "events": events,
        "avg_fps": avg_fps,
        "frames": frames_done,
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: person detection + multi-object tracking")
    # Accepted either positionally (existing convention) or as --video, so the
    # same command shape works in scripts that name every argument.
    parser.add_argument("video", nargs="?", default="",
                        help="Path to input video file")
    parser.add_argument("--video", dest="video_flag", default="",
                        help="Path to input video file (alternative to the positional form)")
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

    counting_group = parser.add_argument_group("Phase 3: entry/exit counting")
    counting_group.add_argument("--no-counting", action="store_true",
                                help="Disable Phase 3; run pure Phase 2 tracking")
    counting_group.add_argument("--camera-config", default="",
                                help="Camera geometry JSON, e.g. config/cameras/camera_01.json")
    counting_group.add_argument("--line", type=int, default=0,
                                help="Counting line y in pixels (overrides the camera config)")
    counting_group.add_argument("--deadband", type=int, default=DEFAULT_DEADBAND,
                                help="Half-height of the jitter buffer around the line (default: 10)")
    counting_group.add_argument("--entry-direction", default=TOP_TO_BOTTOM,
                                choices=[TOP_TO_BOTTOM, BOTTOM_TO_TOP],
                                help="Which direction of travel counts as ENTRY")
    counting_group.add_argument("--short-visit", type=float, default=DEFAULT_SHORT_VISIT,
                                help="Completed visits below this many seconds are flagged")
    counting_group.add_argument("--events", default="",
                                help="Optional JSON file for crossing events")
    counting_group.add_argument("--sessions", default="",
                                help="Optional JSON file for visit sessions + occupancy")
    counting_group.add_argument("--debug", action="store_true",
                                help="Draw per-track side state next to the reference point")
    args = parser.parse_args()

    video_path = args.video or args.video_flag
    if not video_path:
        parser.error("no video given: pass it positionally or with --video")
    if args.video and args.video_flag and args.video != args.video_flag:
        parser.error("two different videos given: {!r} and --video {!r}".format(
            args.video, args.video_flag))

    # Validate the camera config before loading the model: a typo should fail
    # in milliseconds, not after a 30-second run.
    if args.camera_config and not args.no_counting:
        try:
            CameraConfig.from_json_file(args.camera_config)
        except ValueError as exc:
            parser.error(str(exc))

    run_tracking(
        video_path=video_path,
        model_path=args.model,
        tracker_cfg=args.tracker,
        conf_threshold=args.conf,
        max_lost_frames=args.max_lost,
        headless=args.headless,
        output_path=args.out,
        records_path=args.records,
        max_frames=args.max_frames,
        counting=not args.no_counting,
        line_y=args.line,
        deadband=args.deadband,
        entry_direction=args.entry_direction,
        short_visit_seconds=args.short_visit,
        camera_config_path=args.camera_config,
        events_path=args.events,
        sessions_path=args.sessions,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
