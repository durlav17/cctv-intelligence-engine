# Camera configurations

One JSON file per camera. This is what makes the pipeline video-agnostic: the
same code runs any video, and the camera file says where *that camera's*
counting boundary is.

```bash
python src/track_people.py --video data/clip.mp4 --camera-config config/cameras/camera_01.json
```

## Schema

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `camera_id` | string | no (default `camera_01`) | carried into every event and session record |
| `counting_line` | int | **yes** | image row (y, pixels) of the horizontal counting line |
| `entry_direction` | string | no (default `TOP_TO_BOTTOM`) | `TOP_TO_BOTTOM` or `BOTTOM_TO_TOP`; the other direction becomes EXIT |
| `deadband` | int | no (default `10`) | half-height of the jitter buffer around the line, in pixels |
| `short_visit_seconds` | float | no (default `1.0`) | completed visits below this are flagged in the summary |

Unknown keys are rejected, not ignored — a typo like `counting_lines` would
otherwise leave the line at its default and quietly produce wrong counts.

CLI flags (`--line`, `--deadband`, `--entry-direction`, `--short-visit`)
override the file when both are given, so you can sweep a line position
without editing the config.

## Choosing a counting line for a new camera

Nothing here guesses a line for an unknown scene, and the system will not try
to infer one. A human has to look at a frame and decide:

1. Export a frame: `python -c "import cv2;c=cv2.VideoCapture('your.mp4');_,f=c.read();cv2.imwrite('frame.png',f)"`
2. Open it and find the pixel row where the boundary you care about (doorway,
   gate, corridor mouth) crosses the image.
3. Keep the line away from the frame's bottom edge — a person clipped by the
   frame border has a bounding box whose bottom stops tracking their feet,
   which is what the reference point relies on.
4. Decide which way people travel when entering, and set `entry_direction`
   accordingly.

`--debug` draws the line, the deadband edges and each person's reference point,
which is the fastest way to check a chosen value before trusting its counts.
