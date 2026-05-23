# Face Scan

`face-scan` is a small OpenCV toolkit for running face detection across images, live camera feeds, and video files. The project now shares one reusable core instead of separate ad hoc scripts, and it adds structured summaries, audit trails, privacy redaction modes, and better output validation.

## What Changed

The project has been overhauled around a reusable `face_scan/` package:

- Shared detector, media, workflow, and observability modules
- Safer output handling for images, recordings, snapshots, and JSON summaries
- Dedicated video-file processing via `detect_video.py`
- Better CLI parity across image, video, and live capture modes
- Support for camera indexes or device/file paths in live capture
- Cleaner error handling when cascades or output targets are invalid
- Lightweight tests for audit logging, JSON output, and capture-source parsing

## Requirements

- Python 3.x
- OpenCV

Install dependencies:

```bash
/home/wellington/env/bin/python -m pip install -r requirements.txt
```

## Commands

Use `/home/wellington/env/bin/python` for every command below.

### Detect Faces in an Image

```bash
/home/wellington/env/bin/python detect.py image.jpg --output annotated.jpg --summary-json reports/image.json --show-metrics --draw-labels --no-show
```

Useful flags:

- `--privacy blur|pixelate|black`
- `--cascade-sha256 <sha256>`
- `--audit-log logs/audit.jsonl`
- `--log-format json --log-file logs/run.log`

### Detect Faces from a Webcam or Capture Device

```bash
/home/wellington/env/bin/python detectCapture.py --camera 0 --show-metrics --snapshot-dir snapshots --record output/live.mp4 --summary-json reports/live.json
```

Useful flags:

- `--camera 0` for numeric camera indexes
- `--camera /path/to/device-or-video-source` for string sources
- `--timeout 60`
- `--reconnect-attempts 3 --reconnect-delay 0.5`
- `--privacy blur|pixelate|black`

### Detect Faces in a Video File

```bash
/home/wellington/env/bin/python detect_video.py input.mp4 --output output/annotated.mp4 --summary-json reports/video.json --snapshot-dir snapshots/video --sample-every 2 --show-metrics
```

By default this opens an annotated playback window while processing. Press `q` or `Esc` to stop early. Use `--no-display` for headless runs.

Useful flags:

- `--sample-every N` to trade accuracy for speed
- `--max-frames N` to cap long runs
- `--snapshot-dir <dir>` to save sampled frames containing faces
- `--privacy blur|pixelate|black`

## Summary Output

Each workflow can emit a JSON summary with fields such as:

- `mode`
- `source`
- `frames_processed`
- `frames_with_faces`
- `total_faces`
- `max_faces_in_frame`
- `avg_faces_per_processed_frame`
- `avg_detection_seconds`

## Audit Log Verification

If you write an audit log with `--audit-log`, verify the tamper-evident hash chain with:

```bash
/home/wellington/env/bin/python verify_audit.py logs/audit.jsonl
```

## Tests

Run the lightweight regression suite with:

```bash
/home/wellington/env/bin/python -m unittest discover -s tests -p 'test_*.py'
```
