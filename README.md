# Face Recognition with OpenCV

This is a Python-based OpenCV project that detects faces in still images or live camera feeds using Haar Cascade Classifiers. The utilities now offer CLI controls, structured logging, and optional recording/snapshot workflows for production-ready observability.

## Features

* CLI tunable image detection with annotated output and metrics
* Live webcam surveillance with logging, FPS smoothing, automated snapshots, and optional recording
* Automated/manual snapshots dumped to a configurable directory
* Display overlays that show FPS, detection latency, face count, and most recent snapshot

## Requirements

* Python 3.x
* OpenCV

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Detect Faces in an Image

To run the static face detection:

```python
python detect.py
```
You can tune detection behavior through the CLI flags (e.g., `--scale-factor 1.2 --min-size 80 80 --output annotated.jpg --log-level DEBUG`), or skip the GUI display with `--no-show`.

Useful security/ops flags:

* `--privacy blur|pixelate|black` redacts detected faces in the output.
* `--cascade-sha256 ...` verifies the cascade XML hash (integrity).
* `--audit-log audit.jsonl` writes an append-only audit log with a hash chain.
* `--log-format json --log-file logs/run.log` for machine-readable logs + rotation.


Sample Output:

![image](https://github.com/user-attachments/assets/1b4a7fc8-ddac-4092-bddb-9b935c64150b)



### 2. Detect Faces from Webcam

To run real-time face detection using your webcam:

```python
python detectCapture.py --show-metrics --snapshot-dir snapshots
```
This now logs detection metrics, overlays FPS/latency, optionally records to MP4 (`--record output.mp4`), and automatically/mannually saves snapshots when faces appear (`--snapshot-interval`, `s` key). Use `--no-display` for headless runs.

Availability and integrity options:

* `--reconnect-attempts 3 --reconnect-delay 0.5` retries camera reconnects on transient failures.
* `--cascade-sha256 ...` verifies the cascade XML hash (integrity).

## Audit Log Verification

If you write an audit log (`--audit-log audit.jsonl`), you can verify its tamper-evident hash chain:

```bash
python verify_audit.py audit.jsonl
```
