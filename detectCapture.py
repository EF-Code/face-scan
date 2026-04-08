"""Live face detection from webcam with metrics, recordings, and automated snapshots."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2

from face_detector import FaceDetector


class FPSMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self._last: float = 0.0
        self._value: float = 0.0
        self._smoothing = smoothing

    def update(self) -> float:
        now = time.perf_counter()
        if self._last == 0.0:
            self._last = now
            return self._value
        delta = now - self._last
        self._last = now
        if delta <= 0:
            return self._value
        fps = 1.0 / delta
        if self._value == 0:
            self._value = fps
        else:
            self._value = fps * (1 - self._smoothing) + self._value * self._smoothing
        return self._value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam-based face detection experience.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index or path to video capture device (default: 0).",
    )
    parser.add_argument(
        "--cascade",
        default="haarcascade_frontalface_default.xml",
        help="Path to the Haar cascade XML file.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=80,
        help="Minimum face width in pixels.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=80,
        help="Minimum face height in pixels.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Scale factor between pyramid steps.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors to confirm a detection.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Force width for the capture device (0 keeps native width).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Force height for the capture device (0 keeps native height).",
    )
    parser.add_argument(
        "--record",
        help="Path to save annotated video (e.g., output.mp4).",
        default=None,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Target FPS for recording (default: 20).",
    )
    parser.add_argument(
        "--snapshot-dir",
        help="Directory to dump face snapshots (auto-created).",
        default=None,
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between automated snapshots when faces present.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Stop after TIMEOUT seconds (0 is unlimited).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip showing the live window (useful for headless runs).",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Draw FPS/face counters on the feed.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Logging level for the capture experience.",
    )
    return parser.parse_args()


def configure_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def prepare_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open capture source {args.camera}")
    if args.width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return capture


def build_writer(path: str, fps: int, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    return writer


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_snapshot(frame: cv2.Mat, directory: str) -> str:
    ensure_dir(directory)
    suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = os.path.join(directory, f"face_{suffix}.jpg")
    cv2.imwrite(snapshot_path, frame)
    return snapshot_path


def main() -> int:
    args = parse_args()
    logger = configure_logger(args.log_level)

    if not os.path.exists(args.cascade):
        logger.error("Cascade XML is missing: %s", args.cascade)
        return 1

    try:
        capture = prepare_capture(args)
    except RuntimeError as exc:
        logger.error(exc)
        return 1

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.record:
        try:
            writer = build_writer(
                args.record, args.fps, (frame_width, frame_height)
            )
            logger.info("Recording to %s", args.record)
        except RuntimeError as exc:
            logger.warning("Unable to start recording: %s", exc)

    detector = FaceDetector(args.cascade, logger=logger)
    fps_meter = FPSMeter()

    snapshot_dir = args.snapshot_dir
    if snapshot_dir:
        ensure_dir(snapshot_dir)
    last_snapshot = ""
    last_snapshot_time = 0.0
    start_time = time.time()

    try:
        while True:
            if args.timeout and (time.time() - start_time) > args.timeout:
                logger.info("Timeout reached (%.1fs) — exiting.", args.timeout)
                break

            ret, frame = capture.read()
            if not ret or frame is None:
                logger.warning("Capture stream closed.")
                break

            detections, detect_time = detector.detect(
                frame,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
                min_size=(args.min_width, args.min_height),
            )
            detector.draw_detections(frame, detections)
            fps = fps_meter.update()

            if args.show_metrics:
                detector.overlay_metrics(
                    frame,
                    fps=fps,
                    face_count=len(detections),
                    last_snapshot=os.path.basename(last_snapshot)
                    if last_snapshot
                    else None,
                )

            status_text = f"Latency: {detect_time * 1000:.1f}ms | Faces: {len(detections)}"
            cv2.putText(
                frame,
                status_text,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            if snapshot_dir and detections and (time.time() - last_snapshot_time) >= args.snapshot_interval:
                last_snapshot = save_snapshot(frame, snapshot_dir)
                last_snapshot_time = time.time()
                logger.info("Automatic snapshot %s", last_snapshot)

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Face Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and snapshot_dir:
                    last_snapshot = save_snapshot(frame, snapshot_dir)
                    last_snapshot_time = time.time()
                    logger.info("Manual snapshot %s", last_snapshot)
            else:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
