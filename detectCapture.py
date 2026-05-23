"""Live face detection from webcam or capture devices."""

from __future__ import annotations

import argparse
import os
import sys

from face_scan.media import parse_capture_source, write_json
from face_scan.observability import AuditLogger, DEFAULT_CASCADE_SHA256, configure_logger, sha256_file
from face_scan.workflows import run_live_capture, validate_detector


DEFAULT_CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam-based face detection experience.")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index or path to video capture device (default: 0).",
    )
    parser.add_argument("--cascade", default=DEFAULT_CASCADE_PATH, help="Path to the Haar cascade XML file.")
    parser.add_argument(
        "--cascade-sha256",
        default=os.getenv("FACE_SCAN_CASCADE_SHA256") or None,
        help="Expected sha256 for the cascade XML (enables integrity check).",
    )
    parser.add_argument("--skip-cascade-check", action="store_true", help="Skip cascade integrity check.")
    parser.add_argument("--min-width", type=int, default=80, help="Minimum face width in pixels.")
    parser.add_argument("--min-height", type=int, default=80, help="Minimum face height in pixels.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Scale factor between pyramid steps.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Minimum neighbors to confirm a detection.")
    parser.add_argument("--width", type=int, default=0, help="Force width for the capture device.")
    parser.add_argument("--height", type=int, default=0, help="Force height for the capture device.")
    parser.add_argument("--record", default=None, help="Path to save annotated video.")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS for recording.")
    parser.add_argument("--snapshot-dir", default=None, help="Directory to dump face snapshots.")
    parser.add_argument("--snapshot-interval", type=float, default=5.0, help="Minimum seconds between automated snapshots.")
    parser.add_argument(
        "--privacy",
        choices=("none", "blur", "pixelate", "black"),
        default=os.getenv("FACE_SCAN_PRIVACY", "none"),
        help="Redact detected faces for privacy.",
    )
    parser.add_argument("--timeout", type=float, default=0, help="Stop after TIMEOUT seconds (0 is unlimited).")
    parser.add_argument("--reconnect-attempts", type=int, default=0, help="Retry opening the capture source on read failure.")
    parser.add_argument("--reconnect-delay", type=float, default=0.5, help="Seconds to wait between reconnect attempts.")
    parser.add_argument("--no-display", action="store_true", help="Skip showing the live window.")
    parser.add_argument("--show-metrics", action="store_true", help="Draw FPS and face counters on the feed.")
    parser.add_argument("--draw-labels", action="store_true", help="Label each detected face.")
    parser.add_argument("--summary-json", default=None, help="Optional path for a JSON run summary.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("FACE_SCAN_LOG_LEVEL", "INFO"),
        help="Logging level for the capture experience.",
    )
    parser.add_argument("--log-file", default=os.getenv("FACE_SCAN_LOG_FILE") or None, help="Optional log file path.")
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=os.getenv("FACE_SCAN_LOG_FORMAT", "text"),
        help="Log output format.",
    )
    parser.add_argument("--audit-log", default=os.getenv("FACE_SCAN_AUDIT_LOG") or None, help="Optional audit log path.")
    return parser.parse_args()


def verify_cascade(args: argparse.Namespace, logger, audit: AuditLogger | None) -> bool:
    if not os.path.exists(args.cascade):
        logger.error("Cascade XML is missing: %s", args.cascade)
        if audit:
            audit.emit("capture_error", reason="missing_cascade")
        return False

    if args.skip_cascade_check:
        return True

    expected = args.cascade_sha256
    if expected is None and os.path.abspath(args.cascade) == DEFAULT_CASCADE_PATH:
        expected = DEFAULT_CASCADE_SHA256
    if expected:
        actual = sha256_file(args.cascade)
        if actual.lower() != expected.lower():
            logger.error("Cascade integrity check failed for %s", args.cascade)
            logger.error("Expected sha256=%s got=%s", expected, actual)
            if audit:
                audit.emit(
                    "capture_error",
                    reason="cascade_hash_mismatch",
                    expected_sha256=expected,
                    actual_sha256=actual,
                )
            return False
    return True


def main() -> int:
    args = parse_args()
    logger = configure_logger(args.log_level, log_file=args.log_file, log_format=args.log_format)
    audit = AuditLogger(args.audit_log) if args.audit_log else None
    source = parse_capture_source(args.camera)

    if audit:
        audit.emit(
            "capture_start",
            camera=str(source),
            cascade=args.cascade,
            privacy=args.privacy,
            record=args.record,
            snapshot_dir=args.snapshot_dir,
        )

    if not verify_cascade(args, logger, audit):
        return 1

    try:
        detector = validate_detector(args.cascade, logger)
        summary = run_live_capture(
            source=source,
            detector=detector,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=(args.min_width, args.min_height),
            privacy=args.privacy,
            draw_labels=args.draw_labels,
            show_metrics=args.show_metrics,
            width=args.width,
            height=args.height,
            output_path=args.record,
            output_fps=args.fps,
            snapshot_dir=args.snapshot_dir,
            snapshot_interval=args.snapshot_interval,
            timeout=args.timeout,
            reconnect_attempts=args.reconnect_attempts,
            reconnect_delay=args.reconnect_delay,
            no_display=args.no_display,
            logger=logger,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        if audit:
            audit.emit("capture_error", reason="runtime_failure", message=str(exc))
        return 1

    logger.info(
        "Capture finished: frames=%s frames_with_faces=%s total_faces=%s avg_detect=%.4fs",
        summary.frames_processed,
        summary.frames_with_faces,
        summary.total_faces,
        summary.avg_detection_seconds,
    )
    if args.summary_json:
        summary.summary_path = args.summary_json
        write_json(args.summary_json, summary.to_dict())
        logger.info("Wrote summary to %s", args.summary_json)
    if audit:
        audit.emit(
            "capture_finish",
            ok=True,
            frames=summary.frames_processed,
            frames_with_faces=summary.frames_with_faces,
            total_faces=summary.total_faces,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
