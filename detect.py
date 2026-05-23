"""Image-based face detection powered by OpenCV cascades."""

from __future__ import annotations

import argparse
import os
import sys

import cv2

from face_scan.media import write_json
from face_scan.observability import AuditLogger, DEFAULT_CASCADE_SHA256, configure_logger, sha256_file
from face_scan.workflows import run_image_detection, validate_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image, annotate results, and optionally save the output."
    )
    parser.add_argument("image", nargs="?", default="image.jpg", help="Path to the input image.")
    parser.add_argument("-o", "--output", default=None, help="Path to write an annotated copy of the image.")
    parser.add_argument("--summary-json", default=None, help="Optional path for a JSON run summary.")
    parser.add_argument("--cascade", default="haarcascade_frontalface_default.xml", help="Path to the Haar cascade XML file.")
    parser.add_argument(
        "--cascade-sha256",
        default=os.getenv("FACE_SCAN_CASCADE_SHA256") or None,
        help="Expected sha256 for the cascade XML (enables integrity check).",
    )
    parser.add_argument("--skip-cascade-check", action="store_true", help="Skip cascade integrity check.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Scale factor between pyramid steps.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Minimum neighbors needed for a detection.")
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        default=[60, 60],
        metavar=("MIN_WIDTH", "MIN_HEIGHT"),
        help="Minimum face size in pixels.",
    )
    parser.add_argument("--draw-labels", action="store_true", help="Label each detected face.")
    parser.add_argument("--show-metrics", action="store_true", help="Overlay detection metrics on the image.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("FACE_SCAN_LOG_LEVEL", "INFO"),
        help="Log level.",
    )
    parser.add_argument("--log-file", default=os.getenv("FACE_SCAN_LOG_FILE") or None, help="Optional log file path.")
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=os.getenv("FACE_SCAN_LOG_FORMAT", "text"),
        help="Log output format.",
    )
    parser.add_argument(
        "--privacy",
        choices=("none", "blur", "pixelate", "black"),
        default=os.getenv("FACE_SCAN_PRIVACY", "none"),
        help="Redact detected faces for privacy.",
    )
    parser.add_argument("--audit-log", default=os.getenv("FACE_SCAN_AUDIT_LOG") or None, help="Optional audit log path.")
    parser.add_argument("--no-show", action="store_true", help="Skip the display window after detection.")
    return parser.parse_args()


def verify_cascade(args: argparse.Namespace, logger, audit: AuditLogger | None) -> bool:
    if not os.path.exists(args.cascade):
        logger.error("Cascade XML is missing: %s", args.cascade)
        if audit:
            audit.emit("detect_image_error", reason="missing_cascade")
        return False

    if args.skip_cascade_check:
        return True

    expected = args.cascade_sha256
    if expected is None and args.cascade == "haarcascade_frontalface_default.xml":
        expected = DEFAULT_CASCADE_SHA256
    if expected:
        actual = sha256_file(args.cascade)
        if actual.lower() != expected.lower():
            logger.error("Cascade integrity check failed for %s", args.cascade)
            logger.error("Expected sha256=%s got=%s", expected, actual)
            if audit:
                audit.emit(
                    "detect_image_error",
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
    if audit:
        audit.emit(
            "detect_image_start",
            image=args.image,
            cascade=args.cascade,
            privacy=args.privacy,
            output=args.output,
        )

    if not os.path.exists(args.image):
        logger.error("Input image not found: %s", args.image)
        if audit:
            audit.emit("detect_image_error", reason="missing_image")
        return 1

    if not verify_cascade(args, logger, audit):
        return 1

    try:
        detector = validate_detector(args.cascade, logger)
        summary, image, detections = run_image_detection(
            image_path=args.image,
            detector=detector,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
            privacy=args.privacy,
            draw_labels=args.draw_labels,
            show_metrics=args.show_metrics,
            output_path=args.output,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        if audit:
            audit.emit("detect_image_error", reason="runtime_failure", message=str(exc))
        return 1

    logger.info("%s", detector.summarize(detections, summary.avg_detection_seconds))
    if args.output:
        logger.info("Saved annotated image to %s", args.output)
    if args.summary_json:
        summary.summary_path = args.summary_json
        write_json(args.summary_json, summary.to_dict())
        logger.info("Wrote summary to %s", args.summary_json)

    if audit:
        audit.emit(
            "detect_image_result",
            faces=len(detections),
            detect_seconds=summary.avg_detection_seconds,
            output=args.output,
            summary_json=args.summary_json,
        )

    if not args.no_show:
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if audit:
        audit.emit("detect_image_finish", ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
