"""Image-based face detection powered by OpenCV cascades."""
from __future__ import annotations

import argparse
import os
import sys

import cv2

from face_detector import FaceDetector
from utils import AuditLogger, DEFAULT_CASCADE_SHA256, configure_logger, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image, annotate results, and optionally save the output."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the input image.",
        default="image.jpg",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write an annotated copy of the image.",
        default=None,
    )
    parser.add_argument(
        "--cascade",
        help="Path to the Haar cascade XML file.",
        default="haarcascade_frontalface_default.xml",
    )
    parser.add_argument(
        "--cascade-sha256",
        default=os.getenv("FACE_SCAN_CASCADE_SHA256") or None,
        help="Expected sha256 for the cascade XML (enables integrity check).",
    )
    parser.add_argument(
        "--skip-cascade-check",
        action="store_true",
        help="Skip cascade integrity check.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Scale factor between pyramid steps (default: 1.1).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors needed for a detection (default: 5).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        default=[60, 60],
        metavar=("MIN_WIDTH", "MIN_HEIGHT"),
        help="Minimum face size in pixels (default: 60x60).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("FACE_SCAN_LOG_LEVEL", "INFO"),
        help="Log level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("FACE_SCAN_LOG_FILE") or None,
        help="Optional log file path (supports rotation).",
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=os.getenv("FACE_SCAN_LOG_FORMAT", "text"),
        help="Log output format (default: text).",
    )
    parser.add_argument(
        "--privacy",
        choices=("none", "blur", "pixelate", "black"),
        default=os.getenv("FACE_SCAN_PRIVACY", "none"),
        help="Redact detected faces for privacy (default: none).",
    )
    parser.add_argument(
        "--audit-log",
        default=os.getenv("FACE_SCAN_AUDIT_LOG") or None,
        help="Optional append-only JSONL audit log path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the display window after detection.",
    )
    return parser.parse_args()


def load_image(path: str) -> cv2.Mat:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


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

    if not os.path.exists(args.cascade):
        logger.error("Cascade XML is missing: %s", args.cascade)
        if audit:
            audit.emit("detect_image_error", reason="missing_cascade")
        return 1

    if not os.path.exists(args.image):
        logger.error("Input image not found: %s", args.image)
        if audit:
            audit.emit("detect_image_error", reason="missing_image")
        return 1

    if not args.skip_cascade_check:
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
                return 1

    try:
        image = load_image(args.image)
    except FileNotFoundError as exc:
        logger.error(exc)
        if audit:
            audit.emit("detect_image_error", reason="image_load_failed")
        return 1

    detector = FaceDetector(args.cascade, logger=logger)
    detections, duration = detector.detect(
        image,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=tuple(args.min_size),
    )

    if args.privacy != "none":
        detector.redact_faces(image, detections, mode=args.privacy)

    detector.draw_detections(image, detections)
    detector.overlay_metrics(image, face_count=len(detections))

    logger.info("%s", detector.summarize(detections, duration))
    if audit:
        audit.emit(
            "detect_image_result",
            faces=len(detections),
            detect_seconds=duration,
        )

    if args.output:
        saved = cv2.imwrite(args.output, image)
        if not saved:
            logger.error("Failed to write annotated image to %s", args.output)
        else:
            if os.name == "posix":
                try:
                    os.chmod(args.output, 0o600)
                except OSError:
                    logger.debug("Unable to chmod output file %s", args.output)
            logger.info("Saved annotated image to %s", args.output)

    if not args.no_show:
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if audit:
        audit.emit("detect_image_finish", ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
