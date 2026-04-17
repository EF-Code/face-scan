"""Shared utilities for robust face detection and annotation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2


@dataclass(frozen=True)
class FaceDetection:
    rect: Tuple[int, int, int, int]
    area: int
    center: Tuple[int, int]
    coverage: float


class FaceDetector:
    """Helper around OpenCV cascades that records metadata per detection."""

    def __init__(
        self,
        cascade_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise ValueError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(
        self,
        frame: cv2.Mat,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> Tuple[List[FaceDetection], float]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid")

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3
            else frame
        )
        start = time.perf_counter()
        raw = self._cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        duration = time.perf_counter() - start

        height, width = gray.shape[:2]
        frame_area = max(1, width * height)

        detections: List[FaceDetection] = []
        for (x, y, w, h) in raw:
            area = w * h
            center = (int(x + w / 2), int(y + h / 2))
            coverage = min(1.0, area / frame_area)
            detections.append(FaceDetection((x, y, w, h), area, center, coverage))

        self._logger.debug(
            "Detected %s faces (scale=%s,nbr=%s) in %.3f s",
            len(detections),
            scale_factor,
            min_neighbors,
            duration,
        )
        return detections, duration

    @staticmethod
    def draw_detections(
        frame: cv2.Mat,
        detections: List[FaceDetection],
        label: bool = True,
        color: Tuple[int, int, int] = (12, 255, 72),
        thickness: int = 2,
        alpha: float = 0.6,
    ) -> None:
        overlay = frame.copy()
        for idx, detection in enumerate(detections, start=1):
            x, y, w, h = detection.rect
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            if label:
                label_text = f"Face {idx}"
                cv2.putText(
                    overlay,
                    label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def redact_faces(
        frame: cv2.Mat,
        detections: List[FaceDetection],
        *,
        mode: str = "blur",
    ) -> None:
        """
        Redact faces in-place for privacy.

        Modes:
        - blur: gaussian blur the face ROI
        - pixelate: downsample/upsample the ROI
        - black: fill the ROI with black
        """
        if mode not in ("blur", "pixelate", "black"):
            raise ValueError(f"Unknown redaction mode: {mode}")

        height, width = frame.shape[:2]
        for detection in detections:
            x, y, w, h = detection.rect
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(width, int(x + w))
            y1 = min(height, int(y + h))
            if x1 <= x0 or y1 <= y0:
                continue

            roi = frame[y0:y1, x0:x1]
            if mode == "black":
                roi[:] = 0
                continue

            if mode == "pixelate":
                # Keep at least 8px in each dimension so very small faces don't disappear.
                px_w = max(8, roi.shape[1] // 12)
                px_h = max(8, roi.shape[0] // 12)
                small = cv2.resize(roi, (px_w, px_h), interpolation=cv2.INTER_LINEAR)
                roi[:] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                continue

            # Blur mode: scale kernel with ROI size and keep it odd.
            kx = max(9, (roi.shape[1] // 8) | 1)
            ky = max(9, (roi.shape[0] // 8) | 1)
            roi[:] = cv2.GaussianBlur(roi, (kx, ky), 0)

    @staticmethod
    def overlay_metrics(
        frame: cv2.Mat,
        fps: Optional[float] = None,
        face_count: Optional[int] = None,
        last_snapshot: Optional[str] = None,
    ) -> None:
        lines = []
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        if face_count is not None:
            lines.append(f"Faces: {face_count}")
        if last_snapshot is not None:
            lines.append(f"Last snapshot: {last_snapshot}")

        for idx, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (10, 30 + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def summarize(detections: List[FaceDetection], duration: float) -> str:
        if not detections:
            return "No faces detected"
        avg_area = sum(d.area for d in detections) / len(detections)
        avg_coverage = sum(d.coverage for d in detections) / len(detections)
        return (
            f"Found {len(detections)} face(s) | detection time {duration:.3f}s | "
            f"avg area {avg_area:.0f}px | room coverage {avg_coverage:.2%}"
        )
