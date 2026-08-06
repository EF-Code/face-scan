"""Higher-level workflows for image, video, and live capture processing."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .detector import FaceDetection, FaceDetector
from .media import build_writer, load_image, prepare_capture, save_image, save_snapshot


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


@dataclass
class DetectionSummary:
    mode: str
    source: str
    frames_processed: int
    frames_with_faces: int
    total_faces: int
    max_faces_in_frame: int
    avg_faces_per_processed_frame: float
    avg_detection_seconds: float
    output_path: Optional[str] = None
    summary_path: Optional[str] = None
    snapshot_dir: Optional[str] = None
    sample_every: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_detector(
    cascade_path: str,
    logger,
) -> FaceDetector:
    try:
        return FaceDetector(cascade_path, logger=logger)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def apply_presentation(
    detector: FaceDetector,
    frame: cv2.Mat,
    detections: List[FaceDetection],
    *,
    privacy: str,
    draw_labels: bool,
    show_metrics: bool,
    fps: Optional[float],
    latency_ms: float,
    last_snapshot: Optional[str],
    frame_index: Optional[int],
) -> None:
    if privacy != "none":
        detector.redact_faces(frame, detections, mode=privacy)
    detector.draw_detections(frame, detections, label=draw_labels)
    if show_metrics:
        detector.overlay_metrics(
            frame,
            fps=fps,
            face_count=len(detections),
            latency_ms=latency_ms,
            last_snapshot=last_snapshot,
            frame_index=frame_index,
        )


def run_image_detection(
    *,
    image_path: str,
    detector: FaceDetector,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
    privacy: str,
    draw_labels: bool,
    show_metrics: bool,
    output_path: Optional[str],
) -> Tuple[DetectionSummary, cv2.Mat, List[FaceDetection]]:
    image = load_image(image_path)
    detections, duration = detector.detect(
        image,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    apply_presentation(
        detector,
        image,
        detections,
        privacy=privacy,
        draw_labels=draw_labels,
        show_metrics=show_metrics,
        fps=None,
        latency_ms=duration * 1000,
        last_snapshot=None,
        frame_index=None,
    )
    if output_path:
        save_image(output_path, image)

    summary = DetectionSummary(
        mode="image",
        source=image_path,
        frames_processed=1,
        frames_with_faces=1 if detections else 0,
        total_faces=len(detections),
        max_faces_in_frame=len(detections),
        avg_faces_per_processed_frame=float(len(detections)),
        avg_detection_seconds=duration,
        output_path=output_path,
    )
    return summary, image, detections


def run_video_detection(
    *,
    source_path: str,
    detector: FaceDetector,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
    privacy: str,
    draw_labels: bool,
    show_metrics: bool,
    output_path: Optional[str],
    summary_logger,
    sample_every: int,
    max_frames: int,
    snapshot_dir: Optional[str],
    snapshot_interval: float,
    no_display: bool,
) -> DetectionSummary:
    capture = prepare_capture(source_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    writer = build_writer(output_path, input_fps or 20.0, (width, height)) if output_path else None

    frames_seen = 0
    frames_processed = 0
    frames_with_faces = 0
    total_faces = 0
    max_faces = 0
    total_detect_seconds = 0.0
    last_snapshot = ""
    last_snapshot_time = 0.0
    playback_delay_ms = max(1, int(round(1000 / input_fps))) if input_fps and input_fps > 0 else 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            frames_seen += 1
            if max_frames > 0 and frames_seen > max_frames:
                break

            detections: List[FaceDetection] = []
            detect_time = 0.0
            if frames_seen % sample_every == 0:
                detections, detect_time = detector.detect(
                    frame,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=min_size,
                )
                frames_processed += 1
                total_detect_seconds += detect_time
                total_faces += len(detections)
                max_faces = max(max_faces, len(detections))
                if detections:
                    frames_with_faces += 1

            apply_presentation(
                detector,
                frame,
                detections,
                privacy=privacy,
                draw_labels=draw_labels,
                show_metrics=show_metrics,
                fps=input_fps or None,
                latency_ms=detect_time * 1000,
                last_snapshot=os.path.basename(last_snapshot) if last_snapshot else None,
                frame_index=frames_seen,
            )

            # Save only after presentation so privacy redaction also applies to
            # snapshots written to disk.
            if snapshot_dir and detections and (time.time() - last_snapshot_time) >= snapshot_interval:
                last_snapshot = save_snapshot(frame, snapshot_dir, prefix="video_face")
                last_snapshot_time = time.time()

            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("Face Video", frame)
                key = cv2.waitKey(playback_delay_ms) & 0xFF
                if key in (27, ord("q")):
                    break

            if frames_seen % max(1, sample_every * 25) == 0:
                summary_logger.info(
                    "Processed %s frames (%s detection passes, %s frames with faces)",
                    frames_seen,
                    frames_processed,
                    frames_with_faces,
                )
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    avg_faces = (total_faces / frames_processed) if frames_processed else 0.0
    avg_detect = (total_detect_seconds / frames_processed) if frames_processed else 0.0
    return DetectionSummary(
        mode="video",
        source=source_path,
        frames_processed=frames_processed,
        frames_with_faces=frames_with_faces,
        total_faces=total_faces,
        max_faces_in_frame=max_faces,
        avg_faces_per_processed_frame=avg_faces,
        avg_detection_seconds=avg_detect,
        output_path=output_path,
        snapshot_dir=snapshot_dir,
        sample_every=sample_every,
    )


def run_live_capture(
    *,
    source,
    detector: FaceDetector,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
    privacy: str,
    draw_labels: bool,
    show_metrics: bool,
    width: int,
    height: int,
    output_path: Optional[str],
    output_fps: int,
    snapshot_dir: Optional[str],
    snapshot_interval: float,
    timeout: float,
    reconnect_attempts: int,
    reconnect_delay: float,
    no_display: bool,
    logger,
) -> DetectionSummary:
    capture = prepare_capture(source, width=width, height=height)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_writer(output_path, output_fps, (frame_width, frame_height)) if output_path else None

    fps_meter = FPSMeter()
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    last_snapshot = ""
    last_snapshot_time = 0.0
    start_time = time.time()
    reconnect_left = max(0, int(reconnect_attempts))

    frames_processed = 0
    frames_with_faces = 0
    total_faces = 0
    max_faces = 0
    total_detect_seconds = 0.0

    try:
        while True:
            if timeout and (time.time() - start_time) > timeout:
                logger.info("Timeout reached (%.1fs), exiting.", timeout)
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                if reconnect_left > 0:
                    reconnect_left -= 1
                    logger.warning("Capture read failed; reconnecting (%s left).", reconnect_left)
                    capture.release()
                    time.sleep(max(0.0, reconnect_delay))
                    capture = prepare_capture(source, width=width, height=height)
                    continue
                logger.warning("Capture stream closed.")
                break

            frames_processed += 1
            detections, detect_time = detector.detect(
                frame,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size,
            )
            total_detect_seconds += detect_time
            total_faces += len(detections)
            max_faces = max(max_faces, len(detections))
            if detections:
                frames_with_faces += 1

            fps = fps_meter.update()
            apply_presentation(
                detector,
                frame,
                detections,
                privacy=privacy,
                draw_labels=draw_labels,
                show_metrics=show_metrics,
                fps=fps,
                latency_ms=detect_time * 1000,
                last_snapshot=os.path.basename(last_snapshot) if last_snapshot else None,
                frame_index=frames_processed,
            )

            # Automatic snapshots must honor the selected privacy mode.
            if snapshot_dir and detections and (time.time() - last_snapshot_time) >= snapshot_interval:
                last_snapshot = save_snapshot(frame, snapshot_dir)
                last_snapshot_time = time.time()
                logger.info("Automatic snapshot %s", last_snapshot)

            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("Face Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and snapshot_dir:
                    last_snapshot = save_snapshot(frame, snapshot_dir)
                    last_snapshot_time = time.time()
                    logger.info("Manual snapshot %s", last_snapshot)
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    avg_faces = (total_faces / frames_processed) if frames_processed else 0.0
    avg_detect = (total_detect_seconds / frames_processed) if frames_processed else 0.0
    return DetectionSummary(
        mode="capture",
        source=str(source),
        frames_processed=frames_processed,
        frames_with_faces=frames_with_faces,
        total_faces=total_faces,
        max_faces_in_frame=max_faces,
        avg_faces_per_processed_frame=avg_faces,
        avg_detection_seconds=avg_detect,
        output_path=output_path,
        snapshot_dir=snapshot_dir,
    )
