from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from face_scan.detector import FaceDetection
from face_scan.workflows import run_live_capture, run_video_detection


class FakeCapture:
    def __init__(self) -> None:
        self._frames = [np.zeros((24, 32, 3), dtype=np.uint8)]

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 32
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 24
        if prop == cv2.CAP_PROP_FPS:
            return 20
        return 0

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        return None


class RecordingDetector:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def detect(self, frame, **kwargs):
        return [FaceDetection((2, 2, 10, 10), 100, (7, 7), 0.1)], 0.001

    def redact_faces(self, frame, detections, *, mode: str) -> None:
        self.events.append(f"redact:{mode}")

    def draw_detections(self, frame, detections, label: bool) -> None:
        self.events.append("draw")

    def overlay_metrics(self, frame, **kwargs) -> None:
        self.events.append("metrics")


class SnapshotPrivacyTests(unittest.TestCase):
    def test_video_snapshot_is_saved_after_redaction(self) -> None:
        events: list[str] = []
        detector = RecordingDetector(events)

        def record_snapshot(frame, directory, prefix="face") -> str:
            events.append("snapshot")
            return str(Path(directory) / f"{prefix}.jpg")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("face_scan.workflows.prepare_capture", return_value=FakeCapture()),
                patch("face_scan.workflows.save_snapshot", side_effect=record_snapshot),
                patch("face_scan.workflows.cv2.destroyAllWindows"),
            ):
                run_video_detection(
                    source_path="clip.mp4",
                    detector=detector,
                    scale_factor=1.1,
                    min_neighbors=5,
                    min_size=(10, 10),
                    privacy="blur",
                    draw_labels=False,
                    show_metrics=False,
                    output_path=None,
                    summary_logger=logging.getLogger("test"),
                    sample_every=1,
                    max_frames=1,
                    snapshot_dir=tmpdir,
                    snapshot_interval=0,
                    no_display=True,
                )

        self.assertLess(events.index("redact:blur"), events.index("snapshot"))

    def test_live_snapshot_is_saved_after_redaction(self) -> None:
        events: list[str] = []
        detector = RecordingDetector(events)

        def record_snapshot(frame, directory, prefix="face") -> str:
            events.append("snapshot")
            return str(Path(directory) / f"{prefix}.jpg")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("face_scan.workflows.prepare_capture", return_value=FakeCapture()),
                patch("face_scan.workflows.save_snapshot", side_effect=record_snapshot),
                patch("face_scan.workflows.cv2.destroyAllWindows"),
            ):
                run_live_capture(
                    source=0,
                    detector=detector,
                    scale_factor=1.1,
                    min_neighbors=5,
                    min_size=(10, 10),
                    privacy="black",
                    draw_labels=False,
                    show_metrics=False,
                    width=0,
                    height=0,
                    output_path=None,
                    output_fps=20,
                    snapshot_dir=tmpdir,
                    snapshot_interval=0,
                    timeout=0,
                    reconnect_attempts=0,
                    reconnect_delay=0,
                    no_display=True,
                    logger=logging.getLogger("test"),
                )

        self.assertLess(events.index("redact:black"), events.index("snapshot"))


if __name__ == "__main__":
    unittest.main()
