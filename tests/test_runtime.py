from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from face_scan.detector import FaceDetector
from face_scan.runtime import inspect_opencv, require_opencv_api


class OpenCVRuntimeTests(unittest.TestCase):
    def test_inspection_reports_missing_required_apis(self) -> None:
        module = SimpleNamespace(__version__="test", __file__="/tmp/cv2.so", imread=lambda path: None)

        with patch("face_scan.runtime.installed_opencv_distributions", return_value={}):
            report = inspect_opencv(module)

        self.assertFalse(report.ok)
        self.assertIn("CascadeClassifier", report.missing_apis)
        self.assertNotIn("imread", report.missing_apis)

    def test_missing_api_error_contains_runtime_guidance(self) -> None:
        module = SimpleNamespace(__version__="5.0-test", __file__="/tmp/cv2.so")

        with patch(
            "face_scan.runtime.installed_opencv_distributions",
            return_value={"opencv-python-headless": "5.0-test"},
        ):
            with self.assertRaisesRegex(RuntimeError, "Install exactly one compatible OpenCV wheel"):
                require_opencv_api("CascadeClassifier", module)


class DetectorOptionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = object.__new__(FaceDetector)
        self.frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def test_scale_factor_must_be_greater_than_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "scale_factor"):
            self.detector.detect(self.frame, scale_factor=1.0)

    def test_min_neighbors_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_neighbors"):
            self.detector.detect(self.frame, min_neighbors=-1)

    def test_min_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_size"):
            self.detector.detect(self.frame, min_size=(0, 30))


if __name__ == "__main__":
    unittest.main()
