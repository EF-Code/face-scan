from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from face_scan.media import parse_capture_source, write_json
from face_scan.observability import AuditLogger, configure_logger
from verify_audit import verify


class ParseCaptureSourceTests(unittest.TestCase):
    def test_numeric_camera_is_converted(self) -> None:
        self.assertEqual(parse_capture_source("2"), 2)

    def test_path_camera_is_kept_as_string(self) -> None:
        self.assertEqual(parse_capture_source("/tmp/video.mp4"), "/tmp/video.mp4")


class AuditTrailTests(unittest.TestCase):
    def test_audit_log_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            audit = AuditLogger(str(path))
            audit.emit("start", source="test")
            audit.emit("finish", ok=True)

            self.assertEqual(verify(str(path)), 0)

    def test_write_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            payload = {"mode": "image", "faces": 2}
            write_json(str(path), payload)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


class LoggerConfigTests(unittest.TestCase):
    def test_reconfigure_logger_replaces_handlers(self) -> None:
        logger = configure_logger("INFO", log_format="text")
        first_handler_ids = [id(handler) for handler in logger.handlers]

        logger = configure_logger("DEBUG", log_format="json")
        second_handler_ids = [id(handler) for handler in logger.handlers]

        self.assertEqual(logger.level, logging.DEBUG)
        self.assertNotEqual(first_handler_ids, second_handler_ids)
        self.assertGreaterEqual(len(second_handler_ids), 1)


if __name__ == "__main__":
    unittest.main()
