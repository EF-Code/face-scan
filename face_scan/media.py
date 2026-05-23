"""Filesystem and media I/O helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2


CaptureSource = Union[int, str]


def parse_capture_source(raw: str) -> CaptureSource:
    try:
        return int(raw)
    except ValueError:
        return raw


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    ensure_dir(parent)


def secure_file(path: str) -> None:
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def load_image(path: str) -> cv2.Mat:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def save_image(path: str, frame: cv2.Mat) -> None:
    ensure_parent_dir(path)
    if not cv2.imwrite(path, frame):
        raise OSError(f"Failed to write image to {path}")
    secure_file(path)


def timestamped_name(prefix: str, suffix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}{suffix}"


def save_snapshot(frame: cv2.Mat, directory: str, prefix: str = "face") -> str:
    ensure_dir(directory)
    snapshot_path = os.path.join(directory, timestamped_name(prefix, ".jpg"))
    save_image(snapshot_path, frame)
    return snapshot_path


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    secure_file(path)


def prepare_capture(source: CaptureSource, width: int = 0, height: int = 0) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open capture source {source}")
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def build_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    ensure_parent_dir(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    secure_file(path)
    return writer
