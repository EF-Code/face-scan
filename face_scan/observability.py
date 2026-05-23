"""Logging, hashing, and audit trail helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


DEFAULT_CASCADE_SHA256 = "a1b5468d67aa6c291f3b1d2bf98181844e4d0433c31d696a2198029e0e94bc7b"


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logger(
    level: str,
    *,
    log_file: Optional[str] = None,
    log_format: str = "text",
) -> logging.Logger:
    """Configure and return the shared project logger."""

    logger = logging.getLogger("face_scan")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AuditLogger:
    """Append-only JSONL audit log with a hash chain for tamper evidence."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._prev_hash = ""
        self._ensure_parent()
        self._load_prev_hash()

    def _ensure_parent(self) -> None:
        parent = os.path.dirname(os.path.abspath(self._path)) or "."
        os.makedirs(parent, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(parent, 0o700)
            except OSError:
                pass

    def _load_prev_hash(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "rb") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    last_hash = payload.get("hash")
                    if isinstance(last_hash, str):
                        self._prev_hash = last_hash
        except OSError:
            pass

    def emit(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "fields": fields,
            "prev_hash": self._prev_hash,
        }
        canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        entry_hash = hashlib.sha256((self._prev_hash + canonical).encode("utf-8")).hexdigest()
        payload["hash"] = entry_hash

        line = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
            if os.name == "posix":
                try:
                    os.chmod(self._path, 0o600)
                except OSError:
                    pass
        except OSError:
            return
        self._prev_hash = entry_hash
