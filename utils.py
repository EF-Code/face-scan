"""Shared utilities for logging and small helpers."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional


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
    """
    Configure and return a shared project logger.

    Idempotent: calling this multiple times will not duplicate handlers.
    """
    logger = logging.getLogger("face_scan")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
