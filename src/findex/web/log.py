"""JSON logs on stdout. The platform dashboard reads this stream."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per line. The traceback stays in the log, not the HTTP body."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if isinstance(request_id, str):
            payload["request_id"] = request_id
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("findex.web")
    logger.setLevel(level.upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("findex.web")
