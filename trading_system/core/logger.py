"""Structured JSON logging with secret masking."""
import json
import logging
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


_SECRET_PATTERN = re.compile(
    r"(api[_\-]?key|api[_\-]?secret|secret|token|password|auth)",
    re.IGNORECASE,
)


class _SecretMaskingFilter(logging.Filter):
    """Masks secret values in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "extra"):
            record.extra = _mask_dict(record.extra)
        return True


def _mask_dict(d: dict) -> dict:
    masked = {}
    for k, v in d.items():
        if _SECRET_PATTERN.search(str(k)):
            masked[k] = "***REDACTED***"
        elif isinstance(v, dict):
            masked[k] = _mask_dict(v)
        else:
            masked[k] = v
    return masked


class _JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra", {})
        if extra:
            payload.update(_mask_dict(extra))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger for the given module name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.addFilter(_SecretMaskingFilter())

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    file_handler = RotatingFileHandler(
        logs_dir / "trading.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JSONFormatter())
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_JSONFormatter())
    logger.addHandler(console_handler)

    return logger
