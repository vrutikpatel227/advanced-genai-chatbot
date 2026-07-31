"""
utils/logger.py
Single shared logger factory so every module logs consistently
(same format, same destination) instead of each rolling its own.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import app_config, paths_config

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Safe to call repeatedly; only attaches handlers once per process.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    root = logging.getLogger()
    if not _CONFIGURED:
        paths_config.data_dir.mkdir(parents=True, exist_ok=True)
        root.setLevel(app_config.log_level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        try:
            file_handler = RotatingFileHandler(
                paths_config.log_path, maxBytes=2_000_000, backupCount=3
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # Filesystem might be read-only in some deployment targets;
            # console logging still works, so we don't crash the app.
            root.warning("Could not attach file log handler; logging to console only.")

        _CONFIGURED = True

    return logger
