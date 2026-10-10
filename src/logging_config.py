"""Central logging setup."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import WildlifeConfig, PROJECT_ROOT


def setup_logging(cfg: WildlifeConfig) -> logging.Logger:
    logger = logging.getLogger("wildlife")
    if logger.handlers:
        return logger

    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = Path(cfg.logging.file)
    if not log_file.is_absolute():
        log_file = PROJECT_ROOT / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=cfg.logging.max_bytes,
        backupCount=cfg.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
