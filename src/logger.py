"""
FillPac AI - Logger

Thin wrapper around the standard logging module: console + rotating file.
"""

import logging
import logging.handlers
import os


class Logger:

    def __init__(
        self,
        log_file="logs/application.log",
        level=logging.INFO,
        max_bytes=10_485_760,
        backup_count=5,
    ):
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

        self.logger = logging.getLogger("FillPacAI")
        self.logger.setLevel(self._parse_level(level))
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        for handler in (
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_file, encoding="utf-8", maxBytes=max_bytes, backupCount=backup_count
            ),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Direct bindings: no extra Python call layer per log line.
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.exception = self.logger.exception

    @staticmethod
    def _parse_level(level):
        if isinstance(level, int):
            return level

        if isinstance(level, str):
            return (
                logging.getLevelName(level.strip().upper())
                if isinstance(logging.getLevelName(level.strip().upper()), int)
                else logging.INFO
            )

        return logging.INFO