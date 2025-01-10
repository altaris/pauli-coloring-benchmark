"""Everything related to logging"""

import sys

from loguru import logger as logging

LOGGING_LEVELS = ["critical", "debug", "error", "info", "warning"]
"""Allowed logging levels (up to case insensitivity)"""


def setup_logging(logging_level: str = "info") -> None:
    """
    Sets logging format and level. The format is

        %(asctime)s [%(levelname)-8s] %(message)s

    e.g.

        2022-02-01 10:41:43,797 [INFO    ] Hello world
        2022-02-01 10:42:12,488 [CRITICAL] We're out of beans!

    Args:
        logging_level (str): Logging level in `LOGGING_LEVELS` (case
            insensitive).
    """
    if logging_level.lower() not in LOGGING_LEVELS:
        raise ValueError(
            "Logging level must be one of "
            + ", ".join(map(lambda s: f"'{s}'", LOGGING_LEVELS))
            + " (case insensitive)"
        )
    logging.remove()
    logging.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            + "[<level>{level: <8}</level>] "
            + "<level>{message}</level>"
        ),
        level=logging_level.upper(),
        enqueue=True,
        colorize=True,
    )
