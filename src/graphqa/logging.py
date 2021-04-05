import functools
import logging as _logging
from pathlib import Path
from typing import Union

import loguru
import tqdm as _tqdm

# Disable other lib's debug messages
_default_filter = {"matplotlib": "WARNING", "": True}

# Default loguru format, but with time in UTC
_default_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


class InterceptHandler(_logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru.logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = _logging.currentframe(), 2
        while frame.f_code.co_filename == _logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru.logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    # Avoid tqdm and loguru clashes
    loguru.logger.remove()
    loguru.logger.add(
        sink=functools.partial(_tqdm.tqdm.write, end=""),
        colorize=True,
        filter=_default_filter,
        format=_default_format,
    )

    # Intercept standard logging messages toward your Loguru sinks
    _logging.basicConfig(handlers=[InterceptHandler()], level=0)


def add_logfile(path: Union[str, Path], format=None, filter=None):
    if filter is None:
        filter = _default_filter
    if format is None:
        format = _default_format
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    loguru.logger.add(sink=path, format=format, colorize=False, filter=filter, mode="a")
