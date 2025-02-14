"""Utilities"""

from datetime import datetime
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from loguru import logger as logging

P = ParamSpec("P")
Q = TypeVar("Q")


def timed(f: Callable[P, Q]) -> Callable[P, Q]:
    """
    Decorator that reports the time it took to execute a function and logs it
    at the `info` level.
    """

    @wraps(f)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> Q:
        start = datetime.now()
        res = f(*args, **kwargs)
        logging.info("Done in: {}", datetime.now() - start)
        return res

    return _wrapped
