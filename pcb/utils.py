"""Utilities"""

import hashlib
import json
from datetime import datetime
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from loguru import logger as logging

P = ParamSpec("P")
Q = TypeVar("Q")


def flatten_dict(dct: dict, separator: str = "/", parent: str = "") -> dict:
    """
    Transforms a nested dict into a single flat dict. Keys are concatenated
    using the provided separator.
    """
    flat = {}
    for k, v in dct.items():
        pk = (parent + separator + k) if parent else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, separator, pk))
        elif isinstance(v, (list, tuple)):
            raise ValueError(
                "Cannot flatten a dict with list or tuple values. Sorry =]"
            )
        else:
            flat[pk] = v
    return flat


def hash_dict(d: dict) -> str:
    """
    Quick and dirty way to get a unique hash for a (potentially nested)
    dictionary.
    """

    def _to_jsonable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        methods: dict[str, Callable] = {
            "__dict__": lambda o: _to_jsonable(o.__dict__),
            "__str__": str,
            "__hash__": hash,
            "__repr__": repr,
        }
        for m, f in methods.items():
            if hasattr(obj, m):
                return f(obj)
        return None  # ¯\_(ツ)_/¯

    h = hashlib.sha1()
    h.update(json.dumps(_to_jsonable(d), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


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
