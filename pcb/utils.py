"""Utilities"""

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec

_P = ParamSpec("_P")


def cached(
    func: Callable[_P, dict], cache_file: Path, extra: dict | None = None
) -> Callable[_P, dict]:
    """
    Decorator to cache the results of a function.

    Args:
        func (Callable[_P, dict]):
        cache_file (Path):
        extra (dict | None, optional): If provided, the result will be updated
            with these extra key-value pairs. Defaults to None.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> dict:
        try:
            with cache_file.open("r", encoding="utf-8") as fp:
                result: dict = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            result = func(*args, **kwargs)
            if extra:
                result.update(extra)
            with cache_file.open("w", encoding="utf-8") as fp:
                json.dump(result, fp)
        return result

    return wrapper


def hash_dict(d: dict) -> str:
    """
    Quick and dirty way to get a unique hash for a (potentially nested)
    dictionary.
    """
    h = hashlib.sha1()
    h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
