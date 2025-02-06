"""Utilities"""

import hashlib
import json
from typing import Any, Callable


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
