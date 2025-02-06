"""Utilities"""

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec

import h5py
import numpy as np
from loguru import logger as logging

_P = ParamSpec("_P")


def cached(
    func: Callable[_P, dict],
    cache_file: Path,
    extra: dict | None = None,
    load: bool = True,
) -> Callable[_P, dict | None]:
    """
    Decorator to cache the results of a function.

    Args:
        func (Callable[_P, dict]):
        cache_file (Path):
        extra (dict | None, optional): If provided, the result will be updated
            with these extra key-value pairs. Defaults to None.
        load (bool): If True, the cache file will be loaded if it exists.
            Otherwise, `None` is returned.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> dict | None:
        if cache_file.is_file():
            if load:
                with cache_file.open("r", encoding="utf-8") as fp:
                    result: dict = json.load(fp)
                if (hdf5_file := cache_file.with_suffix(".hdf5")).is_file():
                    with h5py.File(hdf5_file, "r") as fp:
                        for k, v in fp.items():
                            result[k] = np.array(v)
                return result
            return None
        try:
            result = func(*args, **kwargs)
            if extra:
                result.update(extra)
            arrays: dict[str, np.ndarray] = {}  # filter out array fields
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    arrays[k] = result.pop(k)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w", encoding="utf-8") as fp:
                json.dump(result, fp)
            if arrays:
                with h5py.File(cache_file.with_suffix(".hdf5"), "w") as fp:
                    for k, v in arrays.items():
                        fp.create_dataset(k, data=v)
            return result.update(arrays)
        except Exception as e:
            logging.error(
                "Failed job tied to cache file {}: {}", cache_file, e
            )
            return None

    return wrapper


def hash_dict(d: dict) -> str:
    """
    Quick and dirty way to get a unique hash for a (potentially nested)
    dictionary.
    """
    h = hashlib.sha1()
    h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
