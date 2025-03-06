"""Utilities"""

import hashlib
import json
from datetime import datetime
from functools import wraps
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    TypeVar,
)

from loguru import logger as logging

P = ParamSpec("P")
Q = TypeVar("Q")


class EarlyStoppingLoop(Iterable[int]):
    """
    A loop that abstract early stopping logics.

    Example:

        ```python
        loop = EarlyStoppingLoop(max_iter=100, patience=10, delta=0.01)
        loop.propose(initial_solution, float("inf"))
        for _ in loop:
            ...
            loop.propose(new_solution, new_score)
        best_solution, best_score = loop.best()
        ```
    """

    max_iter: int
    patience: int
    delta: float
    mode: Literal["min", "max"]

    _best_score: float = float("inf")
    _best_solution: Any = None
    _iteration: int = 0
    _sli: int = 0  # Nb of iterations since last improvement

    def __init__(
        self,
        max_iter: int = 100,
        patience: int = 10,
        delta: float = 0,
        mode: Literal["min", "max"] = "min",
    ):
        """
        Args:
            max_iter (int, optional):
            patience (int, optional):
            delta (float, optional):
            mode (Literal["min", "max"], optional):
        """
        self.max_iter, self.patience, self.delta = max_iter, patience, delta
        self.mode = mode

    def __iter__(self) -> Iterator:
        self._best_solution, self._best_score = None, float("inf")
        self._iteration, self._sli = 0, 0
        return self

    def __len__(self) -> int:
        return self.max_iter

    def __next__(self) -> int:
        if (self._iteration >= self.max_iter) or (self._sli >= self.patience):
            raise StopIteration
        self._iteration += 1
        return self._iteration - 1

    def best(self) -> tuple[Any, float]:
        """Returns the best solution and its score"""
        return self._best_solution, self._best_score

    def propose(self, solution: Any, score: float) -> None:
        """Proposes a new solution and score"""
        improvement = (
            self.mode == "min" and score < self._best_score - self.delta
        ) or (self.mode == "max" and score > self._best_score + self.delta)
        if improvement:
            self._best_solution, self._best_score = solution, score
            self._sli = 0
        else:
            self._sli += 1


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
