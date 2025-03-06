"""Saving / loading made easy =)"""

import gzip
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy


def load(path: str | Path) -> Any:
    """Inverse of `save`"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = ".".join(path.name.split(".")[1:])
    if extension == "qpy":
        with path.open("rb") as fp:
            return qpy.load(fp)[0]
    if extension == "qpy.gz":
        with gzip.open(path, "rb") as fp:
            return qpy.load(fp)[0]
    if extension == "hdf5":
        with h5py.File(path, "r") as fp:
            return {k: np.array(v) for k, v in fp.items()}
    if extension == "json":
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    if extension == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {extension}")


def save(obj: Any, path: str | Path) -> None:
    """
    Saves an object following the file extension:
    - `.qpy`: `QuantumCircuit`
    - `.qpy.gz`: `QuantumCircuit`
    - `.hdf5`: `dict[np.ndarray]`
    - `.json`
    - `.csv`: `pd.DataFrame`
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = ".".join(path.name.split(".")[1:])
    if extension == "qpy":
        assert isinstance(obj, QuantumCircuit)
        with path.open("wb") as fp:
            qpy.dump(obj, fp)
    elif extension == "qpy.gz":
        assert isinstance(obj, QuantumCircuit)
        with gzip.open(path, "wb") as fp:
            qpy.dump(obj, fp)
    elif extension == "hdf5":
        assert isinstance(obj, dict)
        assert all(isinstance(v, np.ndarray) for v in obj.values())
        with h5py.File(path, "w") as fp:
            for k, v in obj.items():
                fp.create_dataset(
                    k, data=v, compression="gzip", compression_opts=9
                )
    elif extension == "json":
        with path.open("w", encoding="utf-8") as fp:
            json.dump(obj, fp)
    elif extension == "csv":
        assert isinstance(obj, pd.DataFrame)
        obj.to_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
