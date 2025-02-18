"""Utilities"""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp


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


def hid_to_file_key(hid: str, ham_dir: str | Path) -> tuple[Path, str]:
    """
    Converts a Hamiltonian ID to a `.hdf5.zip` file path and a key inside the
    HDF5 file.
    """
    p = Path(ham_dir) / ("__".join(hid.split("/")[:-1]) + ".hdf5.zip")
    k = hid.split("/")[-1]
    return p, k


def jid_to_json_path(jid: str, output_dir: str | Path) -> Path:
    """
    Converts a job ID to a JSON file path that looks like

        output_dir / jobs / jid[:2] / jid[2:4] / jid.json

    """
    return (
        Path(output_dir)
        / "jobs"
        / jid[:2]  # spread files in subdirs
        / jid[2:4]
        / f"{jid}.json"
    )


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
    raise ValueError(f"Unsupported file extension: {extension}")


def reorder_operator(
    operator: SparsePauliOp, term_indices: np.ndarray
) -> SparsePauliOp:
    """
    Changes the order of the terms in the operator given a term index vector,
    which is just a permutation of `[0, 1, ..., len(operator) - 1]`.
    """
    terms = operator.to_sparse_list()
    return SparsePauliOp.from_sparse_list(
        [terms[i] for i in term_indices], num_qubits=operator.num_qubits
    )


def save(obj: Any, path: str | Path) -> None:
    """
    Saves an object following the file extension:
    - `.qpy`: `QuantumCircuit`
    - `.qpy.gz`: `QuantumCircuit`
    - `.hdf5`: `dict[np.ndarray]`
    - `.json`
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
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
