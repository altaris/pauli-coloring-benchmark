"""Main module"""

import re
from typing import Generator

from loguru import logger as logging
from qiskit.quantum_info import SparsePauliOp

from .hamlib import all_hamiltonian_files


def _to_spop(raw: bytes) -> SparsePauliOp:
    """
    Converts a serialized sparse Pauli operator (in the format specified in
    `hamlib.all_hamiltonian_files`) to a qiskit
    [`SparsePauliOp`](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp)
    object.
    """

    def _m2t(w_str: str, trm_str: str) -> tuple[str, list[int], float]:
        """
        Example: `('0.5', 'Z66 X81')` becomes `("ZX", [66, 81], 0.5)`.
        """
        return (
            "".join(re.findall(r"[IXYZ]", trm_str)),
            [int(k) for k in re.findall(r"\d+", trm_str)],
            float(w_str),
        )

    pattern = r"([\d.]+) \[([^\]]+)\]"
    matches = re.findall(pattern, raw.decode("utf-8"))
    terms = [_m2t(*m) for m in matches]
    n_qubits = max(max(t[1]) for t in terms) + 1
    return SparsePauliOp.from_sparse_list(terms, n_qubits)


def all_hamiltonians() -> Generator[tuple[SparsePauliOp, str], None, None]:
    """
    Iterator that yields all the Hamiltonians in the HamLib website as a qiskit
    [`SparsePauliOp`](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp)
    object, together with their unique Hamiltonian ID.
    """
    for h5fp, hfid in all_hamiltonian_files():
        for key in h5fp.keys():
            hid = hfid + "/" + key
            try:
                raw: bytes = h5fp[key][()]
                yield _to_spop(raw), hid
            except Exception as e:
                logging.error("Failed to parse Hamiltonian: {}: {}", hid, e)
