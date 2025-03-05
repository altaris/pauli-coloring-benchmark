"""Main module"""

import random
import re

import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def to_evolution_gate(
    h: bytes | str,
    shuffle: bool = False,
    global_phase: float | np.complex128 | None = None,
) -> PauliEvolutionGate:
    """
    Converts a serialized sparse Pauli operator (in the format explained
    `hamlib.open_hamiltonian`) to a qiskit
    [`PauliEvolutionGate`](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.PauliEvolutionGate)
    object.

    The underlying
    [`SparsePauliOp`](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp)
    object can be retrived with `PauliEvolutionGate.operator`. The time
    parameter is an abstract unbound parameter called `δt`.
    """

    def _m2t(w_str: str, p_str: str) -> tuple[str, list[int], np.complex128]:
        """
        Example: `"Z66 X81"` becomes `("ZX", [66, 81], 1.0)`.
        """
        w = np.complex128(w_str if w_str else 1.0)
        if global_phase is not None:
            w *= global_phase
        return (
            "".join(re.findall(r"[IXYZ]", p_str)),
            [int(k) for k in re.findall(r"\d+", p_str)],
            w,
        )

    h = h if isinstance(h, str) else h.decode("utf-8")
    matches = re.findall(r"(\(?[+-.\dj][^\s]*) \[([^\]]+)\]", h)
    if not matches:
        raise ValueError("No terms found in Hamiltonian: " + h)
    terms = [_m2t(*m) for m in matches]
    if shuffle:
        random.shuffle(terms)
    n_qubits = max(max(t[1]) for t in terms) + 1
    operator = SparsePauliOp.from_sparse_list(terms, n_qubits)
    dt = Parameter("δt")
    return PauliEvolutionGate(operator, dt)


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

