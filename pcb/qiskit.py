"""Main module"""

import random
import re

from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def to_evolution_gate(
    h: bytes | str, shuffle: bool = False
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

    def _m2t(trm_str: str) -> tuple[str, list[int], float]:
        """
        Example: `"Z66 X81"` becomes `("ZX", [66, 81], 1.0)`.
        """
        return (
            "".join(re.findall(r"[IXYZ]", trm_str)),
            [int(k) for k in re.findall(r"\d+", trm_str)],
            1.0,
        )

    h = h if isinstance(h, str) else h.decode("utf-8")
    matches = re.findall(r"\[([^\]]+)\]", h)
    if not matches:
        raise ValueError("No terms found in Hamiltonian: " + h)
    terms = [_m2t(m) for m in matches]
    if shuffle:
        random.shuffle(terms)
    n_qubits = max(max(t[1]) for t in terms) + 1
    operator = SparsePauliOp.from_sparse_list(terms, n_qubits)
    dt = Parameter("δt")
    return PauliEvolutionGate(operator, dt)
