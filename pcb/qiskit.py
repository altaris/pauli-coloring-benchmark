"""Main module"""

import re

from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def to_evolution_gate(raw: bytes) -> PauliEvolutionGate:
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
    operator = SparsePauliOp.from_sparse_list(terms, n_qubits)
    dt = Parameter("δt")
    return PauliEvolutionGate(operator, dt)
