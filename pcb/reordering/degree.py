"""
Gate reordering method based on the degree coloring of the conflict graph of a
Hamiltonian
"""

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .utils import (
    Coloring,
    reorder_gate_by_colors,
    smallest_int_not_in,
    term_groups,
)


def degree_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Algorithm 1.2.2.2 of

        Kosowski, Adrian and Krzysztof Manuszewski. “Classical Coloring of Graphs.” (2008).
    """

    def _degree(indices: list[int]) -> int:
        d = 0
        for qubit in indices:
            if qubit in groups:
                d += len(groups[qubit]) - 1
        return d

    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    terms, groups = operator.to_sparse_list(), term_groups(operator)
    color = {term_idx: 0 for term_idx in range(len(terms))}
    everything = list(enumerate(terms))
    everything.sort(key=lambda e: _degree(e[1][1]), reverse=True)
    for term_idx, (_, indices, _) in everything:
        taken: set[int] = set()
        for qubit in indices:
            taken.update(color[i] for i in groups[qubit] if i != term_idx)
        color[term_idx] = smallest_int_not_in(taken)
    return reorder_gate_by_colors(gate, color)
