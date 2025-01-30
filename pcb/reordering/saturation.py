"""
Gate reordering method based on the saturation coloring of the conflict graph of
a Hamiltonian
"""

from collections import defaultdict
from typing import Generator

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .utils import (
    Coloring,
    reorder_gate_by_colors,
    smallest_int_not_in,
    term_groups,
)


def saturation_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Algorithm 1.2.2.8 of

        Kosowski, Adrian and Krzysztof Manuszewski. “Classical Coloring of Graphs.” (2008).
    """

    def _first_uncolored() -> int:
        for i in range(len(terms)):
            if i not in color:
                return i
        raise RuntimeError("All nodes are colored baka")

    def _neighbors(i: int) -> Generator[int, None, None]:
        _, indices, _ = terms[i]
        for qubit in indices:
            for j in groups[qubit]:
                if j != i:
                    yield j

    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    terms, groups = operator.to_sparse_list(), term_groups(operator)
    color: dict[int, int] = {}
    while len(color) < len(terms):
        # Loop invariant: at the beginning of each iteration, every connected
        # component is either fully colored or fully uncolored
        i = _first_uncolored()  # => the conn. cmp. of i is uncolored
        color[i] = 0
        # The fringe dict contains uncolored nodes touching at least one colored
        # node, and maps them to the (necessarily non empty) set of neighboring
        # colors. At any point, the highest saturation uncolored node is
        # necessarily in the fringe, so this restricts the search space
        fringe = defaultdict(set)
        for j in _neighbors(i):
            fringe[j].add(0)
        while fringe:
            # get highest saturation node in the fringe
            i, nci = max(fringe.items(), key=lambda e: len(e[1]))
            color[i] = smallest_int_not_in(nci)  # assign color
            del fringe[i]  # remove i from fringe since it's now colored
            # update neighbors of i (which potentially adds them to fringe)
            for j in _neighbors(i):
                if j not in color:
                    fringe[j].add(color[i])
        # fringe is empty, meaning that we fully colored a connected component
    return reorder_gate_by_colors(gate, color)
