"""Home-baked coloring from 3SAT Hamiltonians"""

from collections import defaultdict
from functools import reduce

import numpy as np
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from pcb.reordering.saturation import saturation_coloring
from pcb.reordering.utils import (
    Coloring,
    reorder_gate_by_colors,
    smallest_int_not_in,
)


def simplicial_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring, np.ndarray]:
    """
    Reordering method for 3SAT Hamiltonians that works in two steps:
    1. detect cliques of ternary terms, build and color the corresponding
       overlap graph;
    2. detect and color the binary terms, taking into account the colors of the
       ternary terms;
    3. color all unary terms with a fresh color.

    Returns:
        See `pcb.reordering.utils.reorder_gate_by_colors`.
    """
    # if not is_3sat(gate):
    #     raise ValueError(
    #         "The input gate's Hamiltonian is not a 3SAT Hamiltonian"
    #     )

    assert isinstance(gate.operator, SparsePauliOp)
    terms = gate.operator.to_sparse_list()

    t1: list[int] = []  # indices of unary terms
    t2_cliques: dict[int, set[int]] = defaultdict(set)
    t3_cliques: dict[int, set[int]] = defaultdict(set)
    for idx, (_, qubits, _) in enumerate(terms):
        if len(qubits) == 1:
            t1.append(idx)
        if len(qubits) == 2:
            for q in qubits:
                t2_cliques[q].add(idx)
        elif len(qubits) == 3:
            for q in qubits:
                t3_cliques[q].add(idx)

    t3_color = saturation_coloring(t3_cliques.values())

    # qubit -> set of colors used by the ternary terms touching it
    colors_at_qubit: dict[int, set[int]] = defaultdict(set)
    for idx, c in t3_color.items():
        q1, q2, q3 = terms[idx][1]  # yes i know i can make a loop but somehow
        colors_at_qubit[q1].add(c)  # i find repeated code clearer in this case
        colors_at_qubit[q2].add(c)
        colors_at_qubit[q3].add(c)

    # idx of a binary term -> set of colors that are forbidden
    forbidden: dict[int, set[int]] = defaultdict(set)
    for idx in reduce(set.union, t2_cliques.values(), set()):  # type: ignore
        q1, q2 = terms[idx][1]
        forbidden[idx] |= colors_at_qubit[q1] | colors_at_qubit[q2]

    t2_color = saturation_coloring(t2_cliques.values(), forbidden)
    for idx, c in t2_color.items():
        q1, q2 = terms[idx][1]
        colors_at_qubit[q1].add(c)
        colors_at_qubit[q2].add(c)

    color = {}
    for idx in t1:
        q = terms[idx][1][0]
        color[idx] = smallest_int_not_in(colors_at_qubit[q])
    color.update(t2_color)
    color.update(t3_color)
    return reorder_gate_by_colors(gate, color)
