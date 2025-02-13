"""
Gate reordering method based on the degree coloring of the conflict graph of a
Hamiltonian
"""

from functools import reduce
from typing import Iterable, TypeVar

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .utils import (
    Coloring,
    reorder_gate_by_colors,
    smallest_int_not_in,
    term_groups,
)

VertexT = TypeVar("VertexT")


def degree_coloring(
    cliques: Iterable[set[VertexT]],
    forbidden: dict[VertexT, set[int]] | None = None,
) -> dict[VertexT, int]:
    """
    Vertex coloring of a graph presented as a union of cliques. The coloring
    uses the degree heuristic.
    """

    def _degree(v: VertexT) -> int:
        return sum(len(k) - 1 for k in cliques if v in k)

    forbidden = forbidden or {}
    vertices: list[VertexT] = list(reduce(set.union, cliques, set()))  # type: ignore
    vertices = sorted(vertices, key=_degree, reverse=True)
    color = {v: 0 for v in vertices}
    for v in vertices:
        taken: set[int] = forbidden.get(v) or set()
        for k in cliques:
            if v in k:
                taken.update(color[u] for u in k if u != v)
        color[v] = smallest_int_not_in(taken)
    return color


def degree_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring, list[int]]:
    """
    Algorithm 1.2.2.2 of

        Kosowski, Adrian and Krzysztof Manuszewski. “Classical Coloring of
        Graphs.” (2008).

    Returns:
        See `pcb.reordering.utils.reorder_gate_by_colors`.
    """
    assert isinstance(gate.operator, SparsePauliOp)
    groups = term_groups(gate.operator)
    color = degree_coloring(groups.values())
    return reorder_gate_by_colors(gate, color)
