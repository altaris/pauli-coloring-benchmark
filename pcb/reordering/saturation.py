"""
Gate reordering method based on the saturation coloring of the conflict graph of
a Hamiltonian
"""

from collections import defaultdict
from functools import reduce
from typing import Generator, Iterable, TypeVar

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .utils import (
    Coloring,
    reorder_gate_by_colors,
    smallest_int_not_in,
    term_groups,
)

VertexT = TypeVar("VertexT")


def saturation_coloring(
    cliques: Iterable[set[VertexT]],
    forbidden: dict[VertexT, set[int]] | None = None,
) -> dict[VertexT, int]:
    """
    Vertex coloring of a graph presented as a union of cliques. The coloring
    uses the degree heuristic. Can also specify a dict that contains forbidden colors for each vertex.
    """

    def _first_uncolored() -> VertexT:
        for v in vertices:
            if v not in color:
                return v
        raise RuntimeError("All nodes are colored baka")

    def _neighbors(v: VertexT) -> Generator[VertexT, None, None]:
        r: set[VertexT] = reduce(
            set.union, [k - {v} for k in cliques if v in k], set()
        )
        yield from r

    forbidden = forbidden or {}
    vertices: list[VertexT] = list(reduce(set.union, cliques, set()))  # type: ignore
    color: dict[VertexT, int] = {}
    while len(color) < len(vertices):
        # Loop invariant: at the beginning of each iteration, every connected
        # component is either fully colored or fully uncolored
        i = _first_uncolored()  # => the conn. cmp. of i is uncolored
        color[i] = smallest_int_not_in(forbidden[i]) if i in forbidden else 0
        # The fringe dict contains uncolored nodes (in the connected component
        # of i) touching at least one colored node, and maps them to the
        # (necessarily non empty) set of neighboring colors. At any point, the
        # highest saturation uncolored node is necessarily in the fringe, so
        # this restricts the search space
        fringe: dict[VertexT, set[int]] = defaultdict(set)
        for j in _neighbors(i):
            fringe[j].add(color[i])
        while fringe:
            # get highest saturation node in the fringe
            i, taken = max(fringe.items(), key=lambda e: len(e[1]))
            taken = taken | forbidden.get(i, set())
            color[i] = smallest_int_not_in(taken)  # assign color
            del fringe[i]  # remove i from fringe since it's now colored
            # update neighbors of i (which potentially adds them to fringe)
            for j in _neighbors(i):
                if j not in color:
                    fringe[j].add(color[i])
        # fringe is empty, meaning that we fully colored a connected component
    return color


def saturation_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Algorithm 1.2.2.8 of

        Kosowski, Adrian and Krzysztof Manuszewski. “Classical Coloring of Graphs.” (2008).
    """
    assert isinstance(gate.operator, SparsePauliOp)
    groups = term_groups(gate.operator)
    color = saturation_coloring(groups.values())
    return reorder_gate_by_colors(gate, color)
