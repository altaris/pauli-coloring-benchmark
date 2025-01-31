"""
Reordering of gates of Ising Hamiltonians using the Misra-Gries algorithm edge
coloring algorithm on the interaction graph of $H$. The interaction graph of $H$
is $G = (V, E)$ where $V = \\{1, ..., N \\}$ is the set of qubits, and edges are
the following:
1. if $v \\neq w$, then $(v, w) \\in E$ if and only if there is a term in $H$
   that touches both qubits $v$ and $w$
2. $(v, v) \\in E$ if and only if there is a term in $H$ that acts on qubit
   $v$ alone.

During the execution of the algorithm, some auxiliary _fan graphs_ $F_v$ are
created, where $v \\in V$. Fan graphs are directed, the vertices of $F_v$ are
edges of the form $(v, w) \\in E$, and there is an edge from $(v, w)$ to $(v,
w')$ if and only if $(v, w')$ is colored (in $G$) and this color is free for
$w$. Starting from an uncolored edge $(v, w) \\in E$, a maximal path in $F_v$
corresponds to a maximal fan of $v$.

See also:
    https://en.wikipedia.org/wiki/Misra_%26_Gries_edge_coloring_algorithm
"""

from itertools import combinations
from typing import TypeVar

import networkx as nx
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .utils import (
    Coloring,
    is_ising,
    reorder_gate_by_colors,
    smallest_int_not_in,
)

_T = TypeVar("_T")


def _free_color(g: nx.Graph, v1: int, v2: int | None = None) -> int:
    """
    Returns the first available color not incident to `v1`. If `v2` is given,
    the color will also not be incident to `v2`.
    """
    taken = _incident_colors(g, v1)
    if v2 is not None:
        taken |= _incident_colors(g, v2)
    return smallest_int_not_in(taken)


def _incident_colors(g: nx.Graph, v: int) -> set[int]:
    """Returns the colors of the colored edges incident to `v`"""
    return set(
        g.edges[(v, w)]["color"]
        for w in g.neighbors(v)
        if "color" in g.edges[(v, w)]
    )


def _invert_cdx_path(
    g: nx.Graph, cdxp: list[tuple[int, int]], c: int, d: int
) -> None:
    """
    Inverts the colors of the edges in the cdx path `cdxp` between `c` and `d`.
    """
    for e in cdxp:
        g.edges[e]["color"] = c if g.edges[e]["color"] == d else d


def _is_fan_graph_valid_edge(
    g: nx.Graph, e1: tuple[int, int], e2: tuple[int, int]
) -> bool:
    """
    `e1` and `e2` are edges in the interaction graph `g` and must share the same
    first endpoint, say `e1 = (v, w1)` and `e2 = (v, w2)`. This function returns
    `True` if the color of `e2` is free for `w1`. If `e2` is uncolored, this
    returns `False`.
    """
    c2 = g.edges[e2].get("color")
    if c2 is None:
        return False
    return _is_free(g, e1[1], c2)
    # return c2 not in _incident_colors(g, e1[1])


def _is_free(g: nx.Graph, v: int, c: int) -> bool:
    """
    Wether an edge incident to `v` has color `c`. This is equivalent to
    `c not in _incident_colors(g, v)` but slightly more efficient in principle.
    """
    for w in g.neighbors(v):
        if g.edges[(v, w)].get("color") == c:
            return False
    return True


def _longest_path(g: nx.Graph, v: _T) -> list[_T]:
    """Longest path of distinct nodes (no loops), found by DFS"""
    longest, to_process = [v], [[v]]
    while to_process:
        path = to_process.pop()
        tip, can_be_extended = path[-1], False
        for b in g.neighbors(tip):
            if b not in path:
                to_process.append(path + [b])
                can_be_extended = True
        if not can_be_extended and len(path) > len(longest):
            longest = path
    return longest


def _max_cdx_path(
    g: nx.Graph, c: int, d: int, v: int
) -> list[tuple[int, int]]:
    """
    Finds the longest path starting at `v`, whose first edge is colored `d`, and
    whose subsequent edges alternate between color `c` and color `d`.
    """
    path = [(v, v)]  # starting with a self-loop for impl. convenience
    want = d  # color we're currently looking for
    extended = True
    while extended:
        extended, tip = False, path[-1][1]
        for w in g.neighbors(tip):
            if g.edges[(tip, w)].get("color") == want and (tip, w) not in path:
                path.append((tip, w))
                extended, want = True, (c if want == d else d)
                break
    return path[1:]  # Remove the initial self-loop


def _max_fan(g: nx.Graph, e: tuple[int, int]) -> list[tuple[int, int]]:
    """Say `e = (v, w)`. Returns the largest fan of `v` starting with `e`."""
    fan_graph, v = nx.DiGraph(), e[0]
    fan_graph.add_nodes_from((v, w) for w in g.neighbors(v))
    for e1, e2 in combinations(fan_graph.nodes, 2):
        if _is_fan_graph_valid_edge(g, e1, e2):
            fan_graph.add_edge(e1, e2)
        if _is_fan_graph_valid_edge(g, e2, e1):  # pylint: disable=arguments-out-of-order
            fan_graph.add_edge(e2, e1)
    return _longest_path(fan_graph, e)


def _max_subfan(
    g: nx.Graph, fan: list[tuple[int, int]], d: int
) -> list[tuple[int, int]]:
    """
    Return `fan[:i]` where `i >= 1` is the largest index such that `fan[:i]` is
    a valid fan, and color `d` is free for vertex `fan[i-1][1]`.
    """
    i_max = 1
    for i in range(1, len(fan)):
        c = g.edges[fan[i]]["color"]
        if not _is_free(g, fan[i - 1][1], c):
            break  # No more valid fans from now on...
        if _is_free(g, fan[i][1], d):
            i_max = i + 1  # fan[:i+1] is acceptable
    return fan[:i_max]


def _rotate_fan(g: nx.Graph, fan: list[tuple[int, int]]) -> None:
    """
    Rotates a fan so that each edge now has the color of the next edge, and the
    last edge is uncolored.
    """
    for i in range(len(fan) - 1):
        g.edges[fan[i]]["color"] = g.edges[fan[i + 1]]["color"]
    if "color" in g.edges[fan[-1]]:  # happens if len(fan) == 1
        del g.edges[fan[-1]]["color"]


def misra_gries(g: nx.Graph) -> nx.Graph:
    """
    The Misra-Gries edge coloring algorithm on a graph with possible self-loops.
    """
    self_loops = list(nx.selfloop_edges(g))
    g.remove_edges_from(self_loops)
    uncolored = list(g.edges)
    for _ in range(len(uncolored)):
        (v, w) = e = uncolored.pop()
        fan = _max_fan(g, e)
        if len(fan) <= 1:
            g.edges[e]["color"] = _free_color(g, w)
            continue
        c, d = _free_color(g, v), _free_color(g, fan[-1][1])
        cdxp = _max_cdx_path(g, c, d, v)
        _invert_cdx_path(g, cdxp, c, d)
        fan = _max_subfan(g, fan, d)
        _rotate_fan(g, fan)
        g.edges[fan[-1]]["color"] = d
    g.add_edges_from(self_loops)
    for v, _ in self_loops:
        g.edges[(v, v)]["color"] = _free_color(g, v)
    return g


def misra_gries_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Reordering method for Ising Hamiltonians based on the Misra-Gries edge
    coloring algorithm.
    """

    def _edge_or_self_loop(e: list[int]) -> tuple[int, int]:
        """
        - if `e` has two elements, returns `e` as a `tuple`;
        - if `e` has only one element, say `v`, returns `(v, v)`;
        - other cases should not happen.
        """
        return tuple(e) if len(e) == 2 else (e[0], e[0])  # type: ignore

    if not is_ising(gate):
        raise ValueError(
            "The input gate's Hamiltonian is not Ising Hamiltonian."
        )
    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    terms = operator.to_sparse_list()
    interaction_graph = nx.Graph()
    interaction_graph.add_nodes_from(range(gate.num_qubits))
    interaction_graph.add_edges_from(
        _edge_or_self_loop(e) for _, e, _ in terms
    )
    misra_gries(interaction_graph)
    color = {
        i: interaction_graph.edges[_edge_or_self_loop(e)]["color"]
        for i, (_, e, _) in enumerate(terms)
    }
    return reorder_gate_by_colors(gate, color)
