"""Coloring and reordering of Pauli terms for `PauliEvolutionGate`s."""

from collections import defaultdict
from itertools import combinations, product
from typing import Iterable, Literal, TypeVar

import rustworkx as rx
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

A = TypeVar("A")
B = TypeVar("B")


def _apply_coloring(
    gate: PauliEvolutionGate, color_dct: dict[int, int]
) -> PauliEvolutionGate:
    """
    Given a mapping that gives a color to each term (index), produces a new
    gate whose underlying operator's terms are ordered consequently.
    """
    assert isinstance(gate.operator, SparsePauliOp)
    terms = gate.operator.to_sparse_list()
    coloring = _invert_dict(color_dct)
    operator = SparsePauliOp.from_sparse_list(
        [terms[i] for grp in coloring.values() for i in grp],
        gate.num_qubits,
    )
    return PauliEvolutionGate(operator, gate.params[0])


def _invert_dict(dct: dict[A, B]) -> dict[B, list[A]]:
    res = defaultdict(list)
    for k, v in dct.items():
        res[v].append(k)
    return res


def _smallest_int_not_in(iterable: Iterable[int]) -> int:
    """Returns the smallest non-negative integer not in `iterable`."""
    i = 0
    for j in sorted(list(iterable)):
        if j > i:
            return i
        i += 1
    return i


def _term_groups(operator: SparsePauliOp) -> dict[int, dict[str, list[int]]]:
    """
    Create a two-level dictionary that groups terms by Pauli operator (except
    $I$) and qubit. For example, if index `i` is in `groups[qubit][pauli]`,
    then term `i` has the Pauli operator `pauli` at qubit `qubit`.

    The keys of the returned dict are among ${0, ..., N_{qubits}}$, and each
    value dictionary has keys among `{"X", "Y", "Z"}`.

    >>> op = SparsePauliOp.from_list([("IIXXII", 1), ("XZIXII", 1)])
    >>> _term_groups(op)
    defaultdict(<function __main__.<lambda>()>,
        {2: defaultdict(list, {'X': [0, 1]}),
         3: defaultdict(list, {'X': [0]}),
         4: defaultdict(list, {'Z': [1]}),
         5: defaultdict(list, {'X': [1]})})
    """
    terms = operator.to_sparse_list()
    groups: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for term_idx, (pauli_str, indices, _) in enumerate(terms):
        for qubit, pauli in zip(indices, pauli_str):
            groups[qubit][pauli].append(term_idx)
    return groups


def greedy_reordering(gate: PauliEvolutionGate) -> PauliEvolutionGate:
    """
    Reorder the terms of a `PauliEvolutionGate` following the bare minimum most
    greedy Pauli coloring scheme.

    Warning:
        The gate's operator must be a single `SparsePauliOp` object.
    """
    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    terms, groups = operator.to_sparse_list(), _term_groups(operator)
    color = {term_idx: 0 for term_idx in range(len(terms))}
    for term_idx, (pauli_str, indices, _) in enumerate(terms):
        taken: set[int] = set()
        for qubit, pauli in zip(indices, pauli_str):
            for k, v in groups[qubit].items():
                if k == pauli:
                    continue
                taken.update(color[i] for i in v)
        color[term_idx] = _smallest_int_not_in(taken)
    return _apply_coloring(gate, color)


def non_commutation_graph(operator: SparsePauliOp) -> rx.PyGraph:
    """
    Create a graph where the vertices are the (indices of the) the Pauli terms
    of the operator, and the edges are between non-commuting terms.
    """
    n_terms, groups = len(operator), _term_groups(operator)
    graph: rx.PyGraph = rx.PyGraph(multigraph=False, node_count_hint=n_terms)  # type: ignore
    graph.add_nodes_from(range(n_terms))
    for v in groups.values():
        for sg1, sg2 in combinations(v.values(), 2):
            graph.add_edges_from(list(product(sg1, sg2, [None])))
    return graph


def reorder(
    gate: PauliEvolutionGate,
    method: Literal["greedy", "degree", "independent_set", "saturation"],
) -> PauliEvolutionGate:
    """
    Applies Pauli coloring to reorder the Pauli terms in the underlying operator
    of the gate.

    The supported coloring methods are:
    * `greedy`: Cheapest but probably least efficient.
    * `degree`: Requires constructing the non-commutation graph of the operator
      which may be very expensive!
    * `independent_set`: same
    * `saturation`: same
    """
    if method == "greedy":
        return greedy_reordering(gate)
    assert isinstance(gate.operator, SparsePauliOp)
    graph = non_commutation_graph(gate.operator)
    strategy = {
        "degree": rx.ColoringStrategy.Degree,  # type: ignore
        "independent_set": rx.ColoringStrategy.IndependentSet,  # type: ignore
        "saturation": rx.ColoringStrategy.Saturation,  # type: ignore
    }[method]
    colors = rx.graph_greedy_color(graph, strategy=strategy)  # type: ignore
    return _apply_coloring(gate, colors)
