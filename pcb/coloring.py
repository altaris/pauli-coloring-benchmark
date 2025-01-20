"""Coloring and reordering of Pauli terms for `PauliEvolutionGate`s."""

from collections import defaultdict
from typing import Generator, Iterable, Literal, TypeAlias, TypeVar

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

_A = TypeVar("_A")
_B = TypeVar("_B")

Coloring: TypeAlias = dict[int, list[int]]


def _reorder_by_colors(
    gate: PauliEvolutionGate, color_dct: dict[int, int]
) -> tuple[PauliEvolutionGate, Coloring]:
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
    return PauliEvolutionGate(operator, gate.params[0]), coloring


def _invert_dict(dct: dict[_A, _B]) -> dict[_B, list[_A]]:
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


def _term_groups(operator: SparsePauliOp) -> dict[int, set[int]]:
    """
    Create a dictionary that groups terms by qubit on which they act. In other
    words, if index `i` is in `groups[qubit]`, then term `i` has a non-$I$ Pauli
    operator at `qubit`.

    The keys of the returned dict are among ${0, ..., N_{qubits}}$.

    >>> op = SparsePauliOp.from_list([("IIXXII", 1), ("XZIXII", 1)])
    >>> _term_groups(op)
    defaultdict(set, {2: {0, 1}, 3: {0}, 4: {1}, 5: {1}})
    """
    terms = operator.to_sparse_list()
    groups: dict[int, set[int]] = defaultdict(set)
    for term_idx, (_, indices, _) in enumerate(terms):
        for qubit in indices:
            groups[qubit].add(term_idx)
    return groups


def degree_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    TODO: cite paper
    """

    def _degree(indices: list[int]) -> int:
        d = 0
        for qubit in indices:
            if qubit in groups:
                d += len(groups[qubit]) - 1
        return d

    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    terms, groups = operator.to_sparse_list(), _term_groups(operator)
    color = {term_idx: 0 for term_idx in range(len(terms))}
    everything = list(enumerate(terms))
    everything.sort(key=lambda e: _degree(e[1][1]), reverse=True)
    for term_idx, (_, indices, _) in everything:
        taken: set[int] = set()
        for qubit in indices:
            taken.update(color[i] for i in groups[qubit] if i != term_idx)
        color[term_idx] = _smallest_int_not_in(taken)
    return _reorder_by_colors(gate, color)


def saturation_reordering(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    TODO: cite paper
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
    terms, groups = operator.to_sparse_list(), _term_groups(operator)
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
            color[i] = _smallest_int_not_in(nci)  # assign color
            del fringe[i]  # remove i from fringe since it's now colored
            # update neighbors of i (which potentially adds them to fringe)
            for j in _neighbors(i):
                if j not in color:
                    fringe[j].add(color[i])
        # fringe is empty, meaning that we fully colored a connected component
    return _reorder_by_colors(gate, color)


def reorder(
    gate: PauliEvolutionGate,
    method: Literal["degree", "saturation"],
) -> PauliEvolutionGate:
    """
    Applies Pauli coloring to reorder the Pauli terms in the underlying operator
    of the gate.

    The supported coloring methods are:
    * `degree`: Least expensive.
    * `saturation`:
    * `independent_set`: Most expensive. TODO: implement
    """
    f = degree_reordering if method == "degree" else saturation_reordering
    gate, _ = f(gate)
    return gate
