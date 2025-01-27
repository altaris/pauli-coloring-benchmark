"""Coloring and reordering of Pauli terms for `PauliEvolutionGate`s."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, TypeAlias, TypeVar

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

_A = TypeVar("_A")
_B = TypeVar("_B")

Coloring: TypeAlias = dict[int, list[int]]


def _gate_to_c(
    gate: PauliEvolutionGate,
) -> tuple[Any, Any, Any, Any]:
    import ctypes

    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    terms = operator.to_sparse_list()
    qb_idx, trm_start_idx, n_qb_trm = [], [], []
    curr_start_idx = 0
    for _, qubits, _ in terms:
        trm_start_idx.append(curr_start_idx)
        qb_idx.extend(qubits)
        n_qb_trm.append(len(qubits))
        curr_start_idx += len(qubits)
    return (
        (ctypes.c_int * len(qb_idx))(*qb_idx),
        (ctypes.c_size_t * len(trm_start_idx))(*trm_start_idx),
        (ctypes.c_size_t * len(n_qb_trm))(*n_qb_trm),
        len(terms),
    )


def _invert_dict(dct: dict[_A, _B]) -> dict[_B, list[_A]]:
    res = defaultdict(list)
    for k, v in dct.items():
        res[v].append(k)
    return res


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


def degree_reordering_c(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Algorithm 1.2.2.2 of

        Kosowski, Adrian and Krzysztof Manuszewski. “Classical Coloring of Graphs.” (2008).
    """
    import ctypes

    # TODO: use .dll for Windows
    library_path = Path(__file__).parent / "lib" / "coloring.so"
    c_module = ctypes.CDLL(str(library_path))
    _degree_coloring = c_module.degree_coloring
    _degree_coloring.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # qb_idx
        ctypes.POINTER(ctypes.c_size_t),  # trm_start_idx
        ctypes.POINTER(ctypes.c_size_t),  # n_qb_trm
        ctypes.c_size_t,  # n_trm
        ctypes.c_size_t,  # n_qb
        ctypes.POINTER(ctypes.c_int),  # trm_col
    ]
    _degree_coloring.restype = None
    qb_idx, trm_start_idx, n_qb_trm, n_trm = _gate_to_c(gate)
    trm_col = (ctypes.c_int * n_trm)()
    _degree_coloring(
        ctypes.cast(qb_idx, ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(trm_start_idx, ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(n_qb_trm, ctypes.POINTER(ctypes.c_size_t)),
        n_trm,
        gate.num_qubits,
        ctypes.cast(trm_col, ctypes.POINTER(ctypes.c_int)),
    )
    return _reorder_by_colors(gate, dict(enumerate(trm_col)))


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
    method: Literal["degree", "degree_c", "saturation"],
) -> PauliEvolutionGate:
    """
    Applies Pauli coloring to reorder the Pauli terms in the underlying operator
    of the gate.

    The supported coloring methods are:
    * `degree`: Least expensive.
    * `saturation`:
    * `independent_set`: Most expensive. TODO: implement
    """
    methods = {
        "degree": degree_reordering,
        "degree_c": degree_reordering_c,
        "saturation": saturation_reordering,
    }
    gate, _ = methods[method](gate)
    return gate
