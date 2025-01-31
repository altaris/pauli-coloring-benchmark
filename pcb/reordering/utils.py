"""Coloring and reordering utilities"""

from collections import defaultdict
from typing import Any, Iterable, TypeAlias, TypeVar

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

_A = TypeVar("_A")
_B = TypeVar("_B")

Coloring: TypeAlias = dict[int, list[int]]


def gate_to_c(
    gate: PauliEvolutionGate,
) -> tuple[Any, Any, Any, Any]:
    """Converts a `PauliEvolutionGate` to a format suitable for C."""
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


def invert_dict(dct: dict[_A, _B]) -> dict[_B, list[_A]]:
    """Inverts a dictionary."""
    res = defaultdict(list)
    for k, v in dct.items():
        res[v].append(k)
    return res


def is_ising(gate: PauliEvolutionGate) -> bool:
    """
    Wether the Hamiltonian underpinning the evolution gate is an Ising
    Hamiltonian.
    """
    operator = gate.operator
    assert isinstance(operator, SparsePauliOp)
    edges = set()
    for pauli_str, qubits, *_ in operator.to_sparse_list():
        if not (pauli_str == "Z" or pauli_str == "ZZ"):
            return False
        qubits = tuple(sorted(qubits))
        if qubits in edges:
            return False
        edges.add(qubits)
    return True


def reorder_gate_by_colors(
    gate: PauliEvolutionGate, color_dct: dict[int, int]
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Given a mapping that gives a color to each term (index), produces a new
    gate whose underlying operator's terms are ordered consequently.
    """
    assert isinstance(gate.operator, SparsePauliOp)
    terms = gate.operator.to_sparse_list()
    coloring = invert_dict(color_dct)
    operator = SparsePauliOp.from_sparse_list(
        [terms[i] for grp in coloring.values() for i in grp],
        gate.num_qubits,
    )
    return PauliEvolutionGate(operator, gate.params[0]), coloring


def smallest_int_not_in(iterable: Iterable[int]) -> int:
    """Returns the smallest non-negative integer not in `iterable`."""
    i = 0
    for j in sorted(list(set(iterable))):
        if j > i:
            return i
        i += 1
    return i


def term_groups(operator: SparsePauliOp) -> dict[int, set[int]]:
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
