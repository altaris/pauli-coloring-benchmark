"""Coloring and reordering utilities"""

from collections import defaultdict
from typing import Any, Iterable, TypeAlias, TypeVar

import numpy as np
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

    assert isinstance(gate.operator, SparsePauliOp)
    terms = gate.operator.to_sparse_list()
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


def coloring_to_array(coloring: Coloring) -> np.ndarray:
    """
    Converts a coloring `dict[int, list[int]]` to a numpy array `c`, where
    `c[i] = k` means that term `i` has colour `k`, i.e. `i in coloring[k]`.
    """
    size = max(max(v) for v in coloring.values()) + 1
    arr = np.empty(size)
    for k, v in coloring.items():
        arr[v] = k
    return arr


def invert_dict(dct: dict[_A, _B]) -> dict[_B, list[_A]]:
    """Inverts a dictionary."""
    res = defaultdict(list)
    for k, v in dct.items():
        res[v].append(k)
    return res


def is_3sat(gate_or_op: PauliEvolutionGate | SparsePauliOp) -> bool:
    """
    A 3SAT Hamiltonian must
    * only contain terms of the form $Z_i$, $Z_i Z_j$, or $Z_i Z_j Z_k$; and
    * every tuple `(i,)`, `(i, j)` and `(i, j, k)` from the terms above is
      unique.
    This is a heuristic check, and probably not correct (doesn't exclude
    certain non-3SAT Hamiltonians).
    """
    if isinstance(gate_or_op, PauliEvolutionGate):
        assert isinstance(gate_or_op.operator, SparsePauliOp)
        gate_or_op = gate_or_op.operator
    assert isinstance(gate_or_op, SparsePauliOp)  # for typechecking
    simplices: set[tuple[int, ...]] = set()
    for pstr, qs, _ in gate_or_op.to_sparse_list():
        if not (set(list(pstr)) == {"Z"} and len(pstr) in [1, 2, 3]):
            return False
        qs = tuple(sorted(qs))
        if qs in simplices:
            return False
        simplices.add(qs)
    return True


def is_ising(
    gate_or_op: PauliEvolutionGate | SparsePauliOp, transverse_ok: bool = False
) -> bool:
    """
    Wether the Hamiltonian underpinning the evolution gate is an Ising
    Hamiltonian. Transverse field Ising Hamiltonians can also be included.
    """
    if isinstance(gate_or_op, PauliEvolutionGate):
        assert isinstance(gate_or_op.operator, SparsePauliOp)
        gate_or_op = gate_or_op.operator
    edges, acceptable_strs = set(), {"Z", "ZZ"}
    if transverse_ok:
        acceptable_strs.add("X")
    for pauli_str, qubits, *_ in gate_or_op.to_sparse_list():
        if pauli_str not in acceptable_strs:
            return False
        qubits = tuple(sorted(qubits))
        if qubits in edges:
            return False
        edges.add(qubits)
    return True


def reorder_gate_by_colors(
    gate: PauliEvolutionGate, color_dct: dict[int, int]
) -> tuple[PauliEvolutionGate, Coloring, np.ndarray]:
    """
    Given a mapping that gives a color to each term (index), produces a new
    gate whose underlying operator's terms are ordered consequently.

    Returns:
        1. the reordered gate;
        2. a `dict[int, list[int]]` that regroup term (indices) by color;
        3. a `int` array of indices that gives the new order of terms.
    """
    assert isinstance(gate.operator, SparsePauliOp)
    terms = gate.operator.to_sparse_list()
    coloring = invert_dict(color_dct)
    index = np.array(sum(coloring.values(), start=[]), dtype=int)
    operator = SparsePauliOp.from_sparse_list(
        [terms[i] for i in index], gate.num_qubits
    )
    return PauliEvolutionGate(operator, gate.params[0]), coloring, index


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
