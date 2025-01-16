"""Coloring and reordering of Pauli terms for `PauliEvolutionGate`s."""

from collections import defaultdict
from typing import Generator, Iterable

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def _iter_coloring(
    coloring: dict[int, list[int]],
) -> Generator[int, None, None]:
    for v in coloring.values():
        yield from v


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
    coloring = defaultdict(list)
    for term_idx, c in color.items():
        coloring[c].append(term_idx)
    operator = SparsePauliOp.from_sparse_list(
        [terms[i] for i in _iter_coloring(coloring)], gate.num_qubits
    )
    return PauliEvolutionGate(operator, gate.params[0])
