"""Gate reordering methods"""

from typing import Literal

from qiskit.circuit.library import PauliEvolutionGate

from .degree import degree_reordering
from .degree_c import degree_reordering_c
from .saturation import saturation_reordering

__all__ = [
    "degree_reordering_c",
    "degree_reordering",
    "reorder",
    "saturation_reordering",
]


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
