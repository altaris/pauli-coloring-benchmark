"""Gate reordering methods"""

from typing import Literal

from qiskit.circuit.library import PauliEvolutionGate

from .degree import degree_reordering
from .degree_c import degree_reordering_c
from .misra_gries import misra_gries_reordering
from .saturation import saturation_reordering
from .simplicial import simplicial_reordering

__all__ = [
    "degree_reordering_c",
    "degree_reordering",
    "misra_gries_reordering",
    "reorder",
    "saturation_reordering",
    "simplicial_reordering",
]


def reorder(
    gate: PauliEvolutionGate,
    method: Literal[
        "degree_c",
        "degree",
        "misra_gries",
        "saturation",
        "simplicial",
    ],
) -> PauliEvolutionGate:
    """
    Applies Pauli coloring to reorder the Pauli terms in the underlying operator
    of the gate.

    The supported coloring methods are:
    * `degree_c`
    * `degree`
    * `misra_gries`: **Only for Ising Hamiltonians**
    * `saturation`
    * `simplicial`: **Only for 3SAT Hamiltonians**
    """
    methods = {
        "degree_c": degree_reordering_c,
        "degree": degree_reordering,
        "misra_gries": misra_gries_reordering,
        "saturation": saturation_reordering,
        "simplicial": simplicial_reordering,
    }
    gate, _ = methods[method](gate)
    return gate
