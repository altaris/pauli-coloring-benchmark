"""Gate reordering methods"""

from typing import Callable, Literal

from qiskit.circuit.library import PauliEvolutionGate

from .degree import degree_reordering
from .degree_c import degree_reordering_c
from .misra_gries import misra_gries_reordering
from .saturation import saturation_reordering
from .simplicial import simplicial_reordering
from .utils import Coloring

__all__ = [
    "Coloring",
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
) -> tuple[PauliEvolutionGate, Coloring]:
    """
    Applies Pauli coloring to reorder the Pauli terms in the underlying operator
    of the gate.

    The supported coloring methods are:
    * `degree_c`: C implementation of the `degree` method
    * `degree`
    * `misra_gries`: **Only for Ising or transverse-Ising Hamiltonians**
    * `saturation`
    * `simplicial`: **Only for 3SAT Hamiltonians**

    Returns the reordered gate and the coloring, which is a `dict[int,
    set[int]]` that maps a color to the let of all term (indices) with that
    color.
    """
    methods: dict[
        str,
        Callable[[PauliEvolutionGate], tuple[PauliEvolutionGate, Coloring]],
    ] = {
        "degree_c": degree_reordering_c,
        "degree": degree_reordering,
        "misra_gries": misra_gries_reordering,
        "saturation": saturation_reordering,
        "simplicial": simplicial_reordering,
    }
    gate, coloring = methods[method](gate)
    return gate, coloring
