"""
Gate reordering method based on the degree coloring (C implementation) of the
conflict graph of a Hamiltonian
"""

from pathlib import Path

import numpy as np
from qiskit.circuit.library import PauliEvolutionGate

from .utils import (
    Coloring,
    gate_to_c,
    reorder_gate_by_colors,
)


def degree_reordering_c(
    gate: PauliEvolutionGate,
) -> tuple[PauliEvolutionGate, Coloring, np.ndarray]:
    """
    C implementation of the degree method of
    `pcb.reordering.degree.degree_reordering`.
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
    qb_idx, trm_start_idx, n_qb_trm, n_trm = gate_to_c(gate)
    trm_col = (ctypes.c_int * n_trm)()
    _degree_coloring(
        ctypes.cast(qb_idx, ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(trm_start_idx, ctypes.POINTER(ctypes.c_size_t)),
        ctypes.cast(n_qb_trm, ctypes.POINTER(ctypes.c_size_t)),
        n_trm,
        gate.num_qubits,
        ctypes.cast(trm_col, ctypes.POINTER(ctypes.c_int)),
    )
    return reorder_gate_by_colors(gate, dict(enumerate(trm_col)))
