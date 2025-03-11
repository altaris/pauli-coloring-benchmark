"""
Converts Ising Hamiltonians (as downloaded from HamLib) to lists of pairs of
interacting qubits.

For $Z_1 Z_4 + Z_2 Z_3$ will be translated to `[[1, 4], [2, 3]]`.
"""

from pathlib import Path

import numpy as np
from loguru import logger as logging
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

from pcb import hid_to_file_key, open_hamiltonian_file, save, to_evolution_gate
from pcb.__main__ import _open_index
from pcb.reordering.utils import is_ising
from pcb.utils import timed

HAM_PATH = Path("out/ham")
PREFIX = "binaryoptimization/maxcut"
INDEX_PATH = Path("out/index.db")
OUT_PATH = Path("out/ising")


def _remove_idle_qubits(op: SparsePauliOp) -> SparsePauliOp:
    """Self-explanatory"""
    strings, weights = zip(*op.to_list())
    if len(strings) == 0:
        raise ValueError("Empty operator")
    if len(strings) == 1:
        return SparsePauliOp.from_list([("ZZ", weights[0])])
    non_idle = []
    for q in range(op.num_qubits):
        if not all(p[q] == "I" for p in strings):
            non_idle.append(q)
    strings = ["".join(p[q] for q in non_idle) for p in strings]
    if len(strings) == 0:
        raise ValueError("All qubits are idle")
    op = SparsePauliOp.from_list(zip(strings, weights))
    assert is_ising(op)
    return op


def _interaction_array(operator: SparsePauliOp) -> np.ndarray:
    """
    Transforms an Ising Hamiltonian to a `(M, 2)` array of interacting qubit
    pairs, where `M` is the number of **non-idle** qubits.
    """
    operator = _remove_idle_qubits(operator)
    return np.array([s for _, s, _ in operator.to_sparse_list()])


@timed
def main() -> None:
    """gabagool"""
    index = _open_index(INDEX_PATH, prefix=PREFIX)
    index["path"] = [
        hid_to_file_key(hid, "out/ham")[0] for hid in index["hid"]
    ]
    p1 = tqdm(index["path"].unique())
    p, _ = hid_to_file_key(
        "binaryoptimization/maxcut/random/ham-graph-star/graph-n-10", "out/ham"
    )
    p1 = tqdm([p])
    for p in p1:
        p1.set_postfix({"file": p.name})
        data: dict[str, np.ndarray] = {}
        with open_hamiltonian_file(p) as fp:
            p2 = tqdm(fp.items(), leave=False)
            for k, v in p2:
                p2.set_postfix({"key": k})
                gate = to_evolution_gate(v[()])
                assert isinstance(gate.operator, SparsePauliOp)
                try:
                    assert is_ising(gate.operator)
                    data[k] = _interaction_array(gate.operator)
                except Exception as e:
                    logging.error("Failed to convert {} {}: {}", p, k, e)
                    logging.error("Operator: {}", gate.operator)
        fn = p.name.split(".")[0] + ".hdf5"
        save(data, OUT_PATH / fn)


if __name__ == "__main__":
    main()
