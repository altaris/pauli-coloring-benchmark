"""Actual benchmark functions"""

from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
from loguru import logger as logging
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .hamlib import open_hamiltonian
from .qiskit import to_evolution_gate


def _bench_gate(
    gate: PauliEvolutionGate,
    method: Literal["lie_trotter", "suzuki_trotter"],
) -> dict:
    """
    Compares Pauli coloring against direct Trotterization on a given evolution
    gate.

    The returned dict has the following columns:
    - `method`: Either `lie_trotter` or `suzuki_trotter`,
    - `n_terms`: Number of terms in the underlying Pauli operator,
    - `base_depth`: The depth of the circuit obtained by direct Trotterization,
    - `pc_depth`: The depth of the circuit obtained by Pauli coloring during
      Trotterization,
    - `base_time`: Time taken by direct Trotterization (in miliseconds),
    - `pc_time`: Time taken by using Pauli coloring during Trotterization (also
      in miliseconds).

    All Trotterizations are done with `reps=1`.
    """
    # logging.debug("Using method: {}", method)
    Trotter = LieTrotter if method == "lie_trotter" else SuzukiTrotter
    base_synth = Trotter(reps=1, preserve_order=True)
    pc_synth = Trotter(reps=1, preserve_order=False)
    start = datetime.now()
    base_circuit = base_synth.synthesize(gate)
    base_time = (start - datetime.now()).microseconds * 1000
    start = datetime.now()
    pc_circuit = pc_synth.synthesize(gate)
    pc_time = (start - datetime.now()).microseconds * 1000
    return {
        "method": method,
        "n_terms": len(gate.operator),
        "base_depth": base_circuit.depth(),
        "pc_depth": pc_circuit.depth(),
        "base_time": base_time,
        "pc_time": pc_time,
    }


def _bench_hamiltonian(hamiltonian: bytes, n_trials: int = 10) -> pd.DataFrame:
    """
    Benchmarks the Pauli coloring Trotterization against direct Trotterization
    one a given (serialized) Hamiltonian.

    The returned dataframe has the same colunms as the dicts returned by
    `_bench_gate`. There are `2 * n_trials` rows in the dataframe.
    """
    rows = []
    for i in range(n_trials):
        # logging.debug("Trial: {}/{}", i + 1, n_trials)
        gate = to_evolution_gate(hamiltonian, shuffle=True)
        rows.append(_bench_gate(gate, "lie_trotter"))
        rows.append(_bench_gate(gate, "suzuki_trotter"))
    return pd.DataFrame(rows)


def benchmark(
    index: pd.DataFrame, n_trials: int = 10, prefix: str = ""
) -> pd.DataFrame:
    """
    Runs the full benchmark using the fiven Hamiltonian index (see
    `hamlib.build_index`). `prefix` can be set to restrict the benchmark to a
    given family of Hamiltonians, for example `binaryoptimization/maxcut`.

    The returned dataframe has the same columns as the dicts returned by
    `_bench_gate` plus an additional Hamiltonian ID (`hid`) column.
    """
    if prefix:
        index = index[index["hfid"].str.startswith(prefix)]
    logging.info("Starting benchmark with {} Hamiltonians", len(index))
    start = datetime.now()
    results = []
    for _, row in index.iterrows():
        with TemporaryDirectory() as tmp:
            with open_hamiltonian(row["url"], output_dir=tmp) as fp:
                for i, k in enumerate(fp.keys()):
                    logging.debug(
                        "Hamiltonian {}/{}: {}",
                        i + 1,
                        len(fp.keys()),
                        k,
                    )
                    df = _bench_hamiltonian(fp[k][()], n_trials)
                    df["hid"] = row["hfid"] + "/" + k
                    results.append(df)
    logging.info("Completed benchmark in: {}", datetime.now() - start)
    return pd.concat(results, ignore_index=True)
