"""Hamiltonian reordering benchmarking"""

from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Literal

import filelock
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from tqdm import tqdm

from ..hamlib import open_hamiltonian_file
from ..qiskit import to_evolution_gate
from ..reordering import reorder
from ..reordering.utils import coloring_to_array
from .consolidate import consolidate
from .utils import hash_dict, hid_to_file_key, jid_to_json_path, save

ONE_MS = timedelta(milliseconds=1)


def _bench_one(
    hid: str,
    ham_dir: str | Path,
    output_file: str | Path,
    trotterization: Literal["lie_trotter", "suzuki_trotter"],
    method: Literal[
        "degree_c",
        "degree",
        "misra_gries",
        "none",
        "saturation",
        "simplicial",
    ],
    order: int = 1,
    n_timesteps: int = 1,
) -> None:
    """
    Compares Pauli coloring against direct Trotterization of the evolution
    operator of a given (serialized) Hamiltonian. The terms of underlying Pauli
    operator are shuffled before the comparison.

    The output JSON file contains the following keys
    - `trotterization`: as passed.
    - `method`: as passed.
    - `n_terms`: number of terms in the underlying Pauli operator (this is of
      course the same before and after reordering).
    - `n_qubits`: number of qubits in the gate.
    - `depth`: the depth of the circuit obtained by Trotterization.
    - `order`: order of the Trotterization. Not relevant if `trotterization` is
      `lie_trotter`.
    - `n_timesteps`: called `reps` in Qiskit.
    - `reordering_time`: in milliseconds.
    - `synthesis_time`: in milliseconds.
    - `hid`: Hamiltonian id, which is the concatenation of the directory, file
      name, and key in the HDF5 file. Example: `binaryoptimization/maxcut/random/ham-graph-complete_bipart/complbipart-n-100_a-50_b-50`.

    If `method` is not `none`, the output HDF5 file contains a dataset named
    `coloring` containing the coloring vector and the index vector under the
    keys `coloring` and `term_indices` respectively. This file is created in the
    same directory as the JSON file, and has the same name but with a `.hdf5`
    extension.
    """

    output_file = Path(output_file)
    if output_file.is_file() and output_file.stat().st_size > 0:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ham_file, key = hid_to_file_key(hid, ham_dir)

    try:
        lock_file = output_file.with_suffix(".lock")
        with filelock.FileLock(lock_file, blocking=False):
            with open_hamiltonian_file(ham_file) as fp:
                gate = to_evolution_gate(fp[key][()], shuffle=False)

            result: dict[str, Any] = {
                "method": method,
                "n_terms": len(gate.operator),
                "n_qubits": gate.num_qubits,
                "n_timesteps": n_timesteps,
                "order": order,
                "trotterization": trotterization,
                "hid": hid,
            }

            reordering_time = 0.0
            if method != "none":
                start = datetime.now()
                gate, coloring, term_indices = reorder(gate, method)
                reordering_time = (datetime.now() - start) / ONE_MS
                coloring_array = coloring_to_array(coloring)

            start = datetime.now()
            if trotterization == "lie_trotter":
                synthesizer = LieTrotter(reps=n_timesteps, preserve_order=True)
            else:  # trotterization == "suzuki_trotter"
                synthesizer = SuzukiTrotter(
                    reps=n_timesteps, order=order, preserve_order=True
                )
            circuit = synthesizer.synthesize(gate)
            synthesis_time = (datetime.now() - start) / ONE_MS

            result.update(
                {
                    "depth": circuit.depth(),
                    "reordering_time": reordering_time,
                    "synthesis_time": synthesis_time,
                }
            )

            save(result, output_file)
            save(circuit, output_file.with_suffix(".qpy.gz"))
            if method != "none":  # coloring_array is defined
                save(
                    {"coloring": coloring_array, "term_indices": term_indices},
                    output_file.with_suffix(".hdf5"),
                )

    except filelock.Timeout:
        pass


def benchmark(
    index: pd.DataFrame,
    ham_dir: str | Path,
    output_dir: str | Path,
    n_trials: int = 10,
    n_jobs: int = 32,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Args:
        index (pd.DataFrame):
        ham_dir (Path): Directory containing the downloaded Hamiltonian
            `.hdf5.zip`files.
        output_dir (Path):
        n_trials (int, optional):
        methods (list[str] | None, optional): Reordering methods to benchmark.
            If left as `None`, defaults to `["degree", "saturation", "none"]`.
    """
    methods = methods or ["degree", "saturation", "none"]
    ham_dir, output_dir = Path(ham_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Number of Hamiltonians: {}", len(index))
    index = index.sort_values("n_terms")

    jobs = []
    progress = tqdm(index.iterrows(), desc="Listing jobs", total=len(index))
    for _, row in progress:
        everything = product(
            ["suzuki_trotter"],
            methods,
            [4],  # order
            [1],  # n_timesteps
            range(n_trials),
        )
        for trotterization, method, order, n_timesteps, i in everything:
            kw = {
                "hid": row["hid"],
                "ham_dir": ham_dir,
                "trotterization": trotterization,
                "method": method,
                "order": order,
                "n_timesteps": n_timesteps,
            }
            jid = hash_dict({"kw": kw, "trial": i})  # unique job identifier
            output_file = jid_to_json_path(jid, output_dir)
            if output_file.is_file() and output_file.stat().st_size > 0:
                continue
            kw["output_file"] = output_file
            jobs.append(delayed(_bench_one)(**kw))
    logging.info("Submitting {} jobs", len(jobs))
    executor = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        timeout=3600 * 24,  # 24h
        verbose=1,
        backend="loky",
    )
    executor(jobs)
    return consolidate(output_dir / "jobs")
