"""Actual benchmark functions"""

import json
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Literal

import filelock
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit import qpy
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from tqdm import tqdm

from .hamlib import open_hamiltonian_file
from .qiskit import to_evolution_gate
from .reordering import reorder
from .reordering.utils import coloring_to_array
from .utils import hash_dict

MIN_N_TERMS = 10  # Hamiltonians with fewer terms are not benchmarked
ONE_MS = timedelta(milliseconds=1)


def _bench_one(
    ham_file: str | Path,
    result_file: str | Path,
    key: str,
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

    ham_file = Path(ham_file)
    result_file = Path(result_file)
    if result_file.is_file():
        return
    result_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = result_file.with_suffix(".lock")
    lock = filelock.FileLock(lock_file, blocking=False)

    try:
        with lock:
            with open_hamiltonian_file(ham_file) as fp:
                gate = to_evolution_gate(fp[key][()], shuffle=False)

            result: dict[str, Any] = {
                "method": method,
                "n_terms": len(gate.operator),
                "n_qubits": gate.num_qubits,
                "n_timesteps": n_timesteps,
                "order": order,
                "trotterization": trotterization,
                "hid": (
                    ham_file.name.split(".")[0].replace("__", "/") + "/" + key
                ),
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

            result.update({
                "depth": circuit.depth(),
                "reordering_time": reordering_time,
                "synthesis_time": synthesis_time,
            })

            with result_file.open("w", encoding="utf-8") as fp:
                json.dump(result, fp)
            with result_file.with_suffix(".qpy").open("wb") as fp:
                qpy.dump(circuit, fp)
            if method != "none":  # coloring_array is defined
                with h5py.File(result_file.with_suffix(".hdf5"), "w") as fp:
                    fp.create_dataset("coloring", data=coloring_array)
                    fp.create_dataset(
                        "term_indices", data=np.array(term_indices, dtype=int)
                    )

    except filelock.Timeout:
        pass

    else:
        lock.release()
        lock_file.unlink(missing_ok=True)


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
    index = index[index["n_terms"] >= MIN_N_TERMS]
    logging.debug(
        "Number of Hamiltonians after n_terms cutoff: {} (MIN_N_TERMS={})",
        len(index),
        MIN_N_TERMS,
    )
    index = index.sort_values("n_terms")

    jobs = []
    progress = tqdm(index.iterrows(), desc="Listing jobs", total=len(index))
    for _, row in progress:
        ham_path = ham_dir / (
            (row["dir"] + row["file"]).replace("/", "__") + ".hdf5.zip"
        )
        everything = product(
            ["suzuki_trotter"],
            methods,
            [4],  # order
            [1],  # n_timesteps
            range(n_trials),
        )
        for trotterization, method, order, n_timesteps, i in everything:
            kw = {
                "ham_file": ham_path,
                "key": row["key"],
                "trotterization": trotterization,
                "method": method,
                "order": order,
                "n_timesteps": n_timesteps,
            }
            jid = hash_dict({"kw": kw, "trial": i})  # unique job identifier
            result_file = (
                output_dir
                / "jobs"
                / jid[:2]  # spread files in subdirs
                / jid[2:4]
                / f"{jid}.json"
            )
            if result_file.is_file():
                continue
            kw["result_file"] = result_file
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


def consolidate(jobs_dir: str | Path) -> pd.DataFrame:
    """
    Gather all the output JSON files produced by `_bench_one` into a single
    dataframe
    """
    jobs_dir, rows = Path(jobs_dir), []
    progress = tqdm(
        jobs_dir.glob("**/*.json"), desc="Consolidating", leave=False
    )
    for file in progress:
        try:
            with open(file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["jid"] = file.stem
                rows.append(data)
        except json.JSONDecodeError as e:
            logging.error("Error reading {}: {}", file, e)
    results = pd.DataFrame(rows)
    results.set_index("hid", inplace=True)
    logging.info("Consolidated {} job results", len(results))
    return results
