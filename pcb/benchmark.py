"""Actual benchmark functions"""

import json
from datetime import datetime
from pathlib import Path
from typing import Generator, Literal

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from tqdm import tqdm

from .hamlib import open_hamiltonian
from .qiskit import to_evolution_gate
from .utils import hash_dict


def _bench_one(
    hamiltonian: bytes,
    method: Literal["lie_trotter", "suzuki_trotter"],
    output_file: Path,
) -> dict:
    """
    Compares Pauli coloring against direct Trotterization of the evolution
    operator of a given (serialized) Hamiltonian. The terms of underlying Pauli
    operator are shuffled before the comparison.

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
    gate = to_evolution_gate(hamiltonian, shuffle=True)
    Trotter = LieTrotter if method == "lie_trotter" else SuzukiTrotter
    base_synth = Trotter(reps=1, preserve_order=True)
    pc_synth = Trotter(reps=1, preserve_order=False)
    start = datetime.now()
    base_circuit = base_synth.synthesize(gate)
    base_time = (start - datetime.now()).microseconds * 1000
    start = datetime.now()
    pc_circuit = pc_synth.synthesize(gate)
    pc_time = (start - datetime.now()).microseconds * 1000
    result = {
        "method": method,
        "n_terms": len(gate.operator),
        "base_depth": base_circuit.depth(),
        "pc_depth": pc_circuit.depth(),
        "base_time": base_time,
        "pc_time": pc_time,
    }
    with output_file.open("w", encoding="utf-8") as fp:
        json.dump(result, fp)
    return result


def _list_jobs(
    path: Path,
    hfid: str,
    output_dir: Path,
    n_trials: int = 10,
) -> list[dict]:
    """

    Returns:
        list[dict]: _description_
    """
    jobs = []
    with open_hamiltonian(path) as fp:
        for k in fp.keys():
            hid = hfid + "/" + k
            for method in ["lie_trotter", "suzuki_trotter"]:
                for i in range(n_trials):
                    jid = hash_dict({"hid": hid, "method": method, "trial": i})
                    output_file = output_dir / f"{jid}.json"
                    jobs.append(
                        {
                            "hamiltonian": fp[k][()],
                            "method": method,
                            "output_file": output_file,
                        }
                    )
    return jobs


def benchmark(
    index: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    n_trials: int = 10,
    prefix: str | None = None,
    n_jobs: int = 32,
) -> pd.DataFrame:
    """
    Args:
        index (pd.DataFrame):
        input_dir (Path): Directory containing the downloaded Hamiltonian files.
        output_dir (Path):
        n_trials (int, optional):
        prefix (str | None, optional): Filter the Hamiltonians to benchmark
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefix:
        index = index[index["hfid"].str.startswith(prefix)]
    executor = Parallel(n_jobs=n_jobs, verbose=1)
    jobs = [
        delayed(_list_jobs)(
            path=input_dir / (row["hfid"].replace("/", "__") + ".hdf5.zip"),
            hfid=row["hfid"],
            output_dir=output_dir,
            n_trials=n_trials,
        )
        for _, row in tqdm(
            index.iterrows(),
            desc="Iterating through the index",
            total=len(index),
        )
    ]
    logging.info("Building job list")
    results: Generator[list[dict], None, None] = executor(jobs)
    jobs = []
    for batch in results:
        jobs.extend(delayed(_bench_one)(kw) for kw in batch)
    logging.info("Submitting {} jobs", len(jobs))
    executor(jobs)
    return consolidate(output_dir)


def consolidate(output_dir: Path) -> pd.DataFrame:
    """
    Gather all the output JSON files produced by `_bench_one` into a single
    dataframe
    """
    logging.info("Consolidating results from {}", output_dir)
    rows = []
    for file in tqdm(output_dir.glob("*.json"), desc="Consolidating"):
        with open(file, "r", encoding="utf-8") as fp:
            rows.append(json.load(fp))
    results = pd.DataFrame(rows)
    logging.info("Consolidated {} job results", len(results))
    return results
