"""Actual benchmark functions"""

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from tqdm import tqdm

from .hamlib import open_hamiltonian_file
from .qiskit import to_evolution_gate
from .reordering import reorder
from .utils import cached, hash_dict


def _bench_one(
    path: str | Path,
    key: str,
    trotterization: Literal["lie_trotter", "suzuki_trotter"],
    coloring: Literal[
        "degree_c",
        "degree",
        "misra_gries",
        "none",
        "saturation",
        "simplicial",
    ],
) -> dict:
    """
    Compares Pauli coloring against direct Trotterization of the evolution
    operator of a given (serialized) Hamiltonian. The terms of underlying Pauli
    operator are shuffled before the comparison.

    The returned dict has the following columns:
    - `trotterization`: as passed,
    - `coloring`: as passed,
    - `n_terms`: number of terms in the underlying Pauli operator,
    - `depth`: the depth of the circuit obtained by Trotterization,
    - `time`: time taken by Trotterization (in milliseconds).

    All Trotterizations are done with `reps=1`.
    """
    with open_hamiltonian_file(path) as fp:
        hamiltonian: bytes = fp[key][()]
    gate = to_evolution_gate(hamiltonian, shuffle=True)
    Trotter = LieTrotter if trotterization == "lie_trotter" else SuzukiTrotter
    start = datetime.now()
    if coloring != "none":
        gate = reorder(gate, coloring)
    synthesizer = Trotter(reps=1, preserve_order=True)
    circuit = synthesizer.synthesize(gate)
    time = (datetime.now() - start).microseconds / 1000
    result = {
        "trotterization": trotterization,
        "coloring": coloring,
        "n_terms": len(gate.operator),
        "depth": circuit.depth(),
        "time": time,
    }
    return result


def benchmark(
    index: pd.DataFrame,
    ham_dir: str | Path,
    output_dir: str | Path,
    n_trials: int = 10,
    prefix: str | None = None,
    n_jobs: int = 32,
) -> pd.DataFrame:
    """
    Args:
        index (pd.DataFrame):
        ham_dir (Path): Directory containing the downloaded Hamiltonian
            `.hdf5.zip`files.
        output_dir (Path):
        n_trials (int, optional):
        prefix (str | None, optional): Filter the Hamiltonians to benchmark
    """
    ham_dir, output_dir = Path(ham_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefix:
        index = index[index["dir"].str.startswith(prefix)]
    jobs = []
    progress = tqdm(index.iterrows(), desc="Listing jobs", total=len(index))
    for _, row in progress:
        ham_path = ham_dir / (
            (row["dir"] + row["file"]).replace("/", "__") + ".hdf5.zip"
        )
        everything = product(
            [
                # "lie_trotter",
                "suzuki_trotter",
            ],
            [
                "none",
                "degree",
                # "degree_c",
                # "misra_gries",
                "saturation",
            ],
            range(n_trials),
        )
        for trotterization, coloring, i in everything:
            hid = row["dir"] + row["file"] + "/" + row["key"]
            jid = hash_dict(
                {
                    "hid": hid,
                    "trotterization": trotterization,
                    "coloring": coloring,
                    "trial": i,
                }
            )
            result_file_path = output_dir / "jobs" / f"{jid}.json"
            if result_file_path.is_file():
                continue
            f = cached(
                _bench_one,
                result_file_path,
                extra={"hid": hid},
            )
            kw = {
                "path": ham_path,
                "key": row["key"],
                "trotterization": trotterization,
                "coloring": coloring,
            }
            jobs.append(delayed(f)(**kw))
    logging.info("Submitting {} jobs", len(jobs))
    executor = Parallel(
        n_jobs=n_jobs, prefer="processes", timeout=3600 * 24, verbose=1
    )
    executor(jobs)
    return consolidate(output_dir / "jobs")


def consolidate(jobs_dir: str | Path) -> pd.DataFrame:
    """
    Gather all the output JSON files produced by `_bench_one` into a single
    dataframe
    """
    jobs_dir, rows = Path(jobs_dir), []
    progress = tqdm(jobs_dir.glob("*.json"), desc="Consolidating", leave=False)
    for file in progress:
        with open(file, "r", encoding="utf-8") as fp:
            rows.append(json.load(fp))
    results = pd.DataFrame(rows)
    logging.info("Consolidated {} job results", len(results))
    return results
