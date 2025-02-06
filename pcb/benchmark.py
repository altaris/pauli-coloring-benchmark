"""Actual benchmark functions"""

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from tqdm import tqdm

from .hamlib import open_hamiltonian_file
from .qiskit import to_evolution_gate
from .reordering import reorder
from .reordering.utils import coloring_to_array
from .utils import cached, hash_dict


def _bench_one(
    path: str | Path,
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
) -> dict:
    """
    Compares Pauli coloring against direct Trotterization of the evolution
    operator of a given (serialized) Hamiltonian. The terms of underlying Pauli
    operator are shuffled before the comparison.

    The returned dict has the following columns:
    - `trotterization`: as passed,
    - `method`: as passed,
    - `n_terms`: number of terms in the underlying Pauli operator (this is of
      course the same before and after reordering),
    - `depth`: the depth of the circuit obtained by Trotterization,
    - `order`: order of the Trotterization, ignored if `trotterization` is
      `lie_trotter`,
    - `n_timesteps`: called `reps` in Qiskit,
    - `reordering_time`: in milliseconds,
    - `synthesis_time`: in milliseconds.

    All Trotterizations are done with `reps=1`.
    """
    shuffle = method != "none"
    with open_hamiltonian_file(path) as fp:
        gate = to_evolution_gate(fp[key][()], shuffle=shuffle)

    result: dict[str, Any] = {
        "method": method,
        "n_terms": len(gate.operator),
        "n_timesteps": n_timesteps,
        "order": order,
        "trotterization": trotterization,
    }

    reordering_time = 0.0
    if method != "none":
        start = datetime.now()
        gate, coloring = reorder(gate, method)
        reordering_time = (datetime.now() - start).microseconds / 1000
        result["coloring"] = coloring_to_array(coloring)

    start = datetime.now()
    if trotterization == "lie_trotter":
        synthesizer = LieTrotter(reps=n_timesteps, preserve_order=True)
    else:  # trotterization == "suzuki_trotter"
        synthesizer = SuzukiTrotter(
            reps=n_timesteps, order=order, preserve_order=True
        )
    circuit = synthesizer.synthesize(gate)
    synthesis_time = (datetime.now() - start).microseconds / 1000

    result.update(
        {
            "depth": circuit.depth(),
            "reordering_time": reordering_time,
            "synthesis_time": synthesis_time,
        }
    )
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
                # "degree",
                # "degree_c",
                # "misra_gries",
                "saturation",
                "simplicial",
            ],
            [2, 4],  # order
            [1],  # n_timesteps
            range(n_trials),
        )
        for trotterization, method, order, n_timesteps, i in everything:
            hid = row["dir"] + row["file"] + "/" + row["key"]
            jid = hash_dict(
                {
                    "hid": hid,
                    "trotterization": trotterization,
                    "method": method,
                    "trial": i,
                }
            )
            result_file_path = (
                output_dir
                / "jobs"
                / jid[:2]  # spread files in subdirs
                / jid[2:4]
                / f"{jid}.json"
            )
            if result_file_path.is_file():
                continue
            f = cached(
                _bench_one,
                result_file_path,
                extra={"hid": hid, "jid": jid},
            )
            kw = {
                "key": row["key"],
                "method": method,
                "n_timesteps": n_timesteps,
                "order": order,
                "path": ham_path,
                "trotterization": trotterization,
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
    progress = tqdm(
        jobs_dir.glob("**/*.json"), desc="Consolidating", leave=False
    )
    for file in progress:
        with open(file, "r", encoding="utf-8") as fp:
            rows.append(json.load(fp))
    results = pd.DataFrame(rows)
    results.set_index("hid", inplace=True)
    logging.info("Consolidated {} job results", len(results))
    return results
