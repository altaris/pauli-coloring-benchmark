"""QAOA benchmarking on an actual IBM quantum computer."""

from itertools import product
from pathlib import Path
from typing import Any

import filelock
import pandas as pd
from loguru import logger as logging
from qiskit.providers import BackendV2 as Backend
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

from ..hamlib import open_hamiltonian_file
from ..qiskit import to_evolution_gate
from .consolidate import consolidate
from .qaoa import qaoa
from .utils import (
    hash_dict,
    hid_to_file_key,
    jid_to_json_path,
    load,
    reorder_operator,
    save,
)


def _bench_one(
    ham_file: str | Path,
    key: str,
    order_file: str | Path | None,
    circuit_file: str | Path,
    backend: Backend,
    output_file: str | Path,
    qaoa_config: dict[str, Any],
) -> None:
    """
    Args:
        ham_file (str | Path): The `.hdf5.zip` file containing the Hamiltonian.
        key (str): The key inside the `ham_file` to the Hamiltonian in question.
        order_file (str | Path | None): The `.hdf5` file containing the color
            and term indices vectors. See also
            `pcb.benchmark.reorder._bench_one`.
        circuit_file (str | Path): The `.qpy.gz` file containing the Trotterized
            circuit of the operator. See also
            `pcb.benchmark.reorder._bench_one`.
        output_file (str | Path): JSON file where the result of this job will be
            written to.
        qaoa_config (dict[str, Any]): Extra parameter to pass to
            `pcb.benchmark.run.qaoa`.
    """
    output_file = Path(output_file)

    if output_file.is_file() and output_file.stat().st_size > 0:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hid = Path(ham_file).name.split(".")[0].replace("__", "/") + "/" + key
    lock_file = output_file.with_suffix(".lock")
    lock = filelock.FileLock(lock_file, blocking=False)

    try:
        with lock:
            with open_hamiltonian_file(ham_file) as fp:
                gate = to_evolution_gate(fp[key][()], shuffle=False)
                operator = gate.operator
                assert isinstance(operator, SparsePauliOp)
            if order_file is not None:
                term_indices = load(order_file)["term_indices"].astype(int)
                operator = reorder_operator(operator, term_indices)
            cost_qc = load(circuit_file)
            _, (all_x, all_e), results = qaoa(
                operator, cost_qc, backend, **qaoa_config
            )
            for r in results:
                r.update({"hid": hid})
            save(results, output_file)
            save(
                {"parameters": all_x, "energy": all_e},
                output_file.with_suffix(".hdf5"),
            )

    except filelock.Timeout:
        pass


def benchmark(
    ham_dir: str | Path,
    reorder_result: pd.DataFrame,
    reorder_result_dir: str | Path,
    output_dir: str | Path,
    backend: Backend,
    n_trials: int = 1,
) -> pd.DataFrame:
    """
    Args:
        ham_dir (str | Path):
        reorder_result (pd.DataFrame): Dataframe containing at least the columns
            `hid`, `trotterization`, `n_timesteps`, `order`, `method`, `jid`.
            You might want to make sure it only has one row per
            `hid`-`trotterization`-`n_timesteps`-`order`-`method` tuple =)
        reorder_result_dir (str | Path):
        output_dir (str | Path):
        backend (IBMBackend):
        n_trials (int, optional):
    """
    ham_dir, reorder_result_dir = Path(ham_dir), Path(reorder_result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Number of Hamiltonians: {}", len(reorder_result))

    everything = list(
        product(
            reorder_result.iterrows(),
            range(n_trials),
            [1, 2, 4],  # n_qaoa_steps
            [3],  # preset manager optimization_level
        )
    )
    for (_, row), i, n_qaoa_steps, pm_opt_lvl in tqdm(everything):
        for i in range(n_trials):
            qaoa_config = {
                "n_qaoa_steps": n_qaoa_steps,
                "n_shots": 1024,
                "pm_optimization_level": pm_opt_lvl,
                "max_iter": 1000,
            }
            jid = hash_dict(  # unique simulation job identifier
                {"row": dict(row), "trial": i, "qaoa_config": qaoa_config}
            )
            output_file = jid_to_json_path(jid, output_dir)
            if output_file.is_file() and output_file.stat().st_size > 0:
                continue
            ham_file, key = hid_to_file_key(row["hid"], ham_dir)
            # ↓ files from the reording jobs are named like this (modulo ext.)
            p = jid_to_json_path(row["jid"], reorder_result_dir)

            _bench_one(
                ham_file=ham_file,
                key=key,
                order_file=(
                    p.with_suffix(".hdf5") if row["method"] != "none" else None
                ),
                circuit_file=p.with_suffix(".qpy.gz"),
                backend=backend,
                output_file=output_file,
                qaoa_config=qaoa_config,
            )

    return consolidate(output_dir / "jobs")
