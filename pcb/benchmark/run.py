"""QAOA benchmarking on an actual IBM quantum computer."""

from itertools import product
from pathlib import Path
from typing import Any

import filelock
import pandas as pd
from loguru import logger as logging
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session

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
    hid: str,
    ham_dir: str | Path,
    reorder_result: str | Path,
    output_file: str | Path,
    qaoa_config: dict[str, Any],
) -> None:
    """
    Args:
        hid (str): Hamiltonian ID
        ham_dir (str | Path): Directory containing the Hamiltonian files.
        rjid (str): Reordering job ID
        reorder_result (str | Path): Path to the file containing the reordering
            results, e.g. `out/reorder/jobs/.../<some_jid>.json`
        output_file (str | Path): JSON file where the result of this job will be
            written to.
        qaoa_config (dict[str, Any]): Extra parameter to pass to
            `pcb.benchmark.run.qaoa`.
    """
    output_file = Path(output_file)
    if output_file.is_file() and output_file.stat().st_size > 0:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reorder_result = Path(reorder_result)
    ham_file, key = hid_to_file_key(hid, ham_dir)
    rjid = reorder_result.name

    try:
        lock_file = output_file.with_suffix(".lock")
        with filelock.FileLock(lock_file, blocking=False):
            with open_hamiltonian_file(ham_file) as fp:
                gate = to_evolution_gate(fp[key][()], shuffle=False)
                operator = gate.operator
                assert isinstance(operator, SparsePauliOp)

            order_file = reorder_result.with_suffix(".hdf5")
            if order_file.is_file():
                term_indices = load(order_file)["term_indices"].astype(int)
                operator = reorder_operator(operator, term_indices)
            cost_qc = load(reorder_result.with_suffix(".qpy.gz"))

            service = QiskitRuntimeService()
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=operator.num_qubits,
            )
            logging.debug("Using backend: {}", backend.name)
            with Session(backend=backend) as session:
                logging.debug("Opened session: {}", session.session_id)
                estimator = Estimator(
                    mode=session,
                    options={
                        "default_shots": qaoa_config.get("n_shots", 1024),
                        "dynamical_decoupling": {"enable": False},
                        # "resilience_level": 3,
                        "twirling": {
                            "enable_gates": True,
                            "num_randomizations": "auto",
                        },
                    },
                )
                (best_x, best_e), (all_x, all_e), results = qaoa(
                    operator=operator,
                    backend=backend,
                    estimator=estimator,
                    cost_qc=cost_qc,
                    n_qaoa_steps=qaoa_config.get("n_qaoa_steps", 1),
                    pm_optimization_level=qaoa_config.get(
                        "pm_optimization_level", 0
                    ),
                    max_iter=qaoa_config.get("max_iter", 1000),
                )

            for r in results:
                r.update({"hid": hid, "reordering_jid": rjid})
            save(results, output_file)
            save(
                {
                    "all_energies": all_e,
                    "all_parameters": all_x,
                    "best_energy": best_e,
                    "best_parameters": best_x,
                },
                output_file.with_suffix(".hdf5"),
            )

    except filelock.Timeout:
        pass
    except Exception as e:
        logging.error(
            "Error while processing hid={}, rjid={}:\n{}", hid, rjid, e
        )


def benchmark(
    ham_dir: str | Path,
    reorder_result: pd.DataFrame,
    reorder_result_dir: str | Path,
    output_dir: str | Path,
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
        n_trials (int, optional):
    """
    ham_dir, reorder_result_dir = Path(ham_dir), Path(reorder_result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Number of Hamiltonians: {}", len(reorder_result))

    everything = product(
        reorder_result.iterrows(),
        range(n_trials),
        [2],  # n_qaoa_steps
        [3],  # preset manager optimization_level
    )
    for (_, row), i, n_qaoa_steps, pm_opt_lvl in everything:
        for i in range(n_trials):
            qaoa_config = {
                "n_qaoa_steps": n_qaoa_steps,
                "n_shots": 1024,
                "pm_optimization_level": pm_opt_lvl,
                "max_iter": 128,
            }
            jid = hash_dict(  # unique simulation job identifier
                {"row": dict(row), "trial": i, "qaoa_config": qaoa_config}
            )
            output_file = jid_to_json_path(jid, output_dir)
            if output_file.is_file() and output_file.stat().st_size > 0:
                continue

            _bench_one(
                hid=row["hid"],
                ham_dir=ham_dir,
                reorder_result=jid_to_json_path(
                    row["jid"], reorder_result_dir
                ),
                output_file=output_file,
                qaoa_config=qaoa_config,
            )
    return consolidate(output_dir / "jobs")
