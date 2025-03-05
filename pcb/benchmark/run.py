"""QAOA benchmarking on an actual IBM quantum computer."""

from pathlib import Path
from typing import Any

import filelock
import pandas as pd
from loguru import logger as logging
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from ..hamlib import hid_to_file_key, open_hamiltonian_file
from ..io import load, save
from ..qaoa import qaoa
from ..qiskit import reorder_operator, to_evolution_gate


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
        Same as `pcb.benchmark.simulate.benchmark`
    """
    from .simulate import benchmark as _benchmark

    return _benchmark(
        ham_dir=ham_dir,
        reorder_result=reorder_result,
        reorder_result_dir=reorder_result_dir,
        output_dir=output_dir,
        n_trials=n_trials,
        n_jobs=1,
        _bench_one_override=_bench_one,
    )
