"""QAOA benchmarking on an actual IBM quantum computer."""

from itertools import product
from pathlib import Path
from typing import Any, Literal

import filelock
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.fake_provider import FakeKawasaki
from tqdm import tqdm

from ..hamlib import hid_to_file_key, open_hamiltonian_file
from ..io import load, save
from ..qaoa import qaoa
from ..qiskit import reorder_operator, to_evolution_gate, trim_qc
from ..utils import hash_dict
from .consolidate import consolidate
from .utils import jid_to_json_path


def _bench_one(
    hid: str,
    ham_dir: str | Path,
    reorder_result: str | Path,
    output_file: str | Path,
    mode: Literal["aer", "ibmq"],
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
    reordering_jid = reorder_result.stem
    ham_file, key = hid_to_file_key(hid, ham_dir)

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
            cost_qc, operator = trim_qc(cost_qc, operator)

            _qaoa = _qaoa_aer if mode == "aer" else _qaoa_ibmq
            (best_x, best_e), (all_x, all_e), results = _qaoa(
                operator=operator,
                cost_qc=cost_qc,
                qaoa_config=qaoa_config,
            )

            for r in results:
                r["hid"] = hid
                r["reordering_jid"] = reordering_jid
            save(results, output_file)
            save(
                {
                    "all_energies": all_e,
                    "all_parameters": all_x,
                    "best_energy": np.array([best_e], dtype=float),
                    # ↑ hdf5py crashes for scalar entries (╯°□°)╯︵ ┻━┻
                    "best_parameters": best_x,
                },
                output_file.with_suffix(".hdf5"),
            )

    except filelock.Timeout:
        pass
    except Exception as e:
        logging.error(
            "Error while processing hid={}, rjid={}:\n{}",
            hid,
            reordering_jid,
            e,
        )


def _qaoa_aer(
    operator: SparsePauliOp,
    cost_qc: QuantumCircuit | None = None,
    qaoa_config: dict[str, Any] | None = None,
) -> tuple[
    tuple[np.ndarray, float], tuple[np.ndarray, np.ndarray], list[dict]
]:
    """Runs QAOA using a qiskit-aer simulator backend"""
    backend = FakeKawasaki()
    logging.debug("Using backend: {}", backend.name)
    estimator = AerEstimator(
        options={
            "backend_options": {
                "method": "tensor_network",
                "coupling_map": backend.coupling_map,
                "noise_model": NoiseModel.from_backend(backend),
                "device": "GPU",
                "cuStateVec_enable": True,
                "blocking_enable": True,
                "blocking_qubits": 23,
                # ↑ log2(smallest gpu mem / 256) - 4
            },
            "run_options": {"seed": 0, "shots": 1024},
        }
    )
    (best_x, best_e), (all_x, all_e), results = qaoa(
        operator=operator,
        backend=backend,
        estimator=estimator,
        cost_qc=cost_qc,
        **(qaoa_config or {}),
    )
    for r in results:
        r["backend"] = backend.name
    return (best_x, best_e), (all_x, all_e), results


def _qaoa_ibmq(
    operator: SparsePauliOp,
    cost_qc: QuantumCircuit | None = None,
    qaoa_config: dict[str, Any] | None = None,
) -> tuple[
    tuple[np.ndarray, float], tuple[np.ndarray, np.ndarray], list[dict]
]:
    """Runs QAOA using on an actual IBM quantum computer"""
    service = QiskitRuntimeService()
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=operator.num_qubits,
    )
    logging.debug("Using backend: {}", backend.name)
    with Session(backend=backend) as session:
        logging.debug("Opened session: {}", session.session_id)
        estimator = IBMEstimator(
            mode=session,
            options={
                "default_shots": 1024,
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
        **(qaoa_config or {}),
    )
    for r in results:
        r["backend"] = backend.name
    return (best_x, best_e), (all_x, all_e), results


def benchmark(
    ham_dir: str | Path,
    reorder_result: pd.DataFrame,
    reorder_result_dir: str | Path,
    output_dir: str | Path,
    n_trials: int = 1,
    n_jobs: int = 8,
    mode: Literal["aer", "ibmq"] = "aer",
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
        n_jobs (int, optional):
        mode (Literal["aer", "ibmq"], optional): Wether to use a qiskit-aer
            simulator or an actual IBM quantum computer
    """
    ham_dir, reorder_result_dir = Path(ham_dir), Path(reorder_result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Number of Hamiltonians: {}", len(reorder_result))

    jobs = []
    everything = list(
        product(
            reorder_result.iterrows(),
            [2],  # n_qaoa_steps
        )
    )
    progress = tqdm(everything, desc="Listing jobs", total=len(everything))
    for (_, row), n_qaoa_steps in progress:
        for i in range(n_trials):
            qaoa_config = {
                "n_qaoa_steps": n_qaoa_steps,
                # "n_shots": 1024,
                # "pm_optimization_level": 3,
                # "batch_size": 16,
                # "max_n_batches": 10,
            }
            jid = hash_dict(  # unique simulation job identifier
                {
                    "reordering_jid": row["jid"],
                    "hid": row["hid"],
                    "trial": i,
                    "qaoa_config": qaoa_config,
                }
            )
            output_file = jid_to_json_path(jid, output_dir)
            if output_file.is_file() and output_file.stat().st_size > 0:
                continue

            kw = {
                "hid": row["hid"],
                "ham_dir": ham_dir,
                "reorder_result": jid_to_json_path(
                    row["jid"], reorder_result_dir
                ),
                "output_file": output_file,
                "mode": mode,
                "qaoa_config": qaoa_config,
            }
            jobs.append(delayed(_bench_one)(**kw))
    logging.info("Submitting {} jobs", len(jobs))
    executor = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        timeout=3600 * 24 if n_jobs > 1 else None,  # 24h
        verbose=1,
        backend="loky",
    )
    executor(jobs)
    return consolidate(output_dir / "jobs")
