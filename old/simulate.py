"""QAOA benchmarking"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import filelock
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_algorithms import VQE, VQEResult
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import fake_provider
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from tqdm import tqdm

from ..hamlib import open_hamiltonian_file
from ..qiskit import to_evolution_gate
from .consolidate import consolidate
from .utils import (
    hash_dict,
    hid_to_file_key,
    jid_to_json_path,
    load,
    reorder_operator,
    save,
)

FAKE_PROVIDERS: dict[str, type[FakeBackendV2]] = {
    "algiers": fake_provider.FakeAlgiers,
    "almaden_v2": fake_provider.FakeAlmadenV2,
    "armonk_v2": fake_provider.FakeArmonkV2,
    "athens_v2": fake_provider.FakeAthensV2,
    "auckland": fake_provider.FakeAuckland,
    "belem_v2": fake_provider.FakeBelemV2,
    "boeblingen_v2": fake_provider.FakeBoeblingenV2,
    "bogota_v2": fake_provider.FakeBogotaV2,
    "brisbane": fake_provider.FakeBrisbane,
    "brooklyn_v2": fake_provider.FakeBrooklynV2,
    "burlington_v2": fake_provider.FakeBurlingtonV2,
    "cairo_v2": fake_provider.FakeCairoV2,
    "cambridge_v2": fake_provider.FakeCambridgeV2,
    "casablanca_v2": fake_provider.FakeCasablancaV2,
    "cusco": fake_provider.FakeCusco,
    "essex_v2": fake_provider.FakeEssexV2,
    "fez": fake_provider.FakeFez,
    "geneva": fake_provider.FakeGeneva,
    "guadalupe_v2": fake_provider.FakeGuadalupeV2,
    "hanoi_v2": fake_provider.FakeHanoiV2,
    "jakarta_v2": fake_provider.FakeJakartaV2,
    "johannesburg_v2": fake_provider.FakeJohannesburgV2,
    "kawasaki": fake_provider.FakeKawasaki,
    "kolkata_v2": fake_provider.FakeKolkataV2,
    "kyiv": fake_provider.FakeKyiv,
    "kyoto": fake_provider.FakeKyoto,
    "lagos_v2": fake_provider.FakeLagosV2,
    "lima_v2": fake_provider.FakeLimaV2,
    "fractionalbackend": fake_provider.FakeFractionalBackend,
    "london_v2": fake_provider.FakeLondonV2,
    "manhattan_v2": fake_provider.FakeManhattanV2,
    "manila_v2": fake_provider.FakeManilaV2,
    "marrakesh": fake_provider.FakeMarrakesh,
    "melbourne_v2": fake_provider.FakeMelbourneV2,
    "montreal_v2": fake_provider.FakeMontrealV2,
    "mumbai_v2": fake_provider.FakeMumbaiV2,
    "nairobi_v2": fake_provider.FakeNairobiV2,
    "osaka": fake_provider.FakeOsaka,
    "oslo": fake_provider.FakeOslo,
    "ourense_v2": fake_provider.FakeOurenseV2,
    "paris_v2": fake_provider.FakeParisV2,
    "peekskill": fake_provider.FakePeekskill,
    "perth": fake_provider.FakePerth,
    "prague": fake_provider.FakePrague,
    "poughkeepsie_v2": fake_provider.FakePoughkeepsieV2,
    "quebec": fake_provider.FakeQuebec,
    "quito_v2": fake_provider.FakeQuitoV2,
    "rochester_v2": fake_provider.FakeRochesterV2,
    "rome_v2": fake_provider.FakeRomeV2,
    "santiago_v2": fake_provider.FakeSantiagoV2,
    "sherbrooke": fake_provider.FakeSherbrooke,
    "singapore_v2": fake_provider.FakeSingaporeV2,
    "sydney_v2": fake_provider.FakeSydneyV2,
    "torino": fake_provider.FakeTorino,
    "toronto_v2": fake_provider.FakeTorontoV2,
    "valencia_v2": fake_provider.FakeValenciaV2,
    "vigo_v2": fake_provider.FakeVigoV2,
    "washington_v2": fake_provider.FakeWashingtonV2,
    "yorktown_v2": fake_provider.FakeYorktownV2,
}


def _bench_one(
    ham_file: str | Path,
    key: str,
    order_file: str | Path,
    circuit_file: str | Path,
    output_file: str | Path,
    qaoa_config: dict[str, Any],
) -> None:
    """
    Args:
        ham_file (str | Path):
        key (str):
        order_file (str | Path): The `hdf5` file containing the color and term
            indices vectors. See `pcb.benchmark.reorder._bench_one`.
        circuit_file (str | Path): The `qpy` file containing the Trotterized
            circuit of the operator. See `pcb.benchmark.reorder._bench_one`.
        output_file (str | Path): JSON file where the result of this job will be
            written to.
        qaoa_config (dict[str, Any]): See `pcb.benchmark.simulate.benchmark`.
    """
    ham_file, order_file = Path(ham_file), Path(order_file)
    circuit_file, output_file = Path(circuit_file), Path(output_file)

    if output_file.is_file() and output_file.stat().st_size > 0:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
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

            cost_operator = load(circuit_file)

            estimator = _make_estimator(
                provider=qaoa_config["provider"],
                n_shots=qaoa_config["n_shots"],
                seed=qaoa_config["seed"],
            )

            results = _qaoa(
                estimator,
                operator,
                cost_operator,
                n_steps=qaoa_config["n_steps"],
                max_iter=qaoa_config["max_iter"],
            )
            save({"qaoa_config": qaoa_config, "results": results}, output_file)

    except filelock.Timeout:
        pass


def _make_estimator(
    provider: str = "fez", n_shots: int = 1024, seed: int = 0
) -> Estimator:
    if provider not in FAKE_PROVIDERS:
        available = ", ".join(FAKE_PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider}. Available: {available}"
        )
    backend = FAKE_PROVIDERS[provider]()
    noise_model = NoiseModel.from_backend(backend)
    return Estimator(
        backend_options={
            "method": "statevector",
            "coupling_map": backend.coupling_map,
            "noise_model": noise_model,
            "device": "GPU",
            "cuStateVec_enable": True,
            "blocking_enable": True,
            "blocking_qubits": 23,  # log2(smallest gpu mem / 256) - 4
        },
        run_options={"seed": seed, "shots": n_shots},
        transpile_options={"seed_transpiler": seed},
    )


def _qaoa(
    estimator: Estimator,
    operator: SparsePauliOp,
    cost_operator: QuantumCircuit,
    n_steps: int = 1,
    max_iter: int = 128,
) -> dict:
    """
    Args:
        estimator (Estimator):
        operator (SparsePauliOp): The Hamiltonian whose energy to minimize
        cost_operator (QuantumCircuit): The Trotterized `operator`
        n_steps (int, optional):
        max_iter (int, optional):

    Returns:
        A dictionary with the following keys:
        - `steps`: a list of data fed to the callback, which are themselves
          dicts containing the following keys:
            - `eval_count` (`int`)
            - `parameters` (`np.ndarray`)
            - `mean` (`float`)
            - `std` (`dict[str, Any]`)
        - `result`: a dict representing a `VQEResult`, see `_vqe_result_to_dict`
        - `time_ms`: the VQE execution time in ms
    """

    def _callback(
        eval_count: int,
        parameters: np.ndarray,
        mean: float,
        std: dict[str, Any],
    ) -> None:
        steps.append(
            {
                "eval_count": eval_count,
                "parameters": parameters,
                "mean": mean,
                "std": std,
            }
        )

    steps: list[dict] = []
    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=n_steps)
    optimizer = SPSA(maxiter=max_iter)
    vqe = VQE(estimator, ansatz, optimizer, callback=_callback)
    start = datetime.now()
    result = vqe.compute_minimum_eigenvalue(operator)
    time_ms = (datetime.now() - start) / timedelta(milliseconds=1)
    return {
        "steps": steps,
        "result": _vqe_result_to_dict(result),
        "time_ms": time_ms,
    }


def _vqe_result_to_dict(r: VQEResult) -> dict:
    """
    Example:

        {
            "cost_function_evals": 4,
            "eigenvalue": 0.390625,
            "optimal_parameters": {
                "β[0]": 4.439977406523218,
                "β[1]": 5.501340621929524,
                "γ[0]": 3.7889538113820396,
                "γ[1]": -2.17851412244563
            },
            "optimizer_time": 249.43356347084045,
        }
    """
    return {
        "cost_function_evals": r.cost_function_evals,
        "eigenvalue": float(r.eigenvalue),
        "optimal_parameters": {
            p.name: float(v) for p, v in r.optimal_parameters.items()
        },
        "optimizer_time,": r.optimizer_time,
    }


def benchmark(
    ham_dir: str | Path,
    reorder_result: pd.DataFrame,
    reorder_result_dir: str | Path,
    output_dir: str | Path,
    n_trials: int = 1,
    n_jobs: int = 32,
    qaoa_config: dict[str, Any] | None = None,
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
        qaoa_config (dict[str, Any] | None, optional): Configuration for QAOA,
            which can contains the following keys:
            - `provider` (str): The name of the fake provider to use
            - `n_shots` (int): Number of shots for the estimator
            - `seed` (int): Seed for the estimator
            - `n_steps` (int): Number of QAOA steps
            - `max_iter` (int): Maximum number of iterations for the optimizer
              ([`SPSA`](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SPSA.html#spsa))
    """
    ham_dir, reorder_result_dir = Path(ham_dir), Path(reorder_result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Number of Hamiltonians: {}", len(reorder_result))

    jobs = []
    progress = tqdm(
        reorder_result.iterrows(),
        desc="Listing jobs",
        total=len(reorder_result),
    )
    for _, row in progress:
        for i in range(n_trials):
            jid = hash_dict(  # unique simulation job identifier
                {
                    "hid": row["hid"],
                    "trotterization": row["trotterization"],
                    "n_timesteps": row["n_timesteps"],
                    "order": row["order"],
                    "method": row["method"],
                    "jid": row["jid"],  # unique reordering job identifier
                    "trial": i,
                }
            )
            output_file = jid_to_json_path(jid, output_dir)
            if output_file.is_file() and output_file.stat().st_size > 0:
                continue
            ham_file, key = hid_to_file_key(row["hid"], ham_dir)
            # ↓ files from the reording jobs are named like this
            bp = jid_to_json_path(row["jid"], reorder_result_dir)
            kw = {
                "ham_file": ham_file,
                "key": key,
                "order_file": bp.with_suffix(".hdf5"),
                "circuit_file": bp.with_suffix(".qpy"),
                "output_file": output_file,
                "qaoa_config": qaoa_config,
            }
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
