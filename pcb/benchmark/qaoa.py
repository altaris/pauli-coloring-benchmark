"""Implementation of the QAOA algorithm."""

import numpy as np
from loguru import logger as logging
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import RuntimeJobV2 as RuntimeJob
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from scipy.optimize import minimize


def _cost_function(
    parameters: np.ndarray,
    ansatz: QuantumCircuit,
    operator: SparsePauliOp,
    estimator: Estimator,
    jobs: list[tuple[np.ndarray, RuntimeJob]] | None = None,
) -> np.float64:
    """
    Function to minimize in the `scipy.optimize.minimize` routine. It returns
    the of energy of the operator given an ansatz and a set of parameters.
    Therefore, minimizing this function will find variational parameters that
    produce a low energy state.

    If `jobs` is not `None`, it will append a `(parameters, job)` tuple to it.
    `job` is a
    [`RuntimeJobV2`](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/runtime-job-v2)
    that handles the actual IBM Quantum Runtime job.
    """
    pub = (ansatz, [operator], [parameters])
    job = estimator.run(pubs=[pub])
    energy: np.float64 = job.result()[0].data.evs[0]
    if jobs is not None:
        jobs.append((parameters, job))
        logging.debug("Iteration: {}, energy: {}", len(jobs), energy.round(5))
    return energy


def _job_to_dict(job: RuntimeJob) -> dict:
    data, meta = job.result()[0].data, job.result()[0].metadata
    return {
        "energy": float(data.evs[0]),
        "std": float(data.stds[0]),
        "ensemble_standard_error": float(data.ensemble_standard_error[0]),
        "n_shots": meta["shots"],
        "target_precision": meta["target_precision"],
        "num_randomizations": meta["num_randomizations"],
        "job_id": job.job_id(),
        "session_id": job.session_id,
    }


def qaoa(
    operator: SparsePauliOp,
    cost_qc: QuantumCircuit,
    backend: IBMBackend,
    n_qaoa_steps: int = 1,
    n_shots: int = 1024,
    pm_optimization_level: int = 0,
    max_iter: int = 1000,
) -> tuple[
    tuple[np.ndarray, float], tuple[np.ndarray, np.ndarray], list[dict]
]:
    """
    Runs QAOA on the given operator using the given backend. The cost operator
    $H_C$ is not obtained from the Trotterization of the operator but rather has
    to be provided explicitly.

    Warning:
        This method tries to *MAXIMIZE* the energy of the operator. If you want
        to minimize, flip the sign of the weights in `operator`.

    Returns:
        1. A tuple containing the optimal parameters and the optimal energy.
        2. A tuple containing array of all parameters that were tried and the
           array of all resulting energies.
        3. A list of dict decribing the results of each job. An element looks
           like:

            {
                "energy": 0.4211,
                "std": 0.2941,
                "ensemble_standard_error": 0.3042,
                "n_shots": 1024,
                "target_precision": 0.0312,
                "num_randomizations": 16,
                "job_id": "...",
                "session_id": "...",
                "step": 1,  # starting at 1
                "n_qaoa_steps": 2,
                "pm_optimization_level": 0,
                "backend": "ibm_...",
            }
    """
    pm = generate_preset_pass_manager(
        target=backend.target, optimization_level=pm_optimization_level
    )
    ansatz = QAOAAnsatz(cost_operator=cost_qc, reps=n_qaoa_steps)
    ansatz.measure_all()
    ansatz_isa = pm.run(ansatz)
    operator_isa = operator.apply_layout(ansatz_isa.layout)

    _jrs: list[tuple[np.ndarray, RuntimeJob]] = []
    with Session(backend=backend) as session:
        logging.debug("Opened session: {}", session.session_id)
        # estimator = Estimator(
        #     mode=session,
        #     options={
        #         "default_shots": n_shots,
        #         "dynamical_decoupling": {"enable": False},
        #         "resilience_level": 1,  # meas. twirling but not for gates
        #         "seed_estimator": 0,
        #     },
        # )
        estimator = Estimator(
            mode=session,
            options={
                "default_shots": n_shots,
                "dynamical_decoupling": {
                    "enable": True,
                    "sequence_type": "XY4",
                },
                "seed_estimator": 0,
                "twirling": {
                    "enable_gates": True,
                    "num_randomizations": "auto",
                },
            },
        )
        x0 = np.array(
            ([np.pi / 2] * (len(ansatz_isa.parameters) // 2))  # β's
            + ([np.pi] * (len(ansatz_isa.parameters) // 2))  # γ's
        )
        minimize(
            _cost_function,
            x0,
            method="cobyla",
            args=(ansatz_isa, operator_isa, estimator, _jrs),
            tol=1e-2,
            options={"maxiter": max_iter, "disp": False},
        )

    best_x, best_e = _jrs[0][0], _jrs[0][1].result()[0].data.evs[0]
    for x, j in _jrs[1:]:
        e = j.result()[0].data.evs[0]
        if e < best_e:
            best_x, best_e = x, e
    all_x = np.stack([x for x, _ in _jrs])
    all_e = np.array([j.result()[0].data.evs[0] for _, j in _jrs])

    job_results = []
    for i, (_, j) in enumerate(_jrs):
        data = _job_to_dict(j)
        data.update(
            {
                "step": i + 1,
                "n_qaoa_steps": n_qaoa_steps,
                "pm_optimization_level": pm_optimization_level,
                "backend": backend.name,
            }
        )
        job_results.append(data)

    return (best_x, best_e), (all_x, all_e), job_results
