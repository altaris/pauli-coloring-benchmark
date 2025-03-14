"""Implementation of the QAOA algorithm."""

import numpy as np
from loguru import logger as logging
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import PrimitiveJob
from qiskit.providers import BackendV2 as Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit_ibm_runtime import RuntimeJobV2 as IBMRuntimeJob
from scipy.optimize import minimize


def _energy(
    x: np.ndarray,
    ansatz: QuantumCircuit,
    operator: SparsePauliOp,
    estimator: IBMEstimator | AerEstimator,
    records: list[tuple[np.ndarray, float, dict]] | None = None,
) -> float:
    """
    Evaluates a set of parameters for the QAOA ansatz. Returns an array of all
    resulting energies and a dictionary containing data about the job submitted
    to IBMQ or Aer. See `_job_to_dict` for the structure of the dictionary.

    Args:
        parameters (np.ndarray): A `(N, n_qaoa_params)` array of real parameters
        ansatz (QuantumCircuit):
        operator (SparsePauliOp):
        estimator (IBMEstimator | AerEstimator):
        records (list, optional): If provided, results of this evaluation will
            be appended.
    """
    if x.ndim != 1:
        raise ValueError(
            f"Parameters must be a 1D array, got shape: {x.shape}"
        )
    pubs = [(ansatz, [operator], [x])]
    job = estimator.run(pubs=pubs)
    results = job.result()
    e = float(results[0].data.evs[0])
    if records is not None:
        records.append((x, e, _job_to_dict(job)))
    return e


def _job_to_dict(job: IBMRuntimeJob | PrimitiveJob) -> dict:
    """
    Note:
        The keys will be prefixed with `ibmq_` even if the job is an Aer job.
    """
    return {
        "ibmq_jid": job.job_id(),
        "ibmq_sid": getattr(job, "session_id", None),
    }


def qaoa(
    operator: SparsePauliOp,
    backend: Backend,
    estimator: IBMEstimator | AerEstimator,
    cost_qc: QuantumCircuit | None = None,
    n_qaoa_steps: int = 2,
    pm_optimization_level: int = 3,
    max_iter: int = 128,
) -> tuple[
    tuple[np.ndarray, float], tuple[np.ndarray, np.ndarray], list[dict]
]:
    """
    Runs QAOA on the given operator using the given backend. The Trotterization
    of the cost operator can be given explicitely as a `QuantumCircuit`.
    Otherwise, it is automatically Trotterized.

    This method submits PUBs in batches, meaning that at every iteration,
    `batch_size` parameters are tested.

    Warning:
        This method tries to *MAXIMIZE* the energy of the operator. If you want
        to minimize, flip the sign of the weights in `operator`.

    Args:
        operator (SparsePauliOp): The operator whose energy to maximize
        backend (Backend):
        estimator (IBMEstimator | AerEstimator):
        cost_qc (QuantumCircuit | None, optional):
        n_qaoa_steps (int, optional):
        pm_optimization_level (int, optional):

    Returns:
        1. A tuple containing the optimal parameters and the optimal energy.
        2. A tuple containing array of all parameters that were tried and the
           array of all resulting energies.
        3. A list of dict decribing the results of each job. An element looks
           like:

            {
                "energy": 0.006450488764709524,
                "ibmq_job_id": "3c94c73c-385b-4e71-8675-e3e1ad76aa4e",
                "ibmq_session_id": "cz4h80039f40008scarg",
                "batch": 3,
            }
    """
    ansatz = QAOAAnsatz(
        cost_operator=cost_qc if cost_qc is not None else operator,
        reps=n_qaoa_steps,
    )
    pm = generate_preset_pass_manager(
        target=backend.target, optimization_level=pm_optimization_level
    )
    ansatz_isa = pm.run(ansatz)
    operator_isa = operator.apply_layout(ansatz_isa.layout)
    logging.debug(
        "ISA ansatz depth/size: {}/{}",
        ansatz_isa.depth(),
        ansatz_isa.size(),
    )
    x0 = np.array(
        ([np.pi / 2] * (len(ansatz_isa.parameters) // 2))  # β's
        + ([np.pi] * (len(ansatz_isa.parameters) // 2))  # γ's
    )
    bounds = (
        ([(0, np.pi)] * (len(ansatz_isa.parameters) // 2))  # β's
        + ([(0, 2 * np.pi)] * (len(ansatz_isa.parameters) // 2))  # γ's
    )
    records: list[tuple[np.ndarray, float, dict]] = []
    minimize(
        lambda *args: -1 * _energy(*args),  # energy maximization
        x0=x0,
        args=(ansatz_isa, operator_isa, estimator, records),
        method="cobyla",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    all_x = np.stack([x for x, _, _ in records])
    all_e = np.array([e for _, e, _ in records])
    results = [
        {"energy": float(e), "step": i, **m}
        for i, (_, e, m) in enumerate(records)
    ]
    j = all_e.argmax()
    return (all_x[j], all_e[j]), (all_x, all_e), results
