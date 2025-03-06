"""Implementation of the QAOA algorithm."""

import nevergrad as ng
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

from pcb.utils import EarlyStoppingLoop


def _cost_function(
    parameters: np.ndarray,
    ansatz: QuantumCircuit,
    operator: SparsePauliOp,
    estimator: IBMEstimator | AerEstimator,
) -> tuple[np.ndarray, dict]:
    """
    Evaluates a set of parameters for the QAOA ansatz. Returns an array of all
    resulting energies and a dictionary containing data about the job submitted
    to IBMQ or Aer. See `_job_to_dict` for the structure of the dictionary.

    Args:
        parameters (np.ndarray): A `(N, n_qaoa_params)` array of real parameters
        ansatz (QuantumCircuit):
        operator (SparsePauliOp):
        estimator (IBMEstimator | AerEstimator):
    """
    if parameters.ndim != 2:
        raise ValueError(
            f"Parameters must be a 2D array, got shape: {parameters.shape}"
        )
    pubs = [(ansatz, [operator], [x]) for x in parameters]
    job = estimator.run(pubs=pubs)
    results = job.result()
    es = np.array([r.data.evs[0] for r in results])
    return es, _job_to_dict(job)


def _job_to_dict(job: IBMRuntimeJob | PrimitiveJob) -> dict:
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
    batch_size: int = 16,
    max_n_batches: int = 10,
    patience: int = 3,
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
        batch_size (int, optional):
        max_n_batches (int, optional):

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

    parametrization = ng.p.Array(
        shape=(len(ansatz_isa.parameters),),
        lower=0,
        upper=np.array(
            ([np.pi] * (len(ansatz_isa.parameters) // 2))  # β's
            + ([2 * np.pi] * (len(ansatz_isa.parameters) // 2))  # γ's
        ),
    )
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization,
        budget=max_n_batches * batch_size,
        num_workers=batch_size,  # to make batch_size asks concurrently
    )

    loop = EarlyStoppingLoop(
        max_iter=max_n_batches, patience=patience, delta=1e-2, mode="max"
    )
    records: list[tuple[np.ndarray, np.float64, dict, int]] = []
    for i in loop:
        ps = [optimizer.ask() for __ in range(batch_size)]
        xs = np.stack([p.value for p in ps])
        es, meta = _cost_function(xs, ansatz_isa, operator_isa, estimator)
        for p, e in zip(ps, es):
            optimizer.tell(p, -1 * e)  # !!!
            records.append((p.value, e, meta, i))
        loop.propose(None, es.max())

    all_x = np.stack([x for x, _, _, _ in records])
    all_e = np.array([e for _, e, _, _ in records])
    results = [{"energy": float(e), "batch": i, **m} for _, e, m, i in records]
    j = all_e.argmax()
    return (all_x[j], all_e[j]), (all_x, all_e), results
