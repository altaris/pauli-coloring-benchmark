"""Implementation of the QAOA algorithm."""

from functools import partial
import numpy as np
from loguru import logger as logging
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.providers import BackendV2 as Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit_ibm_runtime import RuntimeJobV2 as RuntimeJob
import nevergrad as ng


def _cost_function(
    parameters: np.ndarray,
    ansatz: QuantumCircuit,
    operator: SparsePauliOp,
    estimator: IBMEstimator | AerEstimator,
    jobs: list[tuple[np.ndarray, RuntimeJob]] | None = None,
) -> np.float64:
    """
    Function to minimize in the `scipy.optimize.minimize` routine. It returns
    the energy **times $-1$** of the operator given an ansatz and a set of
    parameters.  Therefore, minimizing this function will find variational
    parameters that produce a **high energy** state.

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
    return -energy


def _job_to_dict(job: RuntimeJob) -> dict:
    data = job.result()[0].data
    return {
        "energy": float(data.evs[0]),
        "std": float(data.stds[0]),
        "job_id": job.job_id(),
        "session_id": getattr(job, "session_id", None),
        # ↑ doesn't exist for sims.
    }


def qaoa(
    operator: SparsePauliOp,
    backend: Backend,
    estimator: IBMEstimator | AerEstimator,
    cost_qc: QuantumCircuit | None = None,
    n_qaoa_steps: int = 1,
    pm_optimization_level: int = 0,
    max_iter: int = 1000,
) -> tuple[
    tuple[np.ndarray, float], tuple[np.ndarray, np.ndarray], list[dict]
]:
    """
    Runs QAOA on the given operator using the given backend. The Trotterization
    of the cost operator can be given explicitely as a `QuantumCircuit`.
    Otherwise, it is automatically Trotterized.

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
            "energy": -0.006450488764709524,
            "std": 0.0,
            "job_id": "3c94c73c-385b-4e71-8675-e3e1ad76aa4e",
            "session_id": None,
            "step": 1,
            "n_qaoa_steps": 2,
            "pm_optimization_level": 3,
            "backend": "fake_kawasaki"
        }

    """
    pm = generate_preset_pass_manager(
        target=backend.target, optimization_level=pm_optimization_level
    )
    ansatz = QAOAAnsatz(
        cost_operator=cost_qc if cost_qc is not None else operator,
        reps=n_qaoa_steps,
    )
    ansatz_isa = pm.run(ansatz)
    logging.debug(
        "ISA ansatz depth/size: {}/{}",
        ansatz_isa.depth(),
        ansatz_isa.size(),
    )
    operator_isa = operator.apply_layout(ansatz_isa.layout)
    _jrs: list[tuple[np.ndarray, RuntimeJob]] = []
    parametrization = ng.p.Array(
        shape=(len(ansatz_isa.parameters),),
        lower=0,
        upper=np.array(
            ([np.pi] * (len(ansatz_isa.parameters) // 2))  # β's
            + ([2 * np.pi] * (len(ansatz_isa.parameters) // 2))  # γ's
        ),
    )
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization, budget=max_iter
    )
    optimizer.minimize(
        partial(
            _cost_function,
            ansatz=ansatz_isa,
            operator=operator_isa,
            estimator=estimator,
            jobs=_jrs,
        ),
        verbosity=2,
    )
    best_x, best_e = _jrs[0][0], _jrs[0][1].result()[0].data.evs[0]
    for x, j in _jrs[1:]:
        e = j.result()[0].data.evs[0]
        if e > best_e:  # Energy maximization
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
