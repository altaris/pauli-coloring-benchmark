"""
Part of the benchmark where optimal QAOA parameters (from either the simulation
or actual quantum execution) are loaded, the ansatz is rebuilt and sampled from,
and the results are tallied and the maxcuts computed.
"""

import sqlite3
from itertools import batched
from pathlib import Path
from typing import Generator, TypeAlias

import numpy as np
import pandas as pd
from loguru import logger as logging
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import SamplerPubResult
from qiskit.providers import BackendV2 as Backend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as IBMSampler
from tqdm import tqdm

from pcb.benchmark.utils import jid_to_json_path
from pcb.io import load, save
from pcb.qiskit import trim_qc
from pcb.utils import hash_dict

PREFIX = "binaryoptimization/maxcut"

REORDER_PATH = Path("out/reorder.hcs=m")
SIM_PATH = Path("out/run.aer.hm=p,hc=m,max")
SMPL_PATH = Path("out/smpl.hm=p,hc=m,max")

N_SHOTS = 1024
BATCH_SIZE = 1024
# Number of execution (one shot on one qc.) in a job: BATCH_SIZE * N_SHOTS
# Max allowed: 5M (https://qiskit.qotlabs.org/guides/job-limits#maximum-executions)

Pub: TypeAlias = tuple[QuantumCircuit, list[np.ndarray]]


def build_pub(
    reordering_jid: str,
    simulation_jid: str,
    backend: Backend,
    n_qaoa_steps: int = 2,
) -> Pub:
    """
    Creates a submittable PUB from a reordering job ID (to load the Trotterized
    cost operator) and a simulation job ID (to get the optimal QAOA parameters).
    """
    data = load(
        jid_to_json_path(simulation_jid, SIM_PATH).with_suffix(".hdf5")
    )
    x = data["best_parameters"]
    p = jid_to_json_path(simulation_jid, SMPL_PATH).with_suffix(".qpy.gz")
    if p.is_file() and p.stat().st_size > 0:
        ansatz_isa = load(p)
    else:
        qc = load(
            jid_to_json_path(reordering_jid, REORDER_PATH).with_suffix(
                ".qpy.gz"
            )
        )
        qc, _ = trim_qc(qc)
        ansatz = QAOAAnsatz(cost_operator=qc, reps=n_qaoa_steps)
        ansatz.measure_all()
        pm = generate_preset_pass_manager(
            target=backend.target, optimization_level=3
        )
        ansatz_isa = pm.run(ansatz)
        save(ansatz_isa, p)
    return (ansatz_isa, [x])


def jobs(
    df: pd.DataFrame, backend: Backend
) -> Generator[tuple[dict, Pub], None, None]:
    """
    Generator that yields `(metadata, pub)` tuples. Since running a pass manager
    on an ansatz takes time, it's better to get these as needed.

    This generator skips jobs that have already been ran.
    """
    for _, r in df.iterrows():
        sampling_jid = hash_dict(
            {
                "reordering_jid": r["reordering_jid"],
                "simulation_jid": r["simulation_jid"],
            }
        )
        output_file = jid_to_json_path(sampling_jid, SMPL_PATH)
        if output_file.is_file() and output_file.stat().st_size > 0:
            # logging.debug("Skipping job {} (already ran)", sampling_jid)
            continue
        pub = build_pub(r["reordering_jid"], r["simulation_jid"], backend)
        meta = {
            "hid": r["hid"],
            "n_shots": N_SHOTS,
            "reordering_jid": r["reordering_jid"],
            "sampling_jid": sampling_jid,
            "simulation_jid": r["simulation_jid"],
        }
        yield (meta, pub)


def load_reorder_df() -> pd.DataFrame:
    """Load the reordering benchmark results as a pandas DataFrame."""
    query = f"SELECT * FROM `results` WHERE `hid` LIKE '{PREFIX}%'"
    with sqlite3.connect(REORDER_PATH / "results.db") as db:
        df = pd.read_sql(query, db)
    df.dropna(inplace=True)
    df.drop(["hid"], axis=1, inplace=True)
    df.rename({"jid": "reordering_jid"}, axis=1, inplace=True)
    df["time"] = df["reordering_time"] + df["synthesis_time"]
    df.drop(["reordering_time", "synthesis_time"], axis=1, inplace=True)
    logging.debug("Loaded reordering benchmark results: {} rows", len(df))
    return df


def load_sim_df() -> pd.DataFrame:
    """Load the simulation benchmark results as a pandas DataFrame."""
    clauses = [
        f"`hid` LIKE '{PREFIX}%'",
    ]
    query = "SELECT * FROM `results` WHERE " + " AND ".join(clauses)
    with sqlite3.connect(SIM_PATH / "results.db") as db:
        df = pd.read_sql(query, db)
    df.drop(
        ["ibmq_jid", "ibmq_sid", "backend", "step"],
        axis=1,
        inplace=True,
    )
    df.rename({"jid": "simulation_jid"}, axis=1, inplace=True)
    df.dropna(inplace=True)
    logging.debug("Loaded simulation benchmark results: {} rows", len(df))
    df = df.groupby(["simulation_jid"]).max().reset_index()
    logging.debug(
        (
            "Selected optimization iterations with highest energy, "
            "left with {} rows"
        ),
        len(df),
    )
    return df


def process_result(meta: dict, result: SamplerPubResult) -> None:
    """
    Saves sampling results to a file. The metada are saved in a JSON file and
    the counts in a CSV file.

    TODO:
        Wouldn't it be nice to compute the cut values here too?
    """
    counts = result.data.meas.get_counts()
    df = pd.DataFrame(
        {"string": counts.keys(), "count": counts.values(), "hid": meta["hid"]}
    )
    output_file = jid_to_json_path(meta["sampling_jid"], SMPL_PATH)
    save(meta, output_file)
    save(df, output_file.with_suffix(".csv"))


def main() -> None:
    """Main function (duh)"""

    # =========================================================================
    # LOAD AND JOIN SIMULATION AND REORDERING BENCHMARK RESULTS
    # =========================================================================

    rdf, sdf = load_reorder_df(), load_sim_df()
    rdf, sdf = rdf.set_index("reordering_jid"), sdf.set_index("reordering_jid")
    df = sdf.join(rdf, how="inner")
    df.reset_index(inplace=True)
    if "index" in df.columns:
        df.rename({"index": "reordering_jid"}, axis=1, inplace=True)
    logging.debug("Joined reordering and simulation result dataframes")

    # =========================================================================
    # ONLY KEEP HIDs WHERE ALL METHODS HAVE BEEN RAN
    # =========================================================================

    n_methods, keep = len(df["method"].unique()), set()
    for hid in df["hid"].unique():
        methods = df[df["hid"] == hid]["method"].unique()
        if len(methods) == n_methods:
            keep |= set(df[df["hid"] == hid]["simulation_jid"].unique())
    df = df[df["simulation_jid"].isin(keep)]
    logging.info(
        (
            "Keeping HIDs where all reordering methods have been simulated: "
            "{} HID(s), {} JID(s), {} rows"
        ),
        len(df["hid"].unique()),
        len(keep),
        len(df),
    )

    # =========================================================================
    # GET BACKEND
    # =========================================================================

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True)
    sampler = IBMSampler(backend)
    logging.info("Using backend: {}", backend.name)

    # =========================================================================
    # RUN SAMPLER
    # =========================================================================

    batches = batched(jobs(df, backend), BATCH_SIZE)
    n_batches = (len(df) // BATCH_SIZE) + (
        0 if len(df) % BATCH_SIZE == 0 else 1
    )
    logging.debug(
        (
            "Starting to run jobs: batch size = {}, nb. of shots per pub = {}, "
            "nb. of executions per job = {} (max is 5M), "
            "estimated number of batches = {}"
        ),
        BATCH_SIZE,
        N_SHOTS,
        BATCH_SIZE * N_SHOTS,
        n_batches,
    )
    logging.debug(
        "Note: pass managers need to be run on all ansatzes in a batch "
        "before the job can be submitted to IBMQ. If the batch size is large, "
        "this might take a while..."
    )
    for bn, batch in enumerate(
        tqdm(batches, desc="Running jobs", total=n_batches)
    ):
        try:
            metas, pubs = zip(*batch)
            job = sampler.run(pubs)  # shit happens here
            results = job.result()
            for m, r in zip(metas, results):
                m.update(
                    {
                        "ibmq_jid": job.job_id(),
                        "ibmq_sid": job.session_id,
                        "backend": backend.name,
                    }
                )
                process_result(m, r)
        except Exception as e:
            logging.error("Failed to process batch {}: {}", bn, e)


if __name__ == "__main__":
    main()
