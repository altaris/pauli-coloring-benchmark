"""
"Quick" and dirty way to run the QAOA benchmark by simulation.
"""

import sqlite3

import pandas as pd
from loguru import logger as logging

from pcb.benchmark.simulate import benchmark

if __name__ == "__main__":
    # =========================================================================
    # LOAD ALL MAXCUT RESULTS
    # =========================================================================

    logging.info("Loading reordering results")
    PREFIX = "binaryoptimization/maxcut"
    clauses = [
        f"`hid` LIKE '{PREFIX}%'",
        "`trotterization` = 'suzuki_trotter'",
        "`n_timesteps` = 1",
        "`order` = 4",
        "`n_qubits` <= 64",
        "`n_terms` <= 16",
    ]
    QUERY = "SELECT * FROM `results` WHERE " + " AND ".join(clauses)
    with sqlite3.connect("out/reorder.hcs=m/results.db") as db:
        df = pd.read_sql(QUERY, db)

    # =========================================================================
    # PROCESS THE DATAFRAME
    # =========================================================================

    logging.debug("Basic processing")
    df.dropna(inplace=True)
    df["time"] = df["reordering_time"] + df["synthesis_time"]
    df.drop(
        ["reordering_time", "synthesis_time", "order", "n_timesteps"],
        axis=1,
        inplace=True,
    )
    df = df.astype({"n_terms": int, "depth": int, "n_qubits": int})

    # Keep only one row per (hid, method) tuple (the shallowest one)
    df = df.loc[df.groupby(["hid", "method"])["depth"].idxmin()]

    # add depth% and time% columns
    logging.debug("Adding the time_pc and depth_pc columns")
    dfp, lst = df.pivot(index="hid", columns="method"), []
    for m in df["method"].unique():
        depth_pc = dfp["depth"][m] * 100 / (dfp["depth"]["none"] + 1e-3)
        time_pc = dfp["time"][m] * 100 / (dfp["time"]["none"] + 1e-3)
        lst.append(
            pd.DataFrame(
                {"method": m, "depth_pc": depth_pc, "time_pc": time_pc}
            )
        )
    a = pd.concat(lst)
    a = a.pivot(columns="method")
    df = dfp.join(a).stack(future_stack=True).reset_index()
    df.dropna(inplace=True)
    df = df.astype({"n_terms": int, "depth": int, "n_qubits": int})

    logging.info("Preprocessing done, left with {} rows", len(df))

    # =========================================================================
    # SORT HAMILTONIANS
    # =========================================================================

    df.sort_values(
        ["n_terms", "depth", "n_qubits"], ascending=True, inplace=True
    )

    # =========================================================================
    # RUN THE BENCHMARK AND GO HOME
    # =========================================================================

    logging.info("Starting the benchmark")
    benchmark(
        ham_dir="out/ham",
        reorder_result=df,
        reorder_result_dir="out/reorder.hcs=m",
        output_dir="out/sim.hm=p,hc=m,max",
        n_trials=1,
        n_jobs=8,
    )
