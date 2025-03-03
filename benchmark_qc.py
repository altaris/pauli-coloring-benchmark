"""
"Quick" and dirty way to run the QAOA benchmark on a real quantum computer.
"""

import sqlite3

import pandas as pd
from loguru import logger as logging

from pcb.benchmark.run import benchmark

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
        "`n_qubits` >= 64",
        "`n_qubits` <= 127",
        "`n_terms` >= 16",
        "`n_terms` <= 64",
        "`depth` >= 16",
        "`depth` <= 256",
    ]
    QUERY = "SELECT * FROM `results` WHERE " + " AND ".join(clauses)
    with sqlite3.connect("out/reordering/results.db") as db:
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
    # PICK THE "TOP" HAMILTONIANS
    # =========================================================================

    N_CANDIDATE_HAM = 25
    d, hids = df[df["method"] == "none"], set()
    criterions = [  # (columns name, sort ascending?)
        ("n_terms", False),  # Hams. with most terms
        # ("n_qubits", False),  # Hams. with most qubits
        # ("depth", False),  # Hams. with deepest evo. circuit
        # ("depth_pc", True),  # Hams. with best evo. circuit depth reduction
    ]
    logging.debug(
        "Selecting top {} Hamiltonians by {}",
        N_CANDIDATE_HAM,
        ", ".join(c[0] for c in criterions),
    )
    for column, ascending in criterions:
        e = d.sort_values(column, ascending=ascending).head(N_CANDIDATE_HAM)
        hids |= set(e["hid"])

    # =========================================================================
    # FILTER THE DATAFRAME TO ONLY KEEP THE SELECTED HAMILTONIANS
    # =========================================================================

    df = df[df["hid"].isin(hids)]
    logging.info(
        "Selected {} Hamiltonians corresponding to {} experiments",
        len(hids),
        len(df),
    )
    logging.debug("HIDs:\n    {}", "\n    ".join(hids))
    lst = [
        f"{row['jid']} method={row['method']:<15}, "
        f"n_qubits={row['n_qubits']:<3d}, n_terms={row['n_terms']:<4d}, "
        f"depth={row['depth']:<4d}, depth%={row['depth_pc']:<3.2f}"
        for _, row in df.iterrows()
    ]
    logging.debug("Reordering JIDs:\n    {}", "\n    ".join(lst))

    # =========================================================================
    # RUN THE BENCHMARK AND GO HOME
    # =========================================================================

    logging.info("Starting the benchmark")
    benchmark(
        ham_dir="out/ham",
        reorder_result=df,
        reorder_result_dir="out/reordering",
        output_dir="out/run",
        n_trials=1,
    )
