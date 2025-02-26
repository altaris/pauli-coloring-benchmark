import sqlite3

import pandas as pd
from loguru import logger as logging

if __name__ == "__main__":
    paths = [
        "out/reordering/results.db",  # first one is also the output path
        "out/tsp.db",
    ]
    dfs = []
    for p in paths:
        logging.info("Reading {}", p)
        with sqlite3.connect(p) as db:
            dfs.append(pd.read_sql("SELECT * FROM `results`", db))
    logging.info("Merging")
    df = pd.concat(dfs, ignore_index=True)
    logging.info("Writing to {}", paths[0])
    with sqlite3.connect(paths[0]) as db:
        df.to_sql("results", db, if_exists="replace", index=False)
