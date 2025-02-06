"""CLI module"""

# pylint: disable=import-outside-toplevel

import os
from pathlib import Path
from typing import TextIO

import click
from loguru import logger as logging

from .logging import setup_logging

HAMLIB_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/"


@click.group()
@click.option(
    "--logging-level",
    default=os.getenv("LOGGING_LEVEL", "info"),
    help=(
        "Logging level, case insensitive. Defaults to 'info'. Can also be set "
        "using the LOGGING_LEVEL environment variable."
    ),
    type=click.Choice(
        ["critical", "debug", "error", "info", "warning"],
        case_sensitive=False,
    ),
)
@logging.catch
def main(logging_level: str) -> None:
    """Entrypoint."""
    setup_logging(logging_level)


@main.command()
@click.argument("index_db", type=Path)
@click.argument("ham_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option("--prefix", type=str, default="")
@click.option("--n-trials", type=int, default=10)
@click.option("--n-jobs", type=int, default=32)
def benchmark(
    index_db: Path,
    ham_dir: Path,
    output_dir: Path,
    prefix: str,
    n_trials: int,
    n_jobs: int,
) -> None:
    """Runs a benchmark on some or all Hamiltonian files in the index"""
    from datetime import datetime
    import sqlite3

    import pandas as pd

    from .benchmark import benchmark as _benchmark

    start = datetime.now()

    logging.info("Reading index {}", index_db)
    db = sqlite3.connect(index_db)
    if prefix:
        query = f"SELECT * FROM `index` WHERE `dir` LIKE '{prefix}%'"
    else:
        query = "SELECT * FROM `index`"
    index = pd.read_sql(query, db)
    db.close()

    df = _benchmark(
        index=index,
        ham_dir=ham_dir,
        output_dir=output_dir,
        n_trials=n_trials,
        prefix=prefix,
        n_jobs=n_jobs,
    )
    df.to_csv(output_dir / "results.csv", index=False)

    logging.info("Done in: {}", datetime.now() - start)


@main.command()
@click.argument("index_db", type=Path)
@click.option(
    "--hamlib-url",
    default=HAMLIB_URL,
    help="Defaults to " + HAMLIB_URL,
    type=str,
)
def build_index(hamlib_url: str, index_db: Path) -> None:
    """
    Builds the index of all Hamiltonian files in the HamLib website and writes
    it to a SQLite database, under table `index`.
    """
    from datetime import datetime
    import sqlite3

    from .hamlib import build_index as _build_index

    start = datetime.now()

    logging.info("Building index of Hamiltonian files in: {}", hamlib_url)
    index_db.parent.mkdir(parents=True, exist_ok=True)
    df = _build_index(hamlib_url)

    logging.info("Writing index to: {}", index_db)
    db = sqlite3.connect(index_db)
    df.to_sql("index", db, if_exists="replace", index=True)
    db.close()

    logging.info("Done in: {}", datetime.now() - start)


@main.command()
@click.argument("output_dir", type=Path)
def consolidate(output_dir: Path) -> None:
    """
    Consolidates benchmark results .json files in OUTPUT_DIR/jobs into a single
    .csv file in OUTPUT_DIR/results.csv.
    """
    from datetime import datetime
    import sqlite3

    from .benchmark import consolidate as _consolidate

    start = datetime.now()

    logging.info("Consolidating benchmark results in: {}", output_dir)
    df = _consolidate(output_dir / "jobs")

    results_db = output_dir / "results.db"
    logging.info("Writing results to: {}", results_db)
    db = sqlite3.connect(results_db)
    df.to_sql("results", db, if_exists="replace", index=True)
    db.close()

    logging.info("Done in: {}", datetime.now() - start)


@main.command()
@click.argument("index_db", type=Path)
@click.argument("output_dir", type=Path)
@click.option("--prefix", type=str, default="")
def download(index_db: Path, output_dir: Path, prefix: str) -> None:
    """Downloads some or all Hamiltonian files in the index"""
    from datetime import datetime
    import sqlite3

    import pandas as pd

    from .hamlib import download as _download

    start = datetime.now()

    db = sqlite3.connect(index_db)
    if prefix:
        query = f"SELECT * FROM `index` WHERE `dir` LIKE '{prefix}%'"
    else:
        query = "SELECT * FROM `index`"
    index = pd.read_sql(query, db)
    db.close()

    _download(index, output_dir, prefix)

    logging.info("Done in: {}", datetime.now() - start)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
