"""CLI module"""

# pylint: disable=import-outside-toplevel

import os
import sqlite3
from pathlib import Path

import click
import pandas as pd
from loguru import logger as logging

from .logging import setup_logging
from .utils import timed

HAMLIB_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/"


def _open_index(
    index_db: Path,
    prefix: str = "",
    min_qubits: int | None = None,
    max_qubits: int | None = None,
    min_terms: int | None = None,
    max_terms: int | None = None,
) -> pd.DataFrame:
    query, clauses = "SELECT * FROM `index`", []
    if prefix:
        clauses.append(f"`hid` LIKE '{prefix}%'")
    if min_qubits is not None and min_qubits > 0:
        clauses.append(f"`n_qubits` >= {min_qubits}")
    if max_qubits is not None and max_qubits > 0:
        clauses.append(f"`n_qubits` <= {max_qubits}")
    if min_terms is not None and min_terms > 0:
        clauses.append(f"`n_terms` >= {min_terms}")
    if max_terms is not None and max_terms > 0:
        clauses.append(f"`n_terms` <= {max_terms}")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    logging.debug("Query: {}", query)
    db = sqlite3.connect(index_db)
    df = pd.read_sql(query, db)
    db.close()
    return df


@click.group()
@click.option(  # --logging-level
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
@click.option(  # --prefix
    "--prefix",
    type=str,
    default="",
    help=(
        "Filter the Hamiltonians to benchmark by prefix. Example: "
        "'binaryoptimization/maxcut'"
    ),
)
@click.option("--n-trials", type=int, default=5, help="Defaults to 5")
@click.option("--n-jobs", type=int, default=32, help="Defaults to 32")
@click.option(  # --methods
    "--methods",
    type=str,
    default="none,saturation",
    help=(
        "Comma-separated list of reordering methods to benchmark. Defaults to "
        "'none,saturation'"
    ),
)
@click.option("--min-qubits", type=int, default=0, help="Defaults to 0")
@click.option(  # --max-qubits
    "--max-qubits",
    type=int,
    default=256,
    help="Defaults to 256. Set to 0 to disable",
)
@click.option("--min-terms", type=int, default=0, help="Defaults to 0")
@click.option(  # --max-terms
    "--max-terms",
    type=int,
    default=10000,
    help="Defaults to 10000. Set to 0 to disable",
)
@timed
def benchmark_reorder(
    index_db: Path,
    ham_dir: Path,
    output_dir: Path,
    prefix: str,
    n_trials: int,
    n_jobs: int,
    methods: str,
    min_qubits: int,
    max_qubits: int,
    min_terms: int,
    max_terms: int,
) -> None:
    """Runs a benchmark on some or all Hamiltonian files in the index"""

    from .benchmark.reorder import benchmark as _benchmark

    logging.info("Reading index {}", index_db)
    index = _open_index(
        index_db=index_db,
        prefix=prefix,
        min_qubits=min_qubits,
        max_qubits=max_qubits,
        min_terms=min_terms,
        max_terms=max_terms,
    )

    logging.info("Starting benchmark")
    df = _benchmark(
        index=index,
        ham_dir=ham_dir,
        output_dir=output_dir,
        n_trials=n_trials,
        n_jobs=n_jobs,
        methods=methods.split(","),
    )

    results_db = output_dir / "results.db"
    logging.info("Writing results to: {}", results_db)
    db = sqlite3.connect(results_db)
    df.to_sql("results", db, if_exists="replace", index=True)
    db.close()


@main.command()
@click.argument("ham_dir", type=Path)
@click.argument("reorder_result_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option(  # --prefix
    "--prefix",
    type=str,
    default="",
    help=(
        "Filter the Hamiltonians to benchmark by prefix. Example: "
        "'binaryoptimization/maxcut'"
    ),
)
@click.option("--n-trials", type=int, default=1, help="Defaults to 1")
@click.option("--n-jobs", type=int, default=8, help="Defaults to 8")
@click.option(  # --methods
    "--methods",
    type=str,
    default="none,saturation",
    help=(
        "Comma-separated list of reordering methods to benchmark. Defaults to "
        "'none,saturation'"
    ),
)
@click.option("--min-qubits", type=int, default=0, help="Defaults to 0")
@click.option(  # --max-qubits
    "--max-qubits",
    type=int,
    default=256,
    help="Defaults to 256. Set to 0 to disable",
)
@click.option("--min-terms", type=int, default=0, help="Defaults to 0")
@click.option(  # --max-terms
    "--max-terms",
    type=int,
    default=10000,
    help="Defaults to 10000. Set to 0 to disable",
)
@click.option(  # --qaoa-provider
    "--qaoa-provider",
    type=str,
    default="fez",
    help="Fake provider to use for QAOA simulation. Defaults to 'fez'",
)
@click.option(  # --qaoa-n-shots
    "--qaoa-n-shots",
    type=int,
    default=512,
    help="Number of shots when estimating Hamiltonian energy. Defaults to 512",
)
@click.option(  # --qaoa-n-steps
    "--qaoa-n-steps",
    type=int,
    default=2,
    help=(
        "Number of QAOA stems. Incurs 2*n_steps classical variables to "
        "optimize. Defaults to 2"
    ),
)
@click.option(  # --qaoa-max-iter
    "--qaoa-max-iter",
    type=str,
    default=128,
    help="Max. number of iteration of the classical optimizer. Defaults to 128",
)
@timed
def benchmark_simulate(
    ham_dir: Path,
    reorder_result_dir: Path,
    output_dir: Path,
    prefix: str,
    n_trials: int,
    n_jobs: int,
    methods: str,
    min_qubits: int,
    max_qubits: int,
    min_terms: int,
    max_terms: int,
    qaoa_provider: str,
    qaoa_n_shots: int,
    qaoa_n_steps: int,
    qaoa_max_iter: int,
) -> None:
    """Runs a benchmark on some or all Hamiltonian files in the index"""

    from .benchmark.simulate import benchmark as _benchmark

    p = reorder_result_dir / "results.db"
    logging.info("Reading reordering benchmark results {}", p)
    clauses = [
        f"`hid` LIKE '{prefix}%'",
        (
            "( "
            + " OR ".join([f"`method` = '{m}'" for m in methods.split(",")])
            + " )"
        ),
        f"`n_qubits` >= {min_qubits}",
        f"`n_qubits` <= {max_qubits}",
        f"`n_terms` >= {min_terms}",
        f"`n_terms` <= {max_terms}",
    ]
    query = "SELECT * FROM `results` WHERE " + " AND ".join(clauses)
    logging.debug("Query: {}", query)
    db = sqlite3.connect(p)
    rrdf = pd.read_sql(query, db)
    db.close()

    # Regroup reordering results by parameter and take the shallowest one
    rrdf = (
        rrdf.sort_values("depth")
        .groupby(["hid", "trotterization", "n_timesteps", "order", "method"])
        .first()
        .reset_index()
    )
    logging.info("Number of Hamiltonians: {}", len(rrdf))

    logging.info("Starting benchmark")
    qaoa_config = {
        "max_iter": qaoa_max_iter,
        "n_shots": qaoa_n_shots,
        "n_steps": qaoa_n_steps,
        "provider": qaoa_provider,
        "seed": 0,
    }
    df = _benchmark(
        ham_dir=ham_dir,
        reorder_result=rrdf,
        reorder_result_dir=reorder_result_dir,
        output_dir=output_dir,
        n_trials=n_trials,
        n_jobs=n_jobs,
        qaoa_config=qaoa_config,
    )

    p = output_dir / "results.db"
    logging.info("Writing results to: {}", p)
    db = sqlite3.connect(p)
    df.to_sql("results", db, if_exists="replace", index=True)
    db.close()


@main.command()
@click.argument("index_db", type=Path)
@click.option(  # --hamlib-url"
    "--hamlib-url",
    default=HAMLIB_URL,
    help="Defaults to " + HAMLIB_URL,
    type=str,
)
@timed
def build_index(hamlib_url: str, index_db: Path) -> None:
    """
    Builds the index of all Hamiltonian files in the HamLib website and writes
    it to a SQLite database, under table `index`.
    """
    from .hamlib import build_index as _build_index

    logging.info("Building index of Hamiltonian files in: {}", hamlib_url)
    index_db.parent.mkdir(parents=True, exist_ok=True)
    df = _build_index(hamlib_url)

    logging.info("Writing index to: {}", index_db)
    db = sqlite3.connect(index_db)
    df.to_sql("index", db, if_exists="replace", index=True)
    db.close()


@main.command()
@click.argument("output_dir", type=Path)
@timed
def consolidate(output_dir: Path) -> None:
    """
    Consolidates benchmark results .json files in OUTPUT_DIR/jobs into a single
    .csv file in OUTPUT_DIR/results.csv.
    """

    from .benchmark.consolidate import consolidate as _consolidate

    logging.info("Consolidating benchmark results in: {}", output_dir)
    df = _consolidate(output_dir / "jobs")

    results_db = output_dir / "results.db"
    logging.info("Writing results to: {}", results_db)
    db = sqlite3.connect(results_db)
    df.to_sql("results", db, if_exists="replace", index=True)
    db.close()


@main.command()
@click.argument("index_db", type=Path)
@click.argument("output_dir", type=Path)
@click.option(  # --prefix
    "--prefix",
    type=str,
    default="",
    help=(
        "Filter the Hamiltonians to benchmark by prefix. Example: "
        "'binaryoptimization/maxcut'"
    ),
)
@click.option(  # --hamlib-url
    "--hamlib-url",
    default=HAMLIB_URL,
    help="Defaults to " + HAMLIB_URL,
    type=str,
)
@timed
def download(
    index_db: Path, output_dir: Path, prefix: str, hamlib_url: str
) -> None:
    """Downloads some or all Hamiltonian files in the index"""

    from .hamlib import download as _download

    index = _open_index(index_db, prefix)
    _download(index, output_dir, hamlib_url, prefix)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
