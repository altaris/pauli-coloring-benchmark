"""CLI module"""

import os
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
@click.argument("output_csv", type=click.File("w"))
@click.option(
    "--hamlib-url",
    default=HAMLIB_URL,
    help="Defaults to " + HAMLIB_URL,
    type=str,
)
def build_index(hamlib_url: str, output_csv: TextIO) -> None:
    """Builds the index of all Hamiltonian files in the HamLib website."""
    from datetime import datetime

    from .hamlib import build_index

    logging.info("Building index of Hamiltonian files in: {}", hamlib_url)
    start = datetime.now()
    df = build_index(hamlib_url)
    df.to_csv(output_csv, index=False)
    logging.info("Done in: {}", datetime.now() - start)


@main.command()
@click.argument("index_csv", type=click.File("r"))
@click.argument("output_csv", type=click.File("w"))
@click.option("--prefix", type=str, default="")
@click.option("--n-trials", type=int, default=10)
def run_benchmark(
    index_csv: TextIO, output_csv: TextIO, prefix: str, n_trials: int
) -> None:
    import pandas as pd

    from .benchmark import benchmark

    index = pd.read_csv(index_csv)
    df = benchmark(index, n_trials=n_trials, prefix=prefix)
    df.to_csv(output_csv, index=False)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
