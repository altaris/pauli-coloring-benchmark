"""CLI module"""

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
@click.argument("index_csv", type=click.File("r"))
@click.argument("input_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option("--prefix", type=str, default="")
@click.option("--n-trials", type=int, default=10)
@click.option("--n-jobs", type=int, default=32)
def benchmark(
    index_csv: TextIO,
    input_dir: Path,
    output_dir: Path,
    prefix: str,
    n_trials: int,
    n_jobs: int,
) -> None:
    from datetime import datetime

    import pandas as pd

    from .benchmark import benchmark as _benchmark

    index = pd.read_csv(index_csv)
    start = datetime.now()
    df = _benchmark(
        index=index,
        input_dir=input_dir,
        output_dir=output_dir,
        n_trials=n_trials,
        prefix=prefix,
        n_jobs=n_jobs,
    )
    df.to_csv(output_dir / "results.csv", index=False)
    logging.info("Done in: {}", datetime.now() - start)


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
@click.argument("output_dir", type=Path)
@click.option("--prefix", type=str, default="")
def download(index_csv: TextIO, output_dir: Path, prefix: str) -> None:
    """Downloads all Hamiltonian files in the index."""
    from datetime import datetime

    import pandas as pd

    from .hamlib import download as _download

    start = datetime.now()
    index = pd.read_csv(index_csv)
    _download(index, output_dir, prefix)
    logging.info("Done in: {}", datetime.now() - start)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
