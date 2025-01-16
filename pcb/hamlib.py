"""
Function related to finding and downloading stuff from the HamLib website.

See also:
    https://portal.nersc.gov/cfs/m888/dcamps/hamlib/
"""

import zipfile
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin

import h5py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger as logging
from tqdm import tqdm


def _all_csv_urls(base_url: str) -> Generator[str, None, None]:
    """
    Hamiltonian files in a directory are indexed by a CSV file. This function
    iterates over all the URLs of such files in the HamLib website.
    """
    logging.debug("Finding all index CSVs in the HamLib website: {}", base_url)
    dir_urls = deque([base_url])
    while dir_urls:
        url = dir_urls.pop()
        logging.debug("Exploring: {}", url)
        page = requests.get(url).text
        bs = BeautifulSoup(page, "html.parser")
        for tag in bs.find_all("a"):
            if not tag.has_attr("href"):
                continue
            href = tag["href"]
            new_url = urljoin(url, href)
            if href.endswith("/") and "/" not in href[:-1]:
                logging.debug("Found directory: {}", new_url)
                dir_urls.append(urljoin(url, new_url))
            elif href.endswith(".csv"):
                logging.debug("Found CSV file: {}", new_url)
                yield new_url


@contextmanager
def open_hamiltonian_file(
    path: str | Path,
) -> Generator[h5py.File, None, None]:
    """
    A Hamiltonian file is a ZIP archive containing a single HDF5 file, which
    contains essentially a dict of byte strings, each representing a serialized
    sparse Pauli operator. The serialization format looks like this (after
    decoding):

        >>> with open_hamiltonian_file("...") as fp:
        >>>     k = list(fp.keys())[0]
        >>>     print(fp[k][()].decode("utf-8"))
        22.5 [] +
        -0.5 [Z9 Z20] +
        -0.5 [Z9 Z26] +
        -0.5 [Z9 Z29] +
        -0.5 [Z9 Z56] +

    Warning:
        Note the `[()]` when accessing a key!

    """
    with zipfile.ZipFile(path, mode="r") as fp:
        if not (
            len(fp.namelist()) == 1 and fp.namelist()[0].endswith(".hdf5")
        ):
            raise ValueError(
                f"Expected exactly one .hdf5 file in archive {path}, found "
                + ", ".join(fp.namelist())
            )
        name = fp.namelist()[0]
        with fp.open(name, "r") as h5fp:
            with h5py.File(h5fp, "r") as h5fp:
                yield h5fp


def build_index(base_url: str) -> pd.DataFrame:
    """
    Builds a dataframe containing the URLs of all the Hamiltonian files in the
    HamLib website.

    This crawls the website to find all the CSV files. Each of them lists the
    Hamiltonian files in that directory. Then, this method essentially
    concatenates all of them. The returned dataframe has the following columns:
    - `url` of the ZIP file (NOT the HDF5 file),
    - `hfid`: A unique identifier for the Hamiltonian file.
    """
    data = []
    for url in _all_csv_urls(base_url):
        df = pd.read_csv(url)
        df = pd.DataFrame(
            [
                {
                    "url": urljoin(url, file)[: -len(".hdf5")] + ".zip",
                    "hfid": urljoin(url, file)[len(base_url) : -len(".hdf5")],
                }
                for file in df["File"].unique()
            ]
        )
        data.append(df)
    return pd.concat(data, ignore_index=True)


def download(
    index: pd.DataFrame, output_dir: str | Path, prefix: str | None = None
) -> None:
    """
    Downloads all the compressed HDF5 Hamiltonian files in the given index to
    the given output directory. `prefix` can be set to restrict the benchmark to
    a given family of Hamiltonians, for example `binaryoptimization/maxcut`.

    The downloaded files are not decompressed and named after their Hamiltonian
    file ID.

    This function does not distribute the downloads across multiple processes to
    not slam the HamLib website with requests.
    """
    if prefix:
        index = index[index["hfid"].str.startswith(prefix)]
    logging.info(
        "Downloading {} Hamiltonians files to {}", len(index), output_dir
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(index.iterrows(), desc="Downloading", total=len(index))
    for _, row in progress:
        progress.set_postfix_str(row["hfid"])
        path = output_dir / (row["hfid"].replace("/", "__") + ".hdf5.zip")
        if path.is_file():
            logging.debug("Skipping: {}", row["url"])
            continue
        try:
            response = requests.get(row["url"], stream=True, timeout=60)
            response.raise_for_status()
            with path.open("wb") as fp:
                fp.write(response.content)
        except Exception as e:
            logging.error("Failed to download {}: {}", row["url"], e)
