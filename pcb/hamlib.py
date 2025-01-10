"""
Function related to finding and downloading stuff from the HamLib website.

See also:
    https://portal.nersc.gov/cfs/m888/dcamps/hamlib/
"""

import zipfile
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from urllib.parse import urljoin

import h5py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger as logging

HAMLIB_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/"


def _all_hamiltonians_urls(csv_urls: list[str]) -> list[str]:
    """
    From the URL list of all the index CSVs from `all_index_csv_urls`, this
    returns the list of all the Hamiltonian file URLs.
    """
    logging.debug("Listing all Hamiltonian file URLs")
    urls: list[str] = []
    for url in csv_urls:
        df = pd.read_csv(url)
        lst = [urljoin(url, file) for file in df["File"].unique()]
        urls.extend(lst)
        logging.debug("Found {} Hamiltonian files in: {}", len(lst), url)
    logging.debug("Found a total of {} Hamiltonian files", len(urls))
    return urls


def _all_index_csv_urls(base_url: str) -> list[str]:
    """
    Hamiltonian files in a directory are indexed by a CSV file. This function
    finds all such files in the HamLib website.
    """
    logging.debug("Finding all index CSVs in the HamLib website: {}", base_url)
    dir_urls, csv_urls = deque([base_url]), []
    while dir_urls:
        url = dir_urls.pop()
        logging.debug("Exploring: {}", url)
        page = requests.get(url).text
        bs = BeautifulSoup(page, "html.parser")
        for tag in bs.find_all("a"):
            if not tag.has_attr("href"):
                continue
            if (href := tag["href"]).endswith(".csv"):
                csv_urls.append(urljoin(url, href))
                logging.debug("Found: {}", csv_urls[-1])
    logging.debug("Found a total of {} index CSVs", len(csv_urls))
    return csv_urls


def all_hamiltonian_files() -> Generator[tuple[h5py.File, str], None, None]:
    """
    Find all the Hamiltonian files in the HamLib website. And returns them as an
    iterator over tuples of the form `(h5py.File, hamiltonian_file_id)`.
    """
    urls = _all_hamiltonians_urls(_all_index_csv_urls(HAMLIB_URL))
    for url in urls:
        with TemporaryDirectory() as tmp:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            path = Path(tmp) / url.split("/")[-1]
            with path.open("wb") as fp:
                fp.write(response.content)
            with zipfile.ZipFile(path, mode="r") as fp:
                assert len(fp.namelist()) == 1
                name = fp.namelist()[0]
                hfid = url[len(HAMLIB_URL) : -len(".zip")]
                with fp.open(name, "r") as h5fp:
                    with h5py.File(h5fp, "r") as h5fp:
                        yield h5fp, hfid
