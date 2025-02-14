"""
Function to consolidate the (MANY) job result JSON files in a `output_dir/jobs`
into a single SQLite database
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger as logging
from tqdm import tqdm

from .utils import flatten_dict


def consolidate(jobs_dir: str | Path) -> pd.DataFrame:
    """
    Gather all the output JSON files produced by `_bench_one` into a single
    dataframe
    """
    jobs_dir, rows = Path(jobs_dir), []
    progress = tqdm(
        jobs_dir.glob("**/*.json"), desc="Consolidating", leave=False
    )
    for file in progress:
        try:
            with open(file, "r", encoding="utf-8") as fp:
                data = flatten_dict(json.load(fp))
                data["jid"] = file.stem
                rows.append(data)
        except json.JSONDecodeError as e:
            logging.error("Error reading {}: {}", file, e)
    results = pd.DataFrame(rows)
    results.set_index("hid", inplace=True)
    logging.info("Consolidated {} job results", len(results))
    return results
