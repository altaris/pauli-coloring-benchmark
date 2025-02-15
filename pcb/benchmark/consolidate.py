"""
Function to consolidate the (MANY) job result JSON files in a `output_dir/jobs`
into a single SQLite database
"""

from pathlib import Path

import pandas as pd
from loguru import logger as logging
from tqdm import tqdm

from .utils import flatten_dict, load


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
            if file.stat().st_size == 0:
                logging.warning("Removing empty file: {}", file)
                file.unlink(missing_ok=True)
                continue
            data = flatten_dict(load(file))
            data["jid"] = file.stem
            rows.append(data)
        except Exception as e:
            logging.error("Error reading {}: {}", file, e)
    results = pd.DataFrame(rows)
    results.set_index("hid", inplace=True)
    logging.info("Consolidated {} job results", len(results))
    return results
