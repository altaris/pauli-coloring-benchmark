"""
Function to consolidate the (MANY) job result JSON files in a `output_dir/jobs`
into a single SQLite database
"""

from pathlib import Path

import pandas as pd
from loguru import logger as logging
from tqdm import tqdm

from ..io import load
from ..utils import flatten_dict


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
            data = load(file)
            if isinstance(data, dict):
                data = flatten_dict(data)
                data["jid"] = file.stem
                rows.append(data)
            elif isinstance(data, (list, tuple)):
                for item in data:
                    item = flatten_dict(item)
                    item["jid"] = file.stem
                    rows.append(item)
            else:
                logging.warning(
                    "Unexpected data type {} from file {}. Skipping",
                    type(data),
                    file,
                )
        except Exception as e:
            logging.error("Error reading {}: {}", file, e)
    if not rows:
        logging.warning("No valid JSON files found in {}", jobs_dir)
        return pd.DataFrame()
    results = pd.DataFrame(rows)
    if "hid" in results.columns:
        results.set_index("hid", inplace=True)
    logging.info(
        "Consolidated {} rows from {} jobs",
        len(results),
        len(results["jid"].unique()),
    )
    return results
