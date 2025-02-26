from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger as logging
from tqdm import tqdm

from pcb.benchmark.utils import load, save
from pcb.utils import timed


def job(p: Path) -> None:
    """Compresses one HDF5 file"""
    try:
        save(load(p), p)
    except Exception as e:
        logging.error(f"Failed to compress {p}: {e}")


@timed
def main() -> None:
    jobs = [
        delayed(job)(p)
        for p in tqdm(
            Path("out/reordering/jobs").glob("**/*.hdf5"), desc="Listing jobs"
        )
    ]
    executor = Parallel(n_jobs=64, backend="loky", verbose=1)
    executor(jobs)


if __name__ == "__main__":
    main()
