from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger as logging
from qiskit import qpy
from tqdm import tqdm

from pcb.utils import timed


def job(p: Path) -> None:
    """Compresses one QPY file"""
    try:
        with p.open("rb") as fp:
            qpy.load(fp)
    except Exception as e:
        logging.info(f"Cleaning up {p}: {e}")
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
        p.with_suffix(".hdf5").unlink(missing_ok=True)


@timed
def main() -> None:
    jobs = [
        delayed(job)(p)
        for p in tqdm(
            Path("out/reordering/jobs").glob("**/*.qpy"), desc="Listing jobs"
        )
    ]
    executor = Parallel(n_jobs=64, backend="loky", verbose=1)
    executor(jobs)


if __name__ == "__main__":
    main()
