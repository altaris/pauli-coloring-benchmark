import gzip
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
            qc = qpy.load(fp)
        with gzip.open(p.with_suffix(".qpy.gz"), "wb") as fp:
            qpy.dump(qc, fp)
        p.unlink(missing_ok=True)
    except Exception as e:
        logging.error(f"Failed to compress {p}: {e}")


@timed
def main() -> None:
    jobs = [
        delayed(job)(p)
        for p in tqdm(
            Path("out/reordering/jobs").glob("**/*.qpy"), desc="Listing jobs"
        )
    ]
    executor = Parallel(n_jobs=128, backend="loky", verbose=1)
    executor(jobs)


if __name__ == "__main__":
    main()
