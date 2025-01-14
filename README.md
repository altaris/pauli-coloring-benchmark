# Pauli Coloring Benchmark

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)

# How to do the thing?

1. Install [`uv`](https://docs.astral.sh/uv/).

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:

   ```sh
   uv sync
   ```

3. Build the index. This is a CSV file that lists all the Hamiltonian ZIP files
   present in the HamLib website.

   ```sh
   uv run python -m pcb build-index out/index.csv
   ```

4. Download the Hamiltonian ZIP files.

   ```sh
   uv run python -m pcb download out/index.csv out/ham
   # it is also possible to apply a filter to only download files in a given subdirectory
   uv run python -m pcb download out/index.csv out/ham --prefix discreteoptimization/tsp
   ```

5. Run the benchmark.

   ```sh
   uv run python -m pcb benchmark out/index.csv out/ham out/results
   # or
   uv run python -m pcb benchmark out/index.csv out/ham out/results --prefix discreteoptimization/tsp
   ```
