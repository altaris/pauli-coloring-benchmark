# Pauli Coloring Benchmark

![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-ruff-yellow?logo=ruff)](https://docs.astral.sh/ruff/)
[![Documentation](https://img.shields.io/badge/Documentation-here-pink)](https://cedric.hothanh.fr/pauli-coloring-benchmark/pcb.html)

## How to do the thing?

### Prep. work

1. Install [`uv`](https://docs.astral.sh/uv/).

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:

   ```sh
   uv sync
   ```

3. Build binaries (requires `gcc`). This is only required if you want to use the
   `degree_c` reordering method.

   ```sh
   make dll
   ```

4. Build the index. This is a SQlite3 file that lists all the Hamiltonian from
   all the `hdf5` files present in the HamLib website.

   ```sh
   uv run python -m pcb build-index out/index.db
   ```

5. Download the Hamiltonian ZIP files.

   ```sh
   uv run python -m pcb download out/index.db out/ham
   # it is also possible to apply a filter to only download files in a given subdirectory
   uv run python -m pcb download out/index.db out/ham --prefix binaryoptimization/maxcut
   ```

### Reordering benchmark

1. Run the benchmark.

   ```sh
   uv run python -m pcb benchmark-reorder out/index.db out/ham out/reordering
   # or
   uv run python -m pcb benchmark-reorder out/index.db out/ham out/reordering --prefix binaryoptimization/maxcut --n-trials 1 --n-jobs 200 --methods none,saturation,misra_gries --min-qubits 32 --max-qubits 127 --min-terms 16 --max-terms 256
   ```

2. You can obtain temporary consolidated results during the benchmark by running:

   ```sh
   uv run python -m pcb consolidate out/reordering
   ```

   This is automatically done at the end of the benchmark.

## More help

```sh
uv run python -m pcb --help
```
