[![Repository](https://img.shields.io/badge/-repository-black?logo=github)](https://github.com/altaris/pauli-coloring-benchmark)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14934475.svg)](https://doi.org/10.5281/zenodo.14934475)

This dataset contains results of various intermediate steps from the experiment
pipeline of [`pcb`](https://github.com/altaris/pauli-coloring-benchmark).

## Decompress

```sh
tar -xvzf pcb-maxcut.tar.xz -C pcb-maxcut
```

## Structure of the dataset

- `index.db`: SQLite database indexing the Hamiltonian and Hamiltonian files
  (compressed `.hdf5`) available in the [HamLib
  dataset](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/). It has the
  following columns:
  - `hid`: A unique ID that identifies the Hamiltonian in the dataset. It is
    composed of the path to the `.zip` file containing it in the HamLib website
    (relative to the root URL), a slash, and the key inside the `hdf5` file. For
    example, Hamiltonian
    `binaryoptimization/maxcut/biqmac/rudy-hams/9-pw09_100.0-12` is located at
    key `9-pw09_100.0-12` of `rudy-hams.zip` in folder
    https://portal.nersc.gov/cfs/m888/dcamps/hamlib/binaryoptimization/maxcut/biqmac/.
  - `n_qubits`: Self-explanatory.
  - `n_terms`: Self-explanatory.
- `ising/`: Representation of the MaxCut Hamiltonians of HamLib in terms of
  interaction arrays rather than Pauli operators. The files are uncompressed
  `.hdf5` files with names that ressemble the URL of the corresponding file in
  the HamLib website. A file in `ising/` has the same keys as its corresponding
  HamLib file. Each key is mapped to a NumPy array of shape `(n, 2)` where `n`
  is the number of terms in the corresponding Ising Hamiltonian. For example, $H
  = -1/2 (Z_2 Z_4 + Z_1 Z_4 + Z_1 Z_2)$ will correspond to `[[2, 4], [1, 4], [1,
2]]`.
- `reorder.hcs=m/`: Results of the reordering-Trotterization benchmark for
  MaxCut Hamiltonians, which have form $-1/2 \sum_{(i, j) \in E} Z_i Z_j$.
  - `results.db`: SQLite database containing the reordering results,
    consolidated from `reorder.hcs=m/jobs/**/*.json`. The columns are the same
    as the keys in the JSON files (see below).
  - `jobs/`
    - `job/XX/XX/<UUID4>.json`: (this UUID4 is called the _reordering job ID_
      and referred to as `reordering_jid`) The results of a single reordering trial. For
      example:
      ```json
      {
        "method": "misra_gries",
        "n_terms": 58,
        "n_qubits": 77,
        "n_timesteps": 1,
        "order": 4,
        "trotterization": "suzuki_trotter",
        "hid": "binaryoptimization/maxcut/biqmac/rudy-hams/9-pw09_100.0-12",
        "depth": 120,
        "reordering_time": 4.739,
        "synthesis_time": 3.831
      }
      ```
    - `job/XX/XX/<UUID4>.qpy.gz`: The Trotterized Hamiltonian, serialized using
      the [`qpy` binary format](https://docs.quantum.ibm.com/api/qiskit/qpy) and
      compressed with `gzip`.
    - `job/XX/XX/<UUID4>.hdf5`: Contains 2 keys:
      - `coloring`: The 1D coloring array, where entry `i` is the color of term
        `i`, represented as an non-negative integer. The array is a
        `float` array for some reason...
      - `term_indices`: The index of a given term inside the reordered
        Hamiltonian, i.e. if `term_indices[i] = j`, then the `i`-th term in the
        non-reordered Hamiltonian is placed at position `j` in the reordered one.
- `reorder.hcs=p/`: Similar to `reorder.hcs=m/` for the same Hamiltonians but
  with their global phase flipped, i.e. $1/2 \sum_{(i, j) \in E} Z_i Z_j$. These
  results are not used in subsequent steps.
- `run.aer.hm=p,hc=m,max/`: QAOA results using the Aer simulator of Qiskit.
  - `results.db`: SQLite database containing the reordering results,
    consolidated from `run.aer.hm=p,hc=m,max/jobs/**/*.json`. The columns are
    the same as the keys in the dicts below.
  - `jobs/`
    - `job/XX/XX/<UUID4>.json`: (this UUID4 is called the _simulation job ID_
      and referred to as `simulation_jid`) The QAOA steps results as a list of
      flat dicts. For example:
      ```json
      [
        {
          "energy": -0.02141564613122384,
          "step": 0,
          "ibmq_jid": "98c9b2ae-80ab-4e20-a34c-f31484bfa545",
          "ibmq_sid": null,
          "backend": "fake_kawasaki",
          "hid": "binaryoptimization/maxcut/ciqube/Karloff-hams/9-Karloff_6_3_1.txt-6",
          "reordering_jid": "ca763ff9542a57e6a0cea5fab657908e380bd172"
        },
        ...
      ]
      ```
      All entries in the array have the same structure. The `ibmq_jid` and
      `ibmq_jid` correspond to the IBMQ job ID and session ID. Since a simulator
      was used instead of actual quantum hardware, the job ID is just some
      random UUID4 and the session ID is always `null`. The `reordering_jid` is
      the job ID of the reordering that gave the Hamiltonian that underwent this
      QAOA trial.
    - `job/XX/XX/<UUID4>.hdf5`: Contains 4 keys:
      - `all_energies`: 1D `(N,)` array of expectation values of the Hamiltonian
        at each iteration of the QAOA algorithm, where `N` is the number of
        iterations.
      - `all_parameters`: 2D `(N, m)` array of parameters of the QAOA algorithm
        at each iteration. `m` is the number of parameters, i.e. the number of
        $\beta_j$'s and $\gamma_j$'s.
      - `best_energy`: Self-explanatory. This is a `(1,)` array rather than a
        scalar because `hdf5py` crashes otherwise 😬
      - `best_parameters`: Self-explanatory. This is a `(m,)` array.
- `smpl.hm=p,hc=m,max/`
  - `results.db`: SQLite database containing the sampling experiment metadata.
    Note that it does not actually contain the sample counts; these need to be
    consolidated manually from all the `job/**/*.csv`. The columns are the same
    as the keys in the JSON files (see below).
  - `jobs/`
    - `job/XX/XX/<UUID4>.json`: (this UUID4 is called the _simulation job ID_
      and referred to as `simulation_jid`). Metadata of a sampling job. Example:
      ```json
      {
        "hid": "binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7",
        "n_shots": 1024,
        "reordering_jid": "8bef4ac76fab1d163e955313ad3c92d251188c83",
        "sampling_jid": "001a9dc0828d0c33eaebe2a93f7448f827d25d56",
        "simulation_jid": "7d1bcaeb39523419778d350ac1fb31ef9c285fe6",
        "ibmq_jid": "cz600c110wx0008brqqg",
        "ibmq_sid": null,
        "backend": "ibm_kawasaki"
      }
      ```
    - `job/XX/XX/<UUID4>.qpy.gz`: A cached ISA QAOA ansatz, i.e. a QAOA ansatz
      passed through a pass manager.
    - `job/XX/XX/<UUID4>.csv`: Self-explanatory. For example
      ```csv
      ,string,count,hid
      0,0001001,44,binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7
      1,1001100,22,binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7
      2,1110000,36,binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7
      3,0110101,73,binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7
      4,0100001,75,binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7
      5,0011110,14,binaryoptimization/maxcut/biqmac/rudy-hams/3-g05_60.5-7
      ...
      ```
      The HID is the same for every row. It is technically redundant but it
      makes some data analysis tasks a bit easier. Some useless `Unnamed 0` or
      `Unnamed 0.1` columns might be present.
