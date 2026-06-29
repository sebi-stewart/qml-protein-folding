# QML Protein Folding

Quantum approximate optimisation (QAOA) applied to protein rotamer selection. Given a protein structure, the pipeline extracts pairwise interaction energies between sidechain rotamers, encodes them as an Ising Hamiltonian, runs QAOA to find low-energy rotamer assignments, then validates those assignments against PyRosetta's biological scoring function.

The target protein in the current dataset is BPTI (PDB: 5PTI), with energy mappings pre-extracted across residue segments of 5–22 qubits.

---

## Repository Structure

```
qml_protein_folding/
├── extraction/          # Phase 1a: rotamer extraction and energy tensor generation
├── qaoa/               # Phase 1b: QAOA solver
├── phase2/             # Phase 2: biological validation and rescoring
├── utils/              # Shared utilities (logging, path handling, plotting)
├── inputs/             # Input PDB files
├── intermediates/      # JSON energy mappings (extraction outputs)
├── outputs/            # NPZ results, Parquet dataframes, logs, PDB poses
├── jupyter_notebooks_data_analysis/
├── publication_plots/
└── qaoa_colab.ipynb    # Colab-compatible notebook for running the QAOA stage
```

---

## Modules

### 1. Extraction

Extracts one- and two-body interaction energies from a protein structure and serialises them as JSON tensors ready for the QAOA solver.

**Inputs:**
- A `.pdb` structure file (e.g. `inputs/structural_files/AF-P00974-F1-model_v6.pdb`)
- Residue range and rotamer count parameters (see CLI args below)

**Processing:**
1. Loads the structure via PyRosetta and creates a packing task restricted to the specified residue window.
2. Enumerates the top-N rotamers per residue using PyRosetta's rotamer set, filtering near-duplicates by chi angle difference (threshold: 40°).
3. Builds a two-body interaction graph via `InteractionGraphFactory` and extracts one-body (self-energy) and two-body (pairwise interaction) tensors.
4. Reduces the Hamiltonian by absorbing fixed-residue energies into a global offset, retaining only flexible–flexible interactions.
5. Saves sparse tensors as nested JSON dicts, organised into subdirectories by total qubit count.

**Outputs:**
```
intermediates/energy_mappings/{dataset_name}/{num_qubits}/{protein}_{res_start}_{res_end}_{rot_count}.json
```

JSON structure:
```json
{
  "one_body":  { "<residue_seq>": { "<rotamer_idx>": <energy>, ... }, ... },
  "two_body":  { "<seq_i>,<seq_j>": { "<rot_i>,<rot_j>": <energy>, ... }, ... }
}
```

Keys are serialised as strings (JSON requirement) and cast back to integers on load.

---

### 2. QAOA

Runs layered QAOA over the energy mapping files produced by the extraction stage.

**Encoding:**
Each residue gets a bundle of qubits in a one-hot arrangement — one qubit per retained rotamer, exactly one active per residue. A segment with rotamer counts `[4, 3, 2]` uses 9 qubits with wire offsets `{res_A: 0, res_B: 4, res_C: 7}`.

**Circuit:**
- Initial state: uniform superposition over all valid one-hot configurations, prepared with `StatePrep` (not random initialisation).
- Cost layer: Pauli-Z Hamiltonian derived from the Ising substitution of the interaction tensors.
- Mixer layer: ring-XY interactions within each rotamer bundle, preserving the one-hot constraint.
- Parameters: γ (cost) and β (mixer) per layer, optimised with Adam (via Optax).

**Layered run:**
QAOA depths are swept in order (default: 2 → 4 → 6 → 8 → 12 layers). Each depth warm-starts from the previous layer's parameters (padded or trimmed as needed). The step size is divided by `STEPSIZE_REDUCTION_FACTOR` for each subsequent depth.

**Execution modes:**
- `batched`: vectorises over seeds using JAX/Catalyst JIT. Batch size is capped based on an estimated memory footprint (~16 bytes per complex state vector amplitude) and a compiler cap of 6 for circuits above 14 qubits.
- `sequential`: runs each seed individually; lower memory use, slower.

**Outputs:**
```
outputs/npz_files/{dataset}/{protein}_{res_start}_{res_end}_{rot_count}_{layers}_layers.npz
```

Each NPZ contains:
- `optimized_params`: shape `(num_seeds, 2, num_layers)` — final γ and β per seed
- `target_probs`: probability of sampling the known low-energy bitstrings per seed

**Key configuration (all overridable via CLI):**

| Constant                    | Default               | Description                           |
|-----------------------------|-----------------------|---------------------------------------|
| `QUBIT_COUNTS_TO_ANALYSE`   | `[5, 10, 14, 18, 22]` | Which qubit counts to process         |
| `LIMIT_FILES_PER_QUBIT`     | `10`                  | Max files per qubit count             |
| `QAOA_LAYERS`               | `[2, 4, 6, 8, 12]`    | Layer depths to sweep                 |
| `BASE_EPOCHS`               | `150`                 | Optimiser iterations per depth        |
| `BASE_STEPSIZE`             | `0.01`                | Initial Adam step size                |
| `STEPSIZE_REDUCTION_FACTOR` | `10`                  | Step size divisor between depths      |
| `QAOA_SEED_COUNT`           | `30`                  | Number of random seeds                |
| `QAOA_EXECUTION_MODE`       | `"batched"`           | `"batched"` or `"sequential"`         |
| `QAOA_MAX_MEMORY_GB`        | `12.0`                | Memory cap for batch size calculation |
| `USE_GPU`                   | `False`               | Use `lightning.gpu` backend           |
| `HIGH_TO_LOW_QUBIT_ORDER`   | `True`                | Process largest qubit counts first    |

---

### 3. Phase 2

Validates QAOA solutions against PyRosetta's biological scoring function.

**Workflow:**
1. Loads a QAOA result NPZ (default pattern: `*_12_layers.npz`) and re-runs the QAOA sampler with the stored optimised parameters to collect bitstring samples across all seeds.
2. Deduplicates bitstrings, maps each to a rotamer assignment using the original residue library, constructs a cloned PyRosetta `Pose`, and scores it with the biological energy function.
3. Compares each score against the original pose baseline and classifies the result:
   - `win`: energy difference ≤ −1.5 REU
   - `tie`: within ±1.5 REU
   - `loss`: energy difference > +1.5 REU
4. Optionally, exhaustively evaluates all valid one-hot bitstrings (feasible for small qubit counts).

**Outputs:**
```
outputs/dataframes/{name}.parquet          # per-bitstring results
outputs/structural_files/best_pose_{name}.pdb
outputs/structural_files/worst_conformation_{name}.pdb
```

Parquet columns: `seed`, `bitstring`, `biological_energy`, `energy_diff`, `protein`, `residues`, `residue_count`, `rotamers`, `num_qubits`, `classification`.

---

## Installation

```bash
git clone <repository-url>
cd <repository-root>
```

```bash
pip install -r requirements.txt
```

**PyRosetta:**

PyRosetta installations can be quite tricky compared to standard Python packages, in particular for 
systems which aren't Linux. The `requirements.txt` file includes a placeholder for PyRosetta, but it is not automatically installed. You must download the appropriate PyRosetta wheel for your platform and Python version from the [PyRosetta website](https://www.pyrosetta.org/downloads) and install it manually using pip:

**GPU support (Linux only):**

`pennylane-lightning[gpu]` and the CUDA dependencies in `requirements.txt` are Linux-only and install automatically on that platform. On macOS, GPU execution is not available; set `USE_GPU=False`.

---

## Usage

All three stages are run from the **project root**.

### Extraction

```bash
python extraction/main.py \
  --test_name AF-5PTI_moderate_confidence_region \
  --input_pdb inputs/structural_files/AF-P00974-F1-model_v6.pdb \
  --output_dir intermediates/energy_mappings \
  --min_residue_length 3 --max_residue_length 6 \
  --min_start_pos 1 --max_start_pos 58 \
  --min_rot_count 2 --max_rot_count 4
```

### QAOA

```bash
python qaoa/main.py \
  --source-folder intermediates/energy_mappings/AF-5PTI_moderate_confidence_region \
  --qubit-counts-to-analyse 5 10 14 \
  --qaoa-layers 2 4 6 8 12 \
  --base-epochs 150 \
  --qaoa-seed-count 30 \
  --qaoa-execution-mode batched \
  --outputs-folder outputs/npz_files
```

For Google Colab, use `qaoa_colab.ipynb` at the repository root — upload the entire `qaoa` and `utils` modules, set `PROJECT_ROOT`, adjust constants in the configuration cell, and run all cells.
If Google Colab is used, PyRosetta is not required for the QAOA stage, as the energy mapping JSON files can be pre-extracted and uploaded.

### Phase 2

```bash
python phase2/main.py \
  --input_pdb inputs/structural_files/AF-P00974-F1-model_v6.pdb \
  --qaoa_results_folder outputs/npz_files/AF-5PTI_moderate_confidence_region \
  --file_search_pattern "*_12_layers.npz"
```

Add `--exhaustive_evaluation` to enumerate all valid conformations in addition to QAOA-sampled bitstrings.

### Note
The above run commands do not represent an exhaustive list of all available CLI arguments. Use `--help` on any stage to see the full set of options.

---

## Output Files

| Path                                                        | Description                                                  |
|-------------------------------------------------------------|--------------------------------------------------------------|
| `intermediates/energy_mappings/{dataset}/{n_qubits}/*.json` | Sparse Ising tensors, one file per residue segment           |
| `outputs/npz_files/{dataset}/*_{layers}_layers.npz`         | QAOA optimised parameters and target probabilities           |
| `outputs/dataframes/*.parquet`                              | Phase 2 results with biological energies and classifications |
| `outputs/structural_files/best_pose_*.pdb`                  | Lowest-energy conformation found by QAOA                     |
| `outputs/structural_files/worst_conformation_*.pdb`         | Highest-energy conformation sampled                          |
| `outputs/logs/*.log`                                        | Per-run logs with timing and per-file metrics                |

---

## Notes

- The QAOA stage processes qubit counts in descending order by default (`HIGH_TO_LOW_QUBIT_ORDER=True`). This is useful when running multiple parallel instances, as the most expensive problems are started first.
- `START_AT_FILE_IDX` can be used to divide a file list across multiple concurrent runs without overlap.
- The `batched` execution mode compiles a single JIT-compiled function over all seeds in one pass. Above 14 qubits the state vector grows large enough that the compiler cap (6 seeds per batch) dominates the memory limit.
- Phase 2 shot counts scale with problem size: 10 shots for 5–6 qubits up to 400 shots for 22 qubits.
- PyRosetta is only required for the extraction and phase 2 stages. The QAOA stage has no PyRosetta dependency and can run on systems where PyRosetta is unavailable (e.g. Colab), provided the energy mapping JSON files are already present.
- Energy units throughout are Rosetta Energy Units (REU). The ±1.5 REU classification threshold in phase 2 is hardcoded in `phase2/biological_rescoring.py`.
