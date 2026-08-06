# Molcas Batch Calculation v0.2

## Overview
This Python script automates high-throughput single-point energy calculations using OpenMolcas. Designed to run on HPC clusters via the PBS job scheduler, it automatically aligns molecular structures (`.xyz`) with pre-predicted initial guess orbitals (`.Orb`), sets up isolated calculation directories, and executes array jobs at scale.

## Details of the Computation
- **Path Derivation & Data Discovery:** The script dynamically deduces the root directory based on its execution path. It scans `dataset/open/` to construct a reference set of available structural geometries.
- **Orbital Processing:** It targets compressed predicted orbital files located in the `OpenMolcas_projects` directory. If unextracted, it unzips the files and recursively parses the `.Orb` filenames using regular expressions to extract metadata: basis set, dataset splits (train/valid/test), and structure indices.
- **Task Alignment:** Cross-referencing the parsed orbital indices against the available structure set, the script builds a deterministic list of valid computation tasks.
- **Execution Environment:** Utilizing the `PBS_ARRAY_INDEX` environment variable, the script maps the array job to a specific task. It creates a dedicated output subdirectory, copies the corresponding `.input` template for the parsed basis set, generates symlinks for the heavy structural and orbital files to save disk I/O, and enforces Molcas to retain the working directory (`MOLCAS_KEEP_WORKDIR='YES'`) for post-processing.
- **Hardware Allocation:** Calculates are hardcoded to utilize 16 GB memory and 4 CPU threads via Molcas internal environment variables.

## Usage
**Required Directory Structure and Inputs:**
- **Structures:** `dataset/open/open_g2m05_XXXX/open_g2m05_XXXX.xyz`
- **Orbitals:** Zipped `.Orb` archives located in `OpenMolcas_projects/_own_testset_predicted_Orb_files/` (will be automatically extracted to `./extracted/`).
- **Templates:** Base calculation templates (e.g., `template_VDZP.input`) placed in the same directory as the script.

**Outputs:**
- All calculations are processed in the `Output/` directory within the script's path.
- **Format:** `Output/<basis>_<train>_<valid>_<test>_<index>/`
- Output directories will contain `molcas.input`, symlinked `structure.xyz` and `orbitals.Orb`, and the standard Molcas outputs (wavefunction logs, stdout).

## Example
This script is designed to be submitted directly to a PBS workload manager as an array job:
```bash
# Submit the array job for tasks 1 through 400
qsub molcas_batch_calc_v0.2.py
```
Alternatively, for local testing with a mocked environment variable:
```bash
export PBS_ARRAY_INDEX=1
python molcas_batch_calc_v0.2.py
```

## Conclusions
Executing this pipeline enables the systematic evaluation of AI-predicted initial guess orbitals across diverse molecular configurations and basis sets. The output energies and self-consistent field (SCF) metrics derived from these batch calculations are crucial for quantifying the generalization error of orbital prediction models across varying dataset distributions (train vs. validation vs. test sets). Ultimately, this validates the utility of machine-learning-derived orbitals in accelerating SCF convergence in quantum chemistry computations.
