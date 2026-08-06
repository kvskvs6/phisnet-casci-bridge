# Fock Matrix Extractor Pipeline for OpenMolcas

## Overview
This Python script serves as an automated computational chemistry pipeline designed to parse Nudged Elastic Band (NEB) multi-geometry trajectory files, orchestrate sequential OpenMolcas electronic structure calculations, and compute real-space generalized Fock matrices from molecular orbital parameters. 

## Details of the Computation
The workflow operates in four sequential phases:
1. **Geometry Extraction:** Parses a composite `.xyz` trajectory file (e.g., `Path_Step_0010.xyz`), excising force vectors to isolate strict coordinate matrices (Element, X, Y, Z). The output structures are distributed into a standardized hierarchical directory tree.
2. **Sequential OpenMolcas Execution:** Triggers self-consistent field (SCF) calculations for defined basis sets (`MB`, `VDZP`). The pipeline implements a deterministic lateral shift heuristic (orbital propagation) to expedite SCF convergence: it automatically extracts the converged `.RasOrb` wave function from step *N-1* and symlinks it as the initial orbital guess for step *N*.
3. **Fock Matrix Reconstruction:** Reads overlap matrices ($S$) via the HDF5 output and molecular orbital coefficients/energies ($C, \epsilon$) from the `.RasOrb` files. The generalized Fock matrix ($F$) is mathematically reconstructed utilizing Roothaan's equations ($F = S C \epsilon C^{-1}$) and serialized to `.npy` arrays.
4. **Data Logging:** Synthesizes metadata record files (`used_data_record_*.txt`) detailing test set index distributions and operational timestamps to ensure reproducibility.

## Usage
**Required Input Files (Working Directory):**
*   `Path_Step_0010.xyz`: The composite multi-molecule structural trajectory.
*   `open_g2m05_0000_<basis>.RasOrb`: The baseline starting orbital guess file for propagation.
*   `template_<basis>.input`: The OpenMolcas template command file.

**Generated Outputs:**
*   **Coordinates:** `dataset_NEB/open/open_g2m05_<idx>/open_g2m05_<idx>.xyz`
*   **Fock Matrices:** `dataset_NEB/open/open_g2m05_<idx>/<basis>/fock_real.npy`
*   **Logs:** `used_data_record_1_1_<N>_<basis>.txt` (situated in the root working directory)

## Example
Execute the standalone pipeline directly from the working root. No supplementary flags are required.
```bash
python run_fock_extraction.py
```

## Conclusions
The deployment of this script yields a systematically correlated dataset of $F$-matrices aligned against subtly perturbed non-equilibrium molecular geometries. It validates sequential orbital guessing strategies along reaction coordinates, and the resulting quantum mechanical data serves as foundational groundwork for training or benchmarking out-of-distribution generalized errors in machine learning potentials (MLPs) or surrogate deep learning operators.
