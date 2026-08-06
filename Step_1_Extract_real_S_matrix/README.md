# Overlap Matrix (S Matrix) Extractor

## Overview
This script automates the batch extraction of atomic orbital overlap matrices (S matrices) from OpenMolcas `.rasscf.h5` output files. It iterates through specified basis sets (MB, VDZP) and compiles the extracted matrices into a compressed `.npz` archive for downstream computational chemistry tasks.

## Details of the Computation
The script traverses a nested directory structure within a parent `open` folder, matching the glob pattern `open/open_g2m05_[0-9]{4}/<basis_set>/...h5`. For each matched HDF5 file:
- It parses the unique 4-digit structural identifier directly from the filename.
- It reads the number of basis functions (`NBAS` attribute) to explicitly determine the matrix dimensions.
- It accesses the flattened overlap matrix array from the dataset `AO_OVERLAP_MATRIX` and reshapes it into a 2D square matrix of size `NBAS × NBAS`.
- Missing attributes or corrupted files are handled via `try-except` blocks to ensure robust, uninterrupted batch processing.

## Usage
- **Input:** OpenMolcas `.rasscf.h5` files located in the `open/` directory relative to the script's execution path.
- **Output:** Compressed NumPy archives named `all_S_matrices_<basis_set>.npz` (e.g., `all_S_matrices_MB.npz`) containing a dictionary of S matrices, keyed by their 4-digit identifier (e.g., `'0003'`). These are saved directly in the current working directory.

## Example
Execute the script directly from the directory containing the `open` parent folder:
```bash
python extract_s_matrices.py
```

## Conclusions
By aggregating individual `.rasscf.h5` outputs into unified, basis-set-specific NumPy archives, this script consolidates raw quantum chemistry outputs into a format easily loaded in Python. The resulting datasets prepare the necessary matrix inputs for machine learning models or downstream quantum mechanical workflows requiring overlap matrices, thus enabling deterministic basis alignment, orbital analysis, or generalized eigenvalue problem solving across the dataset.
