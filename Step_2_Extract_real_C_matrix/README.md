# Extract True C Matrices

## Overview
This script is designed to batch-extract true Molecular Orbital (MO) coefficients (C matrices) from OpenMolcas `.RasOrb` files for computational chemistry datasets. It processes multiple geometric configurations across specific basis sets (MB and VDZP) and bundles the extracted matrices into compressed Numpy archives for downstream machine learning tasks.

## Details of the Computation
The script scans a predefined directory structure for OpenMolcas `.RasOrb` plain text files. Utilizing a custom parsing function (`read_orbcoef`), it identifies the `#INFO` and `#ORB` blocks within the file to dynamically compute the formatting blocks. The MO coefficients are read sequentially, cast into float arrays, and transposed to ensure that the columns of the resulting matrix correctly represent the molecular orbitals (C matrix). The extracted matrices are mapped to their respective 4-digit configuration identifiers and retained in memory before batch compression.

## Usage
- **Input:** OpenMolcas orbital files located in specific subdirectories following the pattern: `open/open_g2m05_XXXX/<basis_set>/open_g2m05_XXXX_<basis_set>.RasOrb`.
- **Output:** Compressed `.npz` files (e.g., `all_true_C_matrices_MB.npz` and `all_true_C_matrices_VDZP.npz`) saved in the current working directory. The dictionary keys within the `.npz` file correspond to the 4-digit geometric configuration strings.

## Example
Ensure your working directory is at the same level as the `open` folder containing the dataset. Run the script directly:

```bash
python extract_c_matrices.py
```

## Conclusions
By converting raw quantum chemistry output files into structured, machine-learning-ready `.npz` arrays, this script standardizes the ground-truth electronic structure data. The extracted C matrices serve as high-fidelity targets (True C) for training or validating neural networks (such as EquiFockNet) aimed at predicting Hamiltonian or MO coefficients across mixed or targeted basis sets.
