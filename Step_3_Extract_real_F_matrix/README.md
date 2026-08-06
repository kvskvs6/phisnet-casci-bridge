# Fock Matrix Data Extraction

## Overview
This script automates the batch extraction and aggregation of pre-computed Fock matrices (`fock_real.npy`) from nested directory structures for multiple basis sets (MB and VDZP). It consolidates highly fragmented data into efficiently compressed structures.

## Details of the Computation
The script searches the hierarchical directory structure (`open/open_g2m05_XXXX/<basis_set>/`) for numpy array files containing real-valued Fock matrices. It strictly parses the 4-digit system index (`XXXX`) directly from the parent directory's nomenclature to guarantee unambiguous matrix-to-molecule alignment. The aggregated arrays are subsequently packed into a dictionary-style `.npz` format, indexed by their respective 4-digit string identifiers. 

## Usage
The script is designed to be executed from the root directory containing the `open` dataset folder. 
- **Input:** Reads multiple `fock_real.npy` files from paths matching `open/open_g2m05_XXXX/<basis_set>/`.
- **Output:** Generates compressed `all_true_F_matrices_<basis_set>.npz` files in the current working directory.

## Example
Navigate to the parent directory of `open` and execute the script:
```bash
python extract_fock_matrices.py
```

## Conclusions
This routine significantly streamlines data preprocessing workflows for machine learning applications or downstream quantum chemical analyses [cite: 1]. By consolidating thousands of isolated tensor files into single, highly structured, and compressed `.npz` archives, it ensures robust data loading and rigorously prevents index-mismatch errors during model training or molecular property evaluations.
