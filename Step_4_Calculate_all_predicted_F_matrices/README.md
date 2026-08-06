# Overview

This script acts as an automated inference pipeline for predicting Fock matrices of molecular conformations using pre-trained `PhiSNet` models. It supports both Minimal Basis (MB) and Valence Double-Zeta plus Polarization (VDZP) basis sets. By parsing computational chemistry checkpoints directly, it efficiently reconstructs electronic Hamiltonians and rigorously outputs the results in compressed scientific formats.

## Details of the Computation

- **Data Sourcing & Structural Alignment**: The script continuously parses molecular coordinates from `.xyz` files matching the `open/open_g2m05_XXXX/open_g2m05_XXXX.xyz` directory structure. It evaluates atom configuration consistency across trajectory frames, enforcing the static calculation graph requirement inherent to PhiSNet.
- **Model Construction & Weight Extraction**: The script initializes an empty `CorePhiSNet` architecture, defining orbital mappings according to the chosen basis (MB or VDZP). It loads weights from standard PyTorch Lightning `.ckpt` checkpoints, dynamically stripping `model.` parameter prefixes. Gradient, energy, and force calculations are explicitly disabled to guarantee optimized forward passes for Hamiltonian generation only.
- **Matrix Prediction & Transformation**: The core model infers the Fock matrix initially in the Localized Molecular (LM) basis framework. This tensor is subsequentially transformed back into the Atomic Orbital (AO) basis, followed by enforcing strict Hamiltonian Hermiticity via explicit symmetrization: $F_{sym} = \frac{1}{2}(F + F^T)$.
- **Deterministic Transposition (VDZP specific)**: To resolve discrepancies between neural network internal representations and standard computational software outputs, a reverse reordering algorithm is applied exclusively during VDZP mode. It uses a defined mathematical inverse map (e.g., mapping internal indices `[0, 1, 2, 3, 5, 7...]` back to standard distributions `[0, 1, 2, 3, 6, 4...]`) to restore exact structural compatibility with the standard Molcas ANO-VDZP format.

## Usage

**Input Requirements:**
- Conformation datasets must be mapped to `open/open_g2m05_*/open_g2m05_*.xyz`.
- PyTorch Lightning model checkpoints must reside in the root executing directory, matching the regex-like format: `phisnet_best_model_[SPLIT_SIZES]_[BASIS].ckpt` (e.g., `phisnet_best_model_8_1_1_MB.ckpt`).

**Output Delivery:**
- Standard predicted matrices (MB & VDZP) are aggregated and saved to: 
  `all_predicted_F_matrices_[SPLIT_SIZES]_[BASIS].npz`
- Specifically for the VDZP model, an additional strictly Molcas-aligned data array is generated:
  `all_molcas_predicted_F_matrices_[SPLIT_SIZES]_VDZP.npz`

## Example

Execute the script from the command line without additional arguments. The pipeline relies heavily on auto-discovery of checkpoint files inside the current working directory:

```bash
python predict_fock_matrices.py
```

## Conclusions

Running this script translates fundamental atomic coordinates spanning large dataset trajectories into high-fidelity electronic structure representations (Fock matrices). The batched, serialized `.npz` outputs provide an immediate computational foundation. These arrays facilitate investigations into machine learning model generalization errors, comparative electronic analyses across varying basis set complexities (MB vs. VDZP), or act as highly accurate initial quantum guesses designed to drastically accelerate iterative self-consistent field (SCF) convergences in ab initio packages like OpenMolcas.
